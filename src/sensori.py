import math
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from sensori.components import Conv1DEncoder, TransformerEncoder


def compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    *,
    device: torch.device,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample overlapping fixed-length spans and return dense token indices.

    Span starts are unique. Because expanded spans may overlap, each row is
    randomly trimmed at token level to the batch-minimum count.
    """
    batch_size, token_count = shape
    if mask_length > token_count:
        raise ValueError(f'mask_length={mask_length} exceeds the token count {token_count}.')

    num_starts = token_count - mask_length + 1
    num_spans = min(
        num_starts,
        max(1, math.floor(mask_prob * token_count / mask_length + 0.5)),
    )
    if sample_weights is None:
        start_indices = torch.rand((batch_size, num_starts), device=device).topk(num_spans, dim=1, sorted=False).indices
    else:
        weights = sample_weights.detach().float()[:, :num_starts].add(1e-6)
        start_indices = torch.multinomial(weights, num_spans, replacement=False)

    span_indices = start_indices.unsqueeze(-1) + torch.arange(mask_length, device=device)
    mask = torch.zeros((batch_size, token_count), dtype=torch.bool, device=device)
    mask.scatter_(1, span_indices.flatten(1), True)

    # Overlaps give rows different counts; trim them for the dense loss.
    target_count = int(mask.sum(dim=1).amin().item())
    keep_order = (
        torch.rand((batch_size, token_count), device=device)
        .masked_fill(~mask, -torch.inf)
        .topk(min(token_count, num_spans * mask_length), dim=1)
        .indices
    )
    return keep_order[:, :target_count].sort(dim=1).values


def cosine_lr(optimizer, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            assert es > 0, 'Learning rate t0 must be greater than warmup length'
            lr = 0.5 * (1 + math.cos(math.pi * e / es))
        return lr

    return LambdaLR(optimizer, lr_lambda=_lr_adjuster)


class InfoNCELoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, x):
        batch_size = x.size(0)
        positive_indices = torch.arange(batch_size, device=x.device).roll(batch_size // 2)

        # Normalize once and use a matrix multiplication instead of broadcasting all
        # feature pairs. Matmul is an autocast-down op, so under mixed precision it
        # would recast these FP32 inputs to BF16; disable autocast to keep the
        # similarities and their softmax in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized_x = F.normalize(x.float(), dim=-1, eps=1e-8)
            logits = normalized_x @ normalized_x.T
            logits = logits / self.temp
            logits.fill_diagonal_(float('-inf'))
            nll = F.cross_entropy(logits, positive_indices)

        with torch.no_grad():
            positive_logits = logits.gather(1, positive_indices.unsqueeze(1))
            positive_ranks = (logits > positive_logits).sum(dim=-1)
            metrics = {
                'acc_top1': (positive_ranks == 0).float().mean(),
                'acc_top5': (positive_ranks < 5).float().mean(),
                'acc_mean_pos': 1 + positive_ranks.float().mean(),
            }

        return nll, metrics


class Sensori(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.window_len = int(cfg.window_len)
        if self.window_len < 1:
            raise ValueError(f'model.window_len must be positive, got {self.window_len}')

        encoder_dim = cfg.encoder_dim

        self.layer_norm = nn.LayerNorm(encoder_dim)

        self.feat_encoder = Conv1DEncoder(
            in_channels=cfg.in_channels,
            norm_mode=cfg.norm_mode,
            conv_layer_config=cfg.conv_layer_config,
            conv_bias=cfg.conv_bias,
        )

        self.post_feat_projection = nn.Linear(cfg.conv_layer_config[-1][0], encoder_dim)

        self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_dim).uniform_())

        self.encoder = TransformerEncoder(cfg)

        self.masked_projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.ReLU(), nn.Linear(encoder_dim, encoder_dim)
        )
        self.rank_projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.ReLU(), nn.Linear(encoder_dim, encoder_dim)
        )

        self.contrastive_projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim),
        )

    def sample_negatives(self, targets):
        local_count = self.cfg.n_negatives
        cross_count = self.cfg.cross_sample_negatives
        batch_size, token_count, _ = targets.shape
        device = targets.device
        target_positions = torch.arange(token_count, device=device).view(1, token_count, 1)
        batch_offsets = torch.arange(batch_size, device=device).view(batch_size, 1, 1) * token_count

        def sample_excluding_positive(count, pool_size, positive_indices):
            indices = torch.randint(pool_size - 1, (batch_size, token_count, count), device=device)
            return indices + (indices >= positive_indices)

        negative_groups = []
        if local_count:
            negative_groups.append(
                sample_excluding_positive(local_count, token_count, target_positions) + batch_offsets
            )
        if cross_count:
            negative_groups.append(
                sample_excluding_positive(cross_count, batch_size * token_count, target_positions + batch_offsets)
            )

        return negative_groups[0] if len(negative_groups) == 1 else torch.cat(negative_groups, dim=-1)

    def compute_masked_logits(self, predictions, targets, negative_indices):
        batch_size, target_count, feature_size = targets.shape

        # Keep masked logits in FP32 under mixed precision. Autocast would
        # otherwise cast the matrix multiplication back to BF16.
        with torch.autocast(device_type=predictions.device.type, enabled=False):
            normalized_predictions = F.normalize(predictions.float(), dim=-1, eps=1e-8)
            normalized_targets = F.normalize(targets.float(), dim=-1, eps=1e-8)
            positive_indices = torch.arange(batch_size * target_count, device=targets.device).view(
                batch_size, target_count, 1
            )
            candidate_indices = torch.cat((positive_indices, negative_indices), dim=-1)

            if self.cfg.cross_sample_negatives > 0:
                flat_predictions = normalized_predictions.reshape(-1, feature_size)
                flat_targets = normalized_targets.reshape(-1, feature_size)
                candidate_indices = candidate_indices.reshape(-1, candidate_indices.size(-1))

                # Bound the temporary source-to-global similarity matrix while
                # retaining efficient matrix multiplications.
                logit_chunks = []
                for start in range(0, flat_predictions.size(0), 4096):
                    similarities = flat_predictions[start : start + 4096] @ flat_targets.T
                    logit_chunks.append(similarities.gather(1, candidate_indices[start : start + 4096]))
                logits = torch.cat(logit_chunks).view(batch_size, target_count, -1)
            else:
                similarities = torch.bmm(normalized_predictions, normalized_targets.transpose(1, 2))
                batch_offsets = torch.arange(batch_size, device=targets.device).view(-1, 1, 1) * target_count
                logits = similarities.gather(2, candidate_indices - batch_offsets)

        return logits / self.cfg.masked_logit_temp

    def _compute_rank_pair_logits(self, clean_tokens, bin_ids):
        # Runs on pre-masking tokens: every bin is guaranteed non-empty per person per day.
        P = clean_tokens.size(0) // 2
        if P < 2:
            return None

        bin_start = int(self.cfg.discard_last_bin)
        bin_num = self.cfg.percentage_bin_num

        day1_tok, day2_tok = clean_tokens[:P], clean_tokens[P:]
        day1_bins, day2_bins = bin_ids[:P], bin_ids[P:]

        def sample_token(tok, tok_bins, p, b):
            cands = tok[p, tok_bins[p] == b]
            return cands[torch.randint(cands.size(0), (1,), device=clean_tokens.device)]

        logits = []
        for p in range(P):
            b = np.random.randint(bin_start, bin_num)
            neg_people = [q for q in range(P) if q != p]

            anchor = sample_token(day1_tok, day1_bins, p, b)
            positive = sample_token(day2_tok, day2_bins, p, b)
            negatives = [
                sample_token(*((day1_tok, day1_bins) if np.random.rand() < 0.5 else (day2_tok, day2_bins)), q, b)
                for q in neg_people
            ]

            logits.append(F.cosine_similarity(anchor, torch.cat([positive] + negatives, dim=0), dim=-1))

        return torch.stack(logits, dim=0)

    def forward(
        self,
        x,
        output: Literal['pretraining', 'embedding', 'conv_features'] = 'pretraining',
        compute_contrastive=True,
        compute_rank=True,
    ):
        if output not in {'pretraining', 'embedding', 'conv_features'}:
            raise ValueError(f'Unsupported output={output!r}.')

        B, T, W, C = x.shape
        features = self.feat_encoder(x.reshape(B * T, W, C))

        if output == 'conv_features':
            return features
        if features.size(1) != 1:
            raise ValueError(
                f'The convolutional encoder must return one vector per input window, got {features.size(1)}.'
            )

        features = features.squeeze(1).reshape(B, -1, self.window_len, features.size(-1)).mean(dim=2)
        features = self.layer_norm(self.post_feat_projection(features))

        if output == 'embedding':
            return self.encoder(features).mean(dim=1)

        window_weights = None
        if compute_rank or self.cfg.weighted_masking:
            with torch.no_grad():
                window_weights = torch.norm(x, dim=-1).std(dim=-1).reshape(B, -1, self.window_len).mean(dim=-1)

        rank_pair_logits = None
        if compute_rank:
            projected_tokens = self.rank_projection(features)
            bin_num = self.cfg.percentage_bin_num
            ranks = window_weights.argsort(dim=1).argsort(dim=1)
            bin_ids = (ranks * bin_num // window_weights.size(1)).clamp(max=bin_num - 1)
            rank_pair_logits = self._compute_rank_pair_logits(projected_tokens, bin_ids)

        masked_token_indices = compute_mask_indices(
            features.shape[:2],
            self.cfg.mask_feature_prob,
            self.cfg.mask_feature_length,
            device=features.device,
            sample_weights=window_weights if self.cfg.weighted_masking else None,
        )
        feature_indices = masked_token_indices.unsqueeze(-1).expand(-1, -1, features.size(-1))
        masked_targets = features.gather(1, feature_indices)
        mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=features.device)
        mask.scatter_(1, masked_token_indices, True)
        x = torch.where(mask.unsqueeze(-1), self.mask_emb, features)
        x = self.encoder(x)

        negative_indices = self.sample_negatives(masked_targets)

        masked_predictions = self.masked_projection(x.gather(1, feature_indices))
        masked_logits = self.compute_masked_logits(masked_predictions, masked_targets, negative_indices)

        day_embedding = None
        if compute_contrastive:
            day_embedding = self.contrastive_projection(x.mean(dim=1))
        return day_embedding, masked_logits, rank_pair_logits


def load_sensori(checkpoint_path, cfg, device='cpu'):
    """Load bare weights or a Lightning checkpoint for inference."""
    model = Sensori(cfg)

    if checkpoint_path is not None:
        weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = weights.get('state_dict', weights)
        state_dict = {
            '.'.join(part for part in name.split('.') if part != '_orig_mod').removeprefix('model.'): tensor
            for name, tensor in state_dict.items()
        }
        model.load_state_dict(state_dict)

    return model.to(device).eval()


def load_pretrained_sensori(device='cpu', *, local_files_only=False):
    """Download and load the released Sensori model and configuration."""
    checkpoint_dir = Path(__file__).resolve().parents[1] / 'checkpoint'
    snapshot_download(
        repo_id='light156/Sensori',
        allow_patterns=['config_model.yaml', 'model.pt'],
        local_dir=checkpoint_dir,
        local_files_only=local_files_only,
    )
    cfg = OmegaConf.load(checkpoint_dir / 'config_model.yaml')
    return load_sensori(checkpoint_dir / 'model.pt', cfg, device)
