from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F

from sensori.sensori import InfoNCELoss, Sensori, cosine_lr
from sensori.transforms import ChannelSwap, Scale


class SensoriEmbeddingModule(L.LightningModule):
    """Extract and save ordered day embeddings through distributed validation."""

    def __init__(self, model=None, embedding_paths=()):
        super().__init__()
        self.model = model
        self.embedding_paths = [Path(path) for path in embedding_paths]

    def on_validation_epoch_start(self):
        dataloader_count = len(self.trainer.val_dataloaders)
        self.embedding_batches = [[] for _ in range(dataloader_count)]
        self.embedding_keys = [[] for _ in range(dataloader_count)]
        self.embedding_save_paths = []

        for path_template in self.embedding_paths:
            save_path = str(path_template).format(epoch=self.current_epoch, step=self.global_step)
            save_path = self.trainer.strategy.broadcast(save_path if self.trainer.is_global_zero else None)
            save_path = Path(save_path).expanduser().resolve()
            if self.trainer.is_global_zero:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            self.embedding_save_paths.append(save_path)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        embeddings = self.model(batch['data'], output='embedding')
        keys = torch.stack((batch['participant_index'], batch['day_id']), dim=1)

        if self.trainer.world_size > 1:
            embeddings = self.all_gather(embeddings, sync_grads=False).flatten(0, 1)
            keys = self.all_gather(keys, sync_grads=False).flatten(0, 1)

        if self.trainer.is_global_zero:
            self.embedding_batches[dataloader_idx].append(embeddings.float().cpu().numpy())
            self.embedding_keys[dataloader_idx].append(keys.cpu().numpy())

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            for save_path, embedding_batches, key_batches, dataloader in zip(
                self.embedding_save_paths,
                self.embedding_batches,
                self.embedding_keys,
                self.trainer.val_dataloaders,
            ):
                embeddings = np.concatenate(embedding_batches)
                keys = np.concatenate(key_batches)
                order = np.lexsort((keys[:, 1], keys[:, 0]))
                embeddings = embeddings[order]
                keys = keys[order]

                keep = np.concatenate(([True], np.any(keys[1:] != keys[:-1], axis=1)))
                embeddings = embeddings[keep]
                participant_indices = keys[keep, 0]

                boundaries = np.flatnonzero(participant_indices[1:] != participant_indices[:-1]) + 1
                starts = np.concatenate(([0], boundaries))
                ends = np.concatenate((boundaries, [len(participant_indices)]))
                participant_ids = dataloader.dataset.participant_ids
                grouped = {
                    participant_ids[participant_indices[start]]: embeddings[start:end].copy()
                    for start, end in zip(starts, ends)
                }
                np.save(save_path, grouped)
                print(f'Saved embeddings to: {save_path}')

        self.trainer.strategy.barrier('embeddings saved')
        self.embedding_batches.clear()
        self.embedding_keys.clear()
        self.embedding_save_paths.clear()


class SensoriPretrainingModule(SensoriEmbeddingModule):
    def __init__(self, cfg, embedding_paths=()):
        super().__init__(embedding_paths=embedding_paths)

        # Validate all pretraining-only settings together. Inference builds
        # Sensori directly and therefore does not need these loss parameters.
        if cfg.logit_temp <= 0 or cfg.masked_logit_temp <= 0 or cfg.movement_logit_temp <= 0:
            raise ValueError('All loss temperatures must be positive.')
        if cfg.n_negatives < 0 or cfg.cross_sample_negatives < 0:
            raise ValueError('Negative sample counts must be non-negative.')
        if cfg.n_negatives + cfg.cross_sample_negatives == 0:
            raise ValueError('Masked pretraining requires at least one negative sample.')
        if not 0 < cfg.mask_feature_prob <= 1:
            raise ValueError(f'mask_feature_prob must be in (0, 1], got {cfg.mask_feature_prob}.')
        if cfg.mask_feature_length < 1:
            raise ValueError(f'mask_feature_length must be positive, got {cfg.mask_feature_length}.')
        if cfg.movement_rank_weight < 0 or cfg.contrastive_weight < 0:
            raise ValueError('Loss weights must be non-negative.')
        if cfg.movement_rank_weight + cfg.contrastive_weight > 1:
            raise ValueError('movement_rank_weight + contrastive_weight must not exceed 1.')
        if cfg.train_data.augmentation.axis_scale_std < 0:
            raise ValueError('axis_scale_std must be non-negative.')

        self.cfg = cfg
        self.contrastive_loss_fn = InfoNCELoss(temp=cfg.logit_temp)
        augmentation = cfg.train_data.augmentation
        self.augmentation = (
            torch.nn.Sequential(Scale(augmentation.axis_scale_std), ChannelSwap()) if augmentation.enabled else None
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.training and self.augmentation is not None and torch.is_tensor(batch):
            return self.augmentation(batch)
        return batch

    def configure_model(self):
        if self.model is not None:
            return

        self.model = Sensori(self.cfg)

        if self.cfg.model_compile:
            self.model.feat_encoder.compile(dynamic=False)

    def _compute_contrastive_loss_gate(self) -> float:
        start_epoch = int(getattr(self.cfg, 'contrastive_loss_start_epoch', 0))
        warmup_epochs = int(getattr(self.cfg, 'contrastive_loss_warmup_epochs', 0))
        current_epoch = int(self.current_epoch)

        if current_epoch < start_epoch:
            return 0.0

        if warmup_epochs > 0:
            epochs_since_start = current_epoch - start_epoch
            if epochs_since_start < warmup_epochs:
                return float(epochs_since_start + 1) / float(warmup_epochs)

        return 1.0

    def training_step(self, batch):
        contrastive_loss_gate = self._compute_contrastive_loss_gate()
        contrastive_logits, masked_logits, rank_pair_logits = self.model(
            batch,
            compute_contrastive=contrastive_loss_gate > 0,
            compute_rank=self.cfg.movement_rank_weight > 0,
        )

        flat_masked_logits = masked_logits.reshape(-1, masked_logits.size(-1))
        masked_targets = flat_masked_logits.new_zeros((flat_masked_logits.size(0),), dtype=torch.long)
        masked_loss = F.cross_entropy(flat_masked_logits, masked_targets)
        masking_accuracy = flat_masked_logits.argmax(dim=-1).eq(0).float().mean()
        zero = masked_loss.new_zeros(())

        movement_rank_loss = zero
        if rank_pair_logits is not None:
            rank_pair_targets = rank_pair_logits.new_zeros((rank_pair_logits.size(0),), dtype=torch.long)
            movement_rank_loss = F.cross_entropy(rank_pair_logits / self.cfg.movement_logit_temp, rank_pair_targets)

        contrastive_loss = zero
        contrastive_metrics = {name: zero for name in ('acc_top1', 'acc_top5', 'acc_mean_pos')}
        if contrastive_logits is not None:
            if self.trainer.world_size > 1:
                gathered = self.all_gather(contrastive_logits, sync_grads=True)
                half = gathered.size(1) // 2
                contrastive_logits = torch.cat(
                    (
                        gathered[:, :half].flatten(0, 1),
                        gathered[:, half:].flatten(0, 1),
                    )
                )

            contrastive_loss, contrastive_metrics = self.contrastive_loss_fn(contrastive_logits)

        contrastive_w = self.cfg.contrastive_weight * contrastive_loss_gate
        masked_w = 1.0 - self.cfg.movement_rank_weight - contrastive_w
        total_loss = (
            masked_loss * masked_w
            + movement_rank_loss * self.cfg.movement_rank_weight
            + contrastive_loss * contrastive_w
        )

        metrics = {
            'train/total_loss': total_loss,
            'train/contrastive_loss': contrastive_loss,
            'train/contrastive_loss_gate': total_loss.new_tensor(contrastive_loss_gate),
            'train/masking_loss': masked_loss,
            'train/masking_accuracy': masking_accuracy,
            'train/acc_top1': contrastive_metrics['acc_top1'],
            'train/acc_top5': contrastive_metrics['acc_top5'],
            'train/acc_mean_pos': contrastive_metrics['acc_mean_pos'],
            'train/movement_rank_loss': movement_rank_loss,
        }
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch.size(0),
        )

        return total_loss

    def on_before_optimizer_step(self, _optimizer):
        # Lightning has unscaled mixed-precision gradients at this point.
        # Clip here because its precision plugin rejects fused-optimizer clipping.
        parameters = [parameter for parameter in self.model.parameters() if parameter.grad is not None]
        if not parameters:
            return

        # clip_grad_norm_ returns the total norm before clipping, preserving the
        # meaning of the logged value while supporting fused AdamW under BF16.
        total_grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        self.log(
            'train/grad_norm',
            total_grad_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=1e-2,
            fused=self.device.type == 'cuda' and self.trainer.precision != '16-mixed',
        )

        if self.cfg.lr_scheduler == 'cosine':
            lr_scheduler = {
                'scheduler': cosine_lr(optimizer, self.cfg.warmup_length, self.cfg.lr_t0),
                'name': 'learning_rate',
                'interval': 'step',
            }
            return [optimizer], [lr_scheduler]

        return optimizer
