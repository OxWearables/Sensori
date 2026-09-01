"""Convolutional and Transformer components for Sensori.

The convolutional feature extractor follows torchaudio's wav2vec2 design, and
the positional encoder follows fairseq's wav2vec design.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class Conv1DEncoder(nn.Module):
    """Encode channel-last sensor windows with a configurable 1D convolution stack."""

    def __init__(
        self,
        in_channels: int,
        norm_mode: str,
        conv_layer_config: Sequence[Sequence[int]],
        conv_bias: bool,
    ):
        super().__init__()
        if norm_mode not in {'group_norm', 'layer_norm'}:
            raise ValueError(f'Unsupported norm_mode={norm_mode!r}; expected "group_norm" or "layer_norm".')

        self.conv_layers = nn.ModuleList()
        for index, (out_channels, kernel_size, stride) in enumerate(conv_layer_config):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=conv_bias,
            )
            nn.init.kaiming_normal_(conv.weight)

            if norm_mode == 'layer_norm':
                norm = nn.LayerNorm(out_channels)
            elif index == 0:
                norm = nn.GroupNorm(
                    num_groups=out_channels,
                    num_channels=out_channels,
                    affine=True,
                )
            else:
                norm = nn.Identity()

            # Keep these names and their registration order compatible with released weights.
            self.conv_layers.append(nn.ModuleDict({'layer_norm': norm, 'conv': conv}))
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected a 3D input (batch, time, channel), found {list(x.shape)}.')

        x = x.transpose(1, 2)  # B x C x T
        for layer in self.conv_layers:
            x = layer['conv'](x)
            norm = layer['layer_norm']
            if isinstance(norm, nn.LayerNorm):
                # LayerNorm operates on the last dimension, so expose channels there.
                x = norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = norm(x)
            x = F.gelu(x)

        return x.transpose(1, 2)  # B x T x C


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.positional_embedding = cfg.positional_embedding
        self.dropout = cfg.dropout
        encoder_dim = cfg.encoder_dim

        if self.positional_embedding == 'relative':
            self.pos_conv = nn.Conv1d(
                encoder_dim,
                encoder_dim,
                kernel_size=cfg.conv_pos,
                padding=cfg.conv_pos // 2,
                groups=cfg.conv_pos_groups,
            )
            std = math.sqrt(4 / (cfg.conv_pos * encoder_dim))
            nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
            nn.init.constant_(self.pos_conv.bias, 0)

            self.pos_conv = weight_norm(self.pos_conv, name='weight', dim=2)
        elif self.positional_embedding == 'absolute':
            max_len = 2000
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, encoder_dim, 2) * (-math.log(10000.0) / encoder_dim))
            positional_encoding = torch.zeros(max_len, 1, encoder_dim)
            positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
            positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('positional_encoding', positional_encoding)
        else:
            raise ValueError(
                f'Unsupported positional_embedding={self.positional_embedding!r}; expected "relative" or "absolute".'
            )

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=encoder_dim,
                    nhead=cfg.encoder_attention_heads,
                    dim_feedforward=cfg.encoder_ffn_embed_dim,
                    dropout=self.dropout,
                    activation=cfg.activation_fn,
                    batch_first=True,
                    norm_first=True,
                    bias=True,
                )
                for _ in range(cfg.encoder_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.layerdrop = cfg.encoder_layerdrop

    def forward(self, x):
        if self.positional_embedding == 'relative':
            x_conv = self.pos_conv(x.transpose(1, 2))[..., : x.size(1)]
            x_conv = F.gelu(x_conv)
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = x + self.positional_encoding[: x.size(1)].transpose(0, 1)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            # LayerDrop skips whole encoder layers during training only.
            if self.training and self.layerdrop > 0 and torch.rand(()) < self.layerdrop:
                continue
            x = layer(x)

        return self.layer_norm(x)
