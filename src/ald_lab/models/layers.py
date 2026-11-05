"""Building blocks shared by the deep models."""

from __future__ import annotations

import math

import torch
from torch import nn


class RevIN(nn.Module):
    """Per-window instance normalization (Kim et al., 2022), without affine weights.

    Returns the normalized window and the statistics so a model can either
    de-normalize its forecast (when the target is one of its inputs) or feed
    the statistics back in (when it is not: otherwise the absolute level of
    every input would be erased, which broke the previous project's virtual
    sensor).

    ``center='last'`` (PatchTST's ``subtract_last``) centres each window on its
    last value instead of its mean, so an untrained forecaster starts from
    "the value stays as it is" (persistence) rather than the window mean.
    """

    def __init__(self, eps: float = 1e-5, center: str = "mean") -> None:
        super().__init__()
        if center not in ("mean", "last"):
            raise ValueError(f"RevIN center must be 'mean' or 'last': {center}")
        self.eps = eps
        self.center = center

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x[:, -1:, :] if self.center == "last" else x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        return (x - mean) / std, mean, std


class _BatchNormTokens(nn.Module):
    """BatchNorm over the feature axis of ``(B, N, D)`` tokens (PatchTST default)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _norm(kind: str, d_model: int) -> nn.Module:
    if kind == "batch":
        return _BatchNormTokens(d_model)
    if kind == "layer":
        return nn.LayerNorm(d_model)
    raise ValueError(f"norm must be 'batch' or 'layer': {kind}")


class Attention(nn.Module):
    """Multi-head self-attention that can pass raw scores to the next layer (residual attention)."""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if prev is not None:
            scores = scores + prev
        attn = self.drop(torch.softmax(scores.float(), dim=-1).to(v.dtype))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        return self.out(out), scores


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float,
                 norm: str) -> None:
        super().__init__()
        self.attn = Attention(d_model, n_heads, attn_dropout)
        self.norm1 = _norm(norm, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm2 = _norm(norm, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attn(x, prev)
        x = self.norm1(x + self.drop(attended))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, scores


class Encoder(nn.Module):
    """Post-norm Transformer encoder; ``residual_attention`` adds each layer's scores to the next."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float,
                 attn_dropout: float = 0.0, norm: str = "batch", residual_attention: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout, norm) for _ in range(n_layers)
        )
        self.residual_attention = residual_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = None
        for layer in self.layers:
            x, scores = layer(x, prev)
            prev = scores if self.residual_attention else None
        return x


class MovingAverage(nn.Module):
    """Trend extraction for DLinear: centred moving average with edge replication."""

    def __init__(self, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel_size=kernel, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        front = x[..., :1].expand(-1, -1, (self.kernel - 1) // 2)
        back = x[..., -1:].expand(-1, -1, self.kernel // 2)
        return self.pool(torch.cat([front, x, back], dim=-1))
