"""Deep sequence models behind one interface.

Every model maps a standardized window ``x (B, L, C)`` to ``(B, H, T)``:
``H`` horizons (one for a virtual-sensor nowcast) of ``T`` targets, in the
targets' standardized units. A model is a *backbone* plus the shared
``SeriesModel`` wrapper, which owns RevIN and the output head:

- backbones that keep one representation per channel (PatchTST
  channel-independent, DLinear, iTransformer) or one per window (PatchTST
  mixing, GRU);
- ``head: channel`` (original PatchTST): each target is forecast from its own
  channel's representation. Needs the target among the inputs.
- ``head: flatten``: all channels' representations feed one linear head.
  Needed for virtual sensors, where the target is not an input.
- RevIN de-normalizes the output with the target channel's window statistics
  when the target is an input; otherwise the per-channel window mean/std are
  projected and appended to the head input so absolute levels are kept.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layers import Encoder, MovingAverage, RevIN


class PatchTSTBackbone(nn.Module):
    """PatchTST (Nie et al., 2023). ``channel_mode='independent'`` is the original;
    ``'mixing'`` flattens all channels of a patch into one token (the previous project's best)."""

    def __init__(self, *, n_channels: int, context: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, n_heads: int = 16, n_layers: int = 3, d_ff: int = 256,
                 dropout: float = 0.05, attn_dropout: float = 0.0, norm: str = "batch",
                 residual_attention: bool = True, channel_mode: str = "independent") -> None:
        super().__init__()
        if channel_mode not in ("independent", "mixing"):
            raise ValueError(f"channel_mode must be 'independent' or 'mixing': {channel_mode}")
        if patch_len > context:
            raise ValueError(f"patch_len {patch_len} is longer than the context {context}")
        self.per_channel = channel_mode == "independent"
        self.patch_len, self.stride = int(patch_len), int(stride)
        self.n_patches = (context + self.stride - self.patch_len) // self.stride + 1
        width = self.patch_len if self.per_channel else self.patch_len * n_channels
        self.proj = nn.Linear(width, d_model)
        self.pos = nn.Parameter(torch.empty(1, self.n_patches, d_model).uniform_(-0.02, 0.02))
        self.drop = nn.Dropout(dropout)
        self.encoder = Encoder(d_model, n_heads, n_layers, d_ff, dropout, attn_dropout, norm, residual_attention)
        self.rep_dim = self.n_patches * d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        z = x.transpose(1, 2)
        z = torch.cat([z, z[..., -1:].expand(-1, -1, self.stride)], dim=-1)  # end padding, as in PatchTST
        patches = z.unfold(-1, self.patch_len, self.stride)  # (B, C, N, P)
        n = patches.shape[2]
        if self.per_channel:
            tokens = self.proj(patches.reshape(b * c, n, self.patch_len))
            hidden = self.encoder(self.drop(tokens + self.pos))
            return hidden.reshape(b, c, -1)
        tokens = self.proj(patches.permute(0, 2, 1, 3).reshape(b, n, c * self.patch_len))
        hidden = self.encoder(self.drop(tokens + self.pos))
        return hidden.reshape(b, -1)


class DLinearBackbone(nn.Module):
    """DLinear (Zeng et al., 2023): trend/seasonal split, one shared linear map each."""

    def __init__(self, *, n_channels: int, context: int, kernel: int = 25, d_rep: int = 64) -> None:
        super().__init__()
        self.per_channel = True
        self.decompose = MovingAverage(kernel)
        self.seasonal = nn.Linear(context, d_rep)
        self.trend = nn.Linear(context, d_rep)
        self.rep_dim = d_rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        trend = self.decompose(z)
        return self.seasonal(z - trend) + self.trend(trend)


class ITransformerBackbone(nn.Module):
    """iTransformer (Liu et al., 2024): each channel's whole window is one token;
    attention runs across channels, which suits cross-sensor estimation."""

    def __init__(self, *, n_channels: int, context: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 2, d_ff: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.per_channel = True
        self.embed = nn.Linear(context, d_model)
        self.drop = nn.Dropout(dropout)
        self.encoder = Encoder(d_model, n_heads, n_layers, d_ff, dropout, 0.0, "layer", False)
        self.rep_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.drop(self.embed(x.transpose(1, 2))))


class GRUBackbone(nn.Module):
    """Recurrent baseline: GRU over time on all channels, last hidden state."""

    def __init__(self, *, n_channels: int, context: int, hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.per_channel = False
        self.inp = nn.Linear(n_channels, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.rep_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, h = self.gru(self.inp(x.float()))
        return h[-1]


BACKBONES: dict[str, type[nn.Module]] = {
    "patchtst": PatchTSTBackbone,
    "dlinear": DLinearBackbone,
    "itransformer": ITransformerBackbone,
    "gru": GRUBackbone,
}
WRAPPER_KEYS = ("revin", "revin_center", "head", "head_dropout")
# Keys given in seconds in configs, converted to samples with the sampling interval.
SECONDS_KEYS = {"patch_seconds": "patch_len", "stride_seconds": "stride", "kernel_seconds": "kernel"}


class SeriesModel(nn.Module):
    def __init__(self, backbone: nn.Module, *, n_channels: int, n_horizons: int, n_targets: int,
                 target_channels: list[int] | None, revin: bool = True, revin_center: str = "mean",
                 head: str = "auto", head_dropout: float = 0.0, stats_dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_horizons, self.n_targets = n_horizons, n_targets
        per_channel = bool(backbone.per_channel)
        if head == "auto":
            head = "channel" if per_channel and target_channels is not None else "flatten"
        if head == "channel" and (not per_channel or target_channels is None):
            raise ValueError("head 'channel' needs a per-channel backbone and the targets among the inputs")
        if head not in ("channel", "flatten"):
            raise ValueError(f"head must be auto, channel or flatten: {head}")
        self.head_kind = head
        self.revin = RevIN(center=revin_center) if revin else None
        self.denorm = bool(revin and target_channels is not None)
        self.inject = bool(revin and target_channels is None)
        self.register_buffer("target_index", torch.as_tensor(target_channels or [], dtype=torch.long))
        self.drop = nn.Dropout(head_dropout)
        if head == "channel":
            self.head = nn.Linear(backbone.rep_dim, n_horizons)
        else:
            width = backbone.rep_dim * (n_channels if per_channel else 1)
            self.stats = None
            if self.inject:
                self.stats = nn.Sequential(nn.Linear(2 * n_channels, stats_dim), nn.GELU())
                width += stats_dim
            self.head = nn.Linear(width, n_horizons * n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if self.revin is not None:
            x, mean, std = self.revin(x)
        rep = self.backbone(x)
        if self.head_kind == "channel":
            out = self.head(self.drop(rep[:, self.target_index, :])).transpose(1, 2)
        else:
            flat = rep.reshape(b, -1)
            if self.inject:
                flat = torch.cat([flat, self.stats(torch.cat([mean, std], dim=-1).reshape(b, -1))], dim=-1)
            out = self.head(self.drop(flat)).view(b, self.n_horizons, self.n_targets)
        if self.denorm:
            out = out * std[:, :, self.target_index] + mean[:, :, self.target_index]
        return out


def resolve_params(params: dict[str, Any], sample_seconds: float) -> dict[str, Any]:
    """Convert ``*_seconds`` keys to sample counts for the chosen sampling interval."""

    out = {}
    for key, value in params.items():
        if key in SECONDS_KEYS:
            out[SECONDS_KEYS[key]] = max(1, round(float(value) / sample_seconds))
        else:
            out[key] = value
    return out


def build_deep_model(name: str, params: dict[str, Any], *, n_channels: int, context: int,
                     n_horizons: int, n_targets: int, target_channels: list[int] | None) -> SeriesModel:
    if name not in BACKBONES:
        raise ValueError(f"unknown deep model {name!r}; choose from {sorted(BACKBONES)}")
    backbone_params = {k: v for k, v in params.items() if k not in WRAPPER_KEYS}
    backbone = BACKBONES[name](n_channels=n_channels, context=context, **backbone_params)
    return SeriesModel(
        backbone,
        n_channels=n_channels,
        n_horizons=n_horizons,
        n_targets=n_targets,
        target_channels=target_channels,
        revin=bool(params.get("revin", True)),
        revin_center=str(params.get("revin_center", "mean")),
        head=str(params.get("head", "auto")),
        head_dropout=float(params.get("head_dropout", 0.0)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
