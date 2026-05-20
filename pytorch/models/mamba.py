"""Mamba selective state space modules.

PyTorch port of the original TensorFlow blocks:
    RMSNorm, selective_scan, SSM, MambaBlock, ResidualBlock (renamed to MambaResidualBlock).
Sequences are shaped (B, L, C) throughout.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stddev = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True)).clamp(min=self.eps)
        return x / stddev * self.weight


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Parallel-prefix selective scan, faithful to the TF reference."""
    dA = torch.einsum("bld,dn->bldn", delta, A)
    dB_u = torch.einsum("bld,bld,bln->bldn", delta, u, B)

    # Shift along sequence dim: replicate TF's tf.pad(dA[:, 1:], [[0,0],[1,1],[0,0],[0,0]])[:, 1:]
    dA_shifted = F.pad(dA[:, 1:], (0, 0, 0, 0, 1, 1))[:, 1:, :, :]
    dA_shifted = torch.flip(dA_shifted, dims=[1])
    dA_cumsum = torch.cumsum(dA_shifted, dim=1)
    dA_cumsum = torch.exp(dA_cumsum)
    dA_cumsum = torch.flip(dA_cumsum, dims=[1])

    x = dB_u * dA_cumsum
    x = torch.cumsum(x, dim=1) / (dA_cumsum + 1e-12)
    y = torch.einsum("bldn,bln->bld", x, C)
    return y + u * D


class SSM(nn.Module):
    def __init__(self, d_model: int, expand: int = 2, d_state: int = 16, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        d_inner = d_model * expand

        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=bias)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # A initialised as -log(1..N) duplicated per inner channel (matches reference)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_inner)
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log)
        return selective_scan(x, delta, A, B, C, self.D)


class MambaBlock(nn.Module):
    """Selective-SSM block with SwiGLU-style gating."""

    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        bias: bool = False,
        d_conv: int = 4,
        conv_bias: bool = True,
        d_state: int = 16,
    ):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=bias)
        self.conv = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding="same",
            bias=conv_bias,
        )
        self.ssm = SSM(d_model, expand=expand, d_state=d_state, bias=bias)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x_and_res = self.in_proj(x)
        x, res = x_and_res.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)


class MambaResidualBlock(nn.Module):
    """Multi-path residual wrapper around `MambaBlock` (Mamba + depthwise convs + skip)."""

    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        bias: bool = False,
        d_conv: int = 4,
        conv_bias: bool = True,
        d_state: int = 16,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        # depthwise == groups=channels
        self.dw_conv5 = nn.Conv1d(d_model, d_model, 5, padding="same", groups=d_model)
        self.dw_conv3 = nn.Conv1d(d_model, d_model, 3, padding="same", groups=d_model)
        self.mamba = MambaBlock(
            d_model,
            expand=expand,
            bias=bias,
            d_conv=d_conv,
            conv_bias=conv_bias,
            d_state=d_state,
        )
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x_ch = x.transpose(1, 2)  # (B, C, L)
        path3 = F.silu(self.dw_conv5(x_ch))

        normed = self.norm(x)
        path2 = F.silu(self.dw_conv3(normed.transpose(1, 2)))

        mamba_out = self.mamba(normed)  # (B, L, C)

        merged = mamba_out + x + path3.transpose(1, 2) + path2.transpose(1, 2)
        merged = self.proj(merged.transpose(1, 2)).transpose(1, 2)
        return merged
