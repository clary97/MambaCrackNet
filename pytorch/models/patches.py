"""Patch extraction / embedding / merging / expanding modules (ViT-style).

Inputs follow PyTorch conventions: images are (B, C, H, W); token sequences are (B, L, C).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExtract(nn.Module):
    """Split an NCHW image into non-overlapping (ph, pw) patches.

    Returns (B, num_patches, C*ph*pw) so that a linear projection can embed each patch.
    """

    def __init__(self, patch_size):
        super().__init__()
        self.ph, self.pw = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.ph == 0 and W % self.pw == 0, (
            f"image size ({H}x{W}) must be divisible by patch size ({self.ph}x{self.pw})"
        )
        x = x.unfold(2, self.ph, self.ph).unfold(3, self.pw, self.pw)
        # (B, C, nH, nW, ph, pw) -> (B, nH, nW, C, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.view(B, -1, C * self.ph * self.pw)


class PatchEmbedding(nn.Module):
    """Linear projection + learned positional embedding."""

    def __init__(self, num_patches: int, patch_dim: int, embed_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Embedding(num_patches, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.num_patches, device=x.device)
        return self.proj(x) + self.pos_embed(positions)


class PatchMerging(nn.Module):
    """Halve the spatial token count and double the channel dim (Swin-style)."""

    def __init__(self, num_patch, embed_dim: int):
        super().__init__()
        self.h, self.w = num_patch
        self.embed_dim = embed_dim
        self.linear = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        assert L == self.h * self.w, f"expected L={self.h * self.w}, got L={L}"
        assert self.h % 2 == 0 and self.w % 2 == 0, "patch grid must be even"

        x = x.view(B, self.h, self.w, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, (self.h // 2) * (self.w // 2), 4 * C)
        return self.linear(x)


class PatchExpanding(nn.Module):
    """Inverse of PatchMerging: spatial up-sampling via pixel shuffle."""

    def __init__(self, num_patch, embed_dim: int, upsample_rate: int, return_vector: bool = True):
        super().__init__()
        self.h, self.w = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.linear = nn.Conv2d(
            embed_dim, upsample_rate * embed_dim, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        assert L == self.h * self.w
        x = x.view(B, self.h, self.w, C).permute(0, 3, 1, 2).contiguous()
        x = self.linear(x)  # (B, ur*C, h, w)
        x = F.pixel_shuffle(x, self.upsample_rate)  # (B, C/ur, h*ur, w*ur)
        if self.return_vector:
            B2, C2, H2, W2 = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B2, H2 * W2, C2)
        return x
