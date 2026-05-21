"""Image-domain helper blocks shared by the encoder/decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dims, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvResidualBlock(nn.Module):
    """Pre-activation residual block on NCHW feature maps with optional up/down sampling."""

    def __init__(self, in_ch: int, out_ch: int, downpool: bool = False, uppool: bool = False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # Mirrors the TF reference: concat([conv2_out, input_x]) -> 3x3 conv to out_ch
        self.conv3 = nn.Conv2d(out_ch + in_ch, out_ch, kernel_size=3, padding=1)
        self.downpool = downpool
        self.uppool = uppool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = torch.cat([out, x], dim=1)
        out = self.conv3(out)
        if self.downpool:
            out = F.max_pool2d(out, kernel_size=2, stride=2)
        if self.uppool:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        return out
