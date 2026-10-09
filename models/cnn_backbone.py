"""
VOXEL Module 3: CNN Backbone - XNANet (Self-Attention CNN)
Extracts localized spatial edges, fine textures, and spectral details
from selected/masked hyperspectral satellite channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSelfAttention(nn.Module):
    """
    Self-Attention module capturing non-local spatial context
    and cross-feature edge relationships.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # Query: (B, C//8, N) where N = H*W
        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        # Key: (B, C//8, N)
        proj_key = self.key_conv(x).view(b, -1, h * w)
        # Energy: (B, N, N)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        # Value: (B, C, N)
        proj_value = self.value_conv(x).view(b, -1, h * w)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with batch normalization and GELU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class XNANet(nn.Module):
    """
    XNANet: Self-Attention Augmented CNN Backbone for hyperspectral feature extraction.
    Specialized for localized spatial edges, spectral gradients, and fine textures.
    Outputs a 1024-dimensional feature vector.
    """

    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = 1024,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Multi-scale feature stages
        self.stage1 = ResidualConvBlock(64, 128, stride=2)    # 256 -> 128
        self.stage2 = ResidualConvBlock(128, 256, stride=2)   # 128 -> 64
        self.attention = SpatialSelfAttention(256)             # Self-attention at 64x64
        self.stage3 = ResidualConvBlock(256, 512, stride=2)   # 64 -> 32
        self.stage4 = ResidualConvBlock(512, 512, stride=2)   # 32 -> 16

        # Edge refinement branch (dilated convolutions)
        self.edge_refine = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # Global pooling and feature projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C_in, H, W)
        Returns:
            features: Localized feature vector of shape (B, 1024)
        """
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.attention(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.edge_refine(h)
        pooled = self.global_pool(h)
        out = self.fc(pooled)
        return out


if __name__ == "__main__":
    model = XNANet(in_channels=6, feature_dim=1024)
    x = torch.randn(2, 6, 256, 256)
    out = model(x)
    print(f"XNANet Input: {x.shape} -> Feature Output: {out.shape}")
    assert out.shape == (2, 1024), f"Expected (2, 1024), got {out.shape}"
    print("XNANet sanity check passed!")
