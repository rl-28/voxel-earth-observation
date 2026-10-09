"""
VOXEL Module 3: Vision Transformer Backbone - Tiny-32 ViT
Utilizes timm to extract long-range global spatial dependencies and contextual
spectral relations across satellite imagery patches.
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class Tiny32ViT(nn.Module):
    """
    Compact Tiny-32 Vision Transformer (ViT) via timm.
    Features:
      - Patch size: 32x32 (yields 8x8 = 64 spatial patches for 256x256 input)
      - Embedding dimension: 192 (compact Tiny ViT)
      - Multi-head self-attention with 3 heads (dim 64/head)
      - Depth: 4 Transformer blocks for low latency and < 16 GB GPU VRAM footprint
      - Outputs a 192-dimensional global context embedding vector
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 32,
        in_chans: int = 6,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Native timm VisionTransformer configured as a compact Tiny-32 backbone
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # Pooling mode: outputs the global representation embedding
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            global_pool="token",  # Pool CLS token
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, in_chans, H, W)
        Returns:
            features: Global context embedding vector of shape (B, 192)
        """
        feat = self.vit(x)
        feat = self.norm(feat)
        return feat


if __name__ == "__main__":
    vit_model = Tiny32ViT(img_size=256, patch_size=32, in_chans=6, embed_dim=192)
    x = torch.randn(2, 6, 256, 256)
    out = vit_model(x)
    print(f"Tiny32ViT Input: {x.shape} -> Embedding Output: {out.shape}")
    assert out.shape == (2, 192), f"Expected (2, 192), got {out.shape}"
    print("Tiny32ViT sanity check passed!")
