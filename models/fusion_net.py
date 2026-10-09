"""
VOXEL Module 3: Hybrid CNN-ViT Feature Fusion Network (FusionNet)
Combines localized edge/texture features from XNANet (1024-dim) with
long-range global contextual embeddings from Tiny-32 ViT (192-dim) into
a uniform 1216-dimensional representation vector for classification and vector retrieval.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from models.cnn_backbone import XNANet
from models.vit_transformer import Tiny32ViT


class FusionNet(nn.Module):
    """
    Hybrid CNN-ViT Fusion Network for Sentinel-2 Land Cover Classification.
    Features:
      - 1D Binary Mask slicing: extracts only selected spectral bands from (B, 13, H, W).
      - Dual parallel branches: XNANet (CNN) + Tiny-32 (ViT).
      - Depth-wise feature concatenation: 1024-dim (CNN) + 192-dim (ViT) = 1216-dim embedding.
      - Classification head: Multi-layer perceptron outputting land cover logits.
    """

    def __init__(
        self,
        num_classes: int = 10,
        total_bands: int = 13,
        selected_channels: int = 6,
        cnn_feature_dim: int = 1024,
        vit_feature_dim: int = 192,
        img_size: int = 256,
        patch_size: int = 32,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.total_bands = total_bands
        self.selected_channels = selected_channels
        self.cnn_feature_dim = cnn_feature_dim
        self.vit_feature_dim = vit_feature_dim
        self.embedding_dim = cnn_feature_dim + vit_feature_dim  # Exactly 1216

        # Default fallback band mask (e.g. Bands 2, 3, 4, 8, 11, 12)
        default_mask = torch.zeros(total_bands, dtype=torch.float32)
        default_indices = [1, 2, 3, 7, 11, 12][:selected_channels]
        default_mask[default_indices] = 1.0
        self.register_buffer("default_mask", default_mask)

        # Dynamic channel adapter (maps any number of active channels to selected_channels)
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(selected_channels, selected_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(selected_channels),
            nn.GELU(),
        )

        # Dual parallel backbones
        self.cnn_backbone = XNANet(
            in_channels=selected_channels,
            feature_dim=cnn_feature_dim,
        )

        self.vit_backbone = Tiny32ViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=selected_channels,
            embed_dim=vit_feature_dim,
        )

        # Feature normalization for balanced fusion
        self.cnn_norm = nn.LayerNorm(cnn_feature_dim)
        self.vit_norm = nn.LayerNorm(vit_feature_dim)

        # Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(classifier_dropout / 2),
            nn.Linear(128, num_classes),
        )

    def slice_channels(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Slice input tensor channels using the 1D binary mask.
        Args:
            x: Input tensor (B, 13, H, W)
            mask: 1D binary mask of shape (13,) or (B, 13)
        Returns:
            x_sliced: Tensor with selected channels (B, K, H, W)
        """
        if mask is None:
            mask = self.default_mask

        if mask.dim() == 2:
            # If batch of identical masks, use first row
            mask_1d = mask[0]
        else:
            mask_1d = mask

        active_indices = torch.nonzero(mask_1d > 0.5, as_tuple=False).squeeze(-1)

        if len(active_indices) == 0:
            raise ValueError("Band selection mask contains 0 active bands!")

        x_sliced = x[:, active_indices, :, :]

        # Adapt channel dimension if dynamic slice differs from expected selected_channels
        if x_sliced.shape[1] != self.selected_channels:
            adapter = nn.Conv2d(
                x_sliced.shape[1],
                self.selected_channels,
                kernel_size=1,
                device=x.device,
            )
            x_sliced = adapter(x_sliced)

        return self.channel_adapter(x_sliced)

    def extract_features(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract the uniform 1216-dimensional representation embedding vector.
        Args:
            x: Input tensor (B, 13, H, W)
            mask: Optional 1D binary mask (13,)
        Returns:
            fused_embedding: Tensor of shape (B, 1216)
        """
        # 1. Slice selected spectral channels
        x_selected = self.slice_channels(x, mask)

        # 2. Parallel feature extraction
        cnn_feats = self.cnn_backbone(x_selected)       # (B, 1024)
        vit_feats = self.vit_backbone(x_selected)       # (B, 192)

        # 3. Normalization
        cnn_feats = self.cnn_norm(cnn_feats)
        vit_feats = self.vit_norm(vit_feats)

        # 4. Depth-wise concatenation into 1216-dim representation vector
        fused_embedding = torch.cat([cnn_feats, vit_feats], dim=1)  # (B, 1216)
        return fused_embedding

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input tensor (B, 13, H, W)
            mask: Optional 1D binary mask (13,)
            return_embedding: If True, returns (logits, embedding)
        Returns:
            logits: Classification logits (B, num_classes)
            fused_embedding (optional): Representation vector (B, 1216)
        """
        fused_embedding = self.extract_features(x, mask)
        logits = self.classifier(fused_embedding)

        if return_embedding:
            return logits, fused_embedding
        return logits


if __name__ == "__main__":
    model = FusionNet(num_classes=10, selected_channels=6)
    x = torch.randn(2, 13, 256, 256)
    mask = torch.tensor([0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1], dtype=torch.float32)
    logits, emb = model(x, mask, return_embedding=True)
    print(f"FusionNet Input: {x.shape} with Mask -> Logits: {logits.shape}, Embedding: {emb.shape}")
    assert emb.shape == (2, 1216), f"Expected (2, 1216), got {emb.shape}"
    assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"
    print("FusionNet sanity check passed!")
