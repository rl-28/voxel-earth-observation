"""
VOXEL Module 3: Deep Learning Architectures
"""

from models.cnn_backbone import XNANet
from models.vit_transformer import Tiny32ViT
from models.fusion_net import FusionNet

__all__ = ["XNANet", "Tiny32ViT", "FusionNet"]
