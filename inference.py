"""
VOXEL Module 3: Inference & Spatial Vector Retrieval Engine
Provides an end-to-end API to classify new 13-band Sentinel-2 tiles
and retrieve visually and spectrally similar terrain scenes using FAISS.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import numpy as np

# Ensure Windows UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from models.fusion_net import FusionNet
from explainability.faiss_search import query_similar_terrain

LAND_COVER_CLASSES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway / Infrastructure",
    "Industrial Area",
    "Pasture",
    "Permanent Crop",
    "Residential Area",
    "River / Inland Water",
    "Sea / Marine Water",
]

_CACHED_MODEL = None
_CACHED_MASK = None
_CACHED_DEVICE = None


def load_model(
    checkpoint_path: str = "checkpoints/best_fusion_net.pth",
    device: Optional[str] = None,
) -> Tuple[FusionNet, torch.Tensor, torch.device]:
    """Loads and caches the trained FusionNet model and registered band mask."""
    global _CACHED_MODEL, _CACHED_MASK, _CACHED_DEVICE

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    if _CACHED_MODEL is not None and _CACHED_DEVICE == device_obj:
        return _CACHED_MODEL, _CACHED_MASK, _CACHED_DEVICE

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    mask = checkpoint.get("band_mask", None)
    if mask is not None:
        mask = mask.to(device_obj)
        active_channels = int(mask.sum().item())
    else:
        active_channels = 6

    model = FusionNet(
        num_classes=len(LAND_COVER_CLASSES),
        total_bands=13,
        selected_channels=active_channels,
        cnn_feature_dim=1024,
        vit_feature_dim=192,
    ).to(device_obj)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _CACHED_MODEL = model
    _CACHED_MASK = mask
    _CACHED_DEVICE = device_obj

    return model, mask, device_obj


def predict_and_search(
    tile_tensor: Union[torch.Tensor, np.ndarray],
    band_mask: Optional[torch.Tensor] = None,
    top_k: int = 5,
    checkpoint_path: str = "checkpoints/best_fusion_net.pth",
    index_path: str = "checkpoints/terrain_index.faiss",
) -> Dict[str, Any]:
    """
    Classify a satellite tile and query FAISS for similar terrain scenes.
    
    Args:
        tile_tensor: (13, H, W) or (1, 13, H, W) normalized Float32 Sentinel-2 tile.
        band_mask: Optional 1D binary mask of 13 elements.
        top_k: Number of nearest similar terrain scenes to retrieve.
        checkpoint_path: Path to best_fusion_net.pth.
        index_path: Path to terrain_index.faiss.
        
    Returns:
        dict: {
            "predicted_class_id": int,
            "predicted_class_name": str,
            "confidence": float,
            "all_probabilities": dict,
            "embedding": np.ndarray (shape: (1216,)),
            "similar_terrain": list[dict],
            "retrieval_latency_ms": float,
        }
    """
    model, default_mask, device = load_model(checkpoint_path)

    # Prepare input tensor
    if isinstance(tile_tensor, np.ndarray):
        tile_tensor = torch.from_numpy(tile_tensor).float()

    if tile_tensor.dim() == 3:
        tile_tensor = tile_tensor.unsqueeze(0)  # (1, 13, H, W)

    tile_tensor = tile_tensor.to(device)

    if band_mask is None:
        band_mask = default_mask
    elif isinstance(band_mask, torch.Tensor):
        band_mask = band_mask.to(device)

    # 1. Forward Pass on GPU
    with torch.no_grad():
        logits, embedding = model(tile_tensor, mask=band_mask, return_embedding=True)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)

    pred_class_id = int(torch.argmax(probs).item())
    pred_class_name = LAND_COVER_CLASSES[pred_class_id]
    confidence_pct = float(probs[pred_class_id].item() * 100.0)
    embedding_np = embedding.squeeze(0).cpu().numpy()  # (1216,)

    # 2. Vector Retrieval Query
    scores, indices, metadata, latency_ms = query_similar_terrain(
        query_embedding=embedding_np,
        top_k=top_k,
        index_path=index_path,
    )

    similar_tiles = []
    for rank, (score, idx, meta) in enumerate(zip(scores, indices, metadata), 1):
        similar_tiles.append(
            {
                "rank": rank,
                "similarity_score": float(score),
                "index": int(idx),
                "tile_id": meta.get("tile_id", f"TILE_{idx}"),
                "split": meta.get("split", "unknown"),
                "class_id": meta.get("true_class", -1),
                "class_name": LAND_COVER_CLASSES[meta["true_class"]]
                if 0 <= meta.get("true_class", -1) < len(LAND_COVER_CLASSES)
                else "N/A",
            }
        )

    return {
        "predicted_class_id": pred_class_id,
        "predicted_class_name": pred_class_name,
        "confidence": confidence_pct,
        "all_probabilities": {
            LAND_COVER_CLASSES[i]: float(probs[i].item() * 100.0)
            for i in range(len(LAND_COVER_CLASSES))
        },
        "embedding": embedding_np,
        "similar_terrain": similar_tiles,
        "retrieval_latency_ms": latency_ms,
    }


if __name__ == "__main__":
    from tests.mock_handshake import generate_mock_batch

    print("=" * 70)
    print(" [INFERENCE] Testing Satellite Tile Prediction & Vector Retrieval")
    print("=" * 70)

    # Generate single test tile
    tiles, _, mask = generate_mock_batch(batch_size=1, seed=999)
    test_tile = tiles[0]

    result = predict_and_search(test_tile, band_mask=mask, top_k=5)

    print(f"Predicted Class:         {result['predicted_class_name']} (ID: {result['predicted_class_id']})")
    print(f"Prediction Confidence:   {result['confidence']:.2f}%")
    print(f"Extracted Embedding:     Shape {result['embedding'].shape} (dim=1216)")
    print(f"FAISS Retrieval Latency: {result['retrieval_latency_ms']:.3f} ms")
    print("-" * 70)
    print("Top-5 Similar Terrain Tiles Retrieved:")
    for tile in result["similar_terrain"]:
        print(
            f"  Rank {tile['rank']}: Similarity={tile['similarity_score']:.4f} | "
            f"Tile={tile['tile_id']} | Land Cover={tile['class_name']}"
        )
    print("=" * 70)
    print("Inference test passed successfully!")
