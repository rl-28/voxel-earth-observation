"""
VOXEL Module 3: Vector Retrieval Engine - FAISS Vector Database
Indexes the 1216-dimensional spatial-spectral embeddings for sub-second
similarity searches, land cover retrieval, and explainability scene comparisons.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import faiss


class SpatialVectorSearchEngine:
    """
    FAISS-based High-Performance Vector Retrieval Engine.
    Indexes 1216-dimensional multimodal CNN-ViT representations
    for rapid geospatial tile similarity and land cover retrieval.
    """

    def __init__(self, dim: int = 1216, metric: str = "cosine"):
        """
        Args:
            dim (int): Vector embedding dimensionality (default: 1216).
            metric (str): 'cosine' (normalized inner product) or 'l2' (Euclidean distance).
        """
        self.dim = dim
        self.metric = metric.lower()

        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(dim)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Choose 'cosine' or 'l2'.")

        self.metadata_store: List[Dict[str, Any]] = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalizes feature vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return (vectors / norms).astype(np.float32)

    def add_embeddings(
        self,
        embeddings: Any,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add 1216-dim vectors to the index with associated scene metadata.
        Args:
            embeddings: Numpy array or PyTorch Tensor of shape (N, 1216)
            metadata: Optional list of N metadata dicts (e.g. tile_id, coordinates, class)
        Returns:
            total_indexed: Number of items in the FAISS index.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Dimension mismatch: Expected {self.dim}, got {embeddings.shape[1]}"
            )

        if self.metric == "cosine":
            embeddings = self._normalize(embeddings)

        num_vectors = embeddings.shape[0]
        self.index.add(embeddings)

        # Store metadata
        if metadata is not None:
            if len(metadata) != num_vectors:
                raise ValueError("Metadata length must match number of vectors.")
            self.metadata_store.extend(metadata)
        else:
            for i in range(num_vectors):
                self.metadata_store.append({"id": len(self.metadata_store)})

        return self.index.ntotal

    def search(
        self,
        query: Any,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]], float]:
        """
        Search for top_k most similar spatial tiles.
        Args:
            query: Query vector(s) of shape (1, 1216) or (Q, 1216)
            top_k: Number of nearest neighbors to retrieve
        Returns:
            distances: Similarity scores / distances (Q, top_k)
            indices: Integer indices of nearest vectors (Q, top_k)
            retrieved_metadata: Metadata for each neighbor
            search_time_ms: Search execution latency in milliseconds
        """
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()

        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dim:
            raise ValueError(
                f"Dimension mismatch: Expected {self.dim}, got {query.shape[1]}"
            )

        if self.metric == "cosine":
            query = self._normalize(query)

        start_time = time.perf_counter()
        distances, indices = self.index.search(query, top_k)
        search_time_ms = (time.perf_counter() - start_time) * 1000.0

        batch_metadata = []
        for row in indices:
            row_meta = [
                self.metadata_store[idx] if 0 <= idx < len(self.metadata_store) else {}
                for idx in row
            ]
            batch_metadata.append(row_meta)

        return distances, indices, batch_metadata, search_time_ms

    def save_index(self, index_file: str):
        """Save FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
        faiss.write_index(self.index, index_file)
        meta_file = index_file + ".meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2)

    def load_index(self, index_file: str):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(index_file)
        meta_file = index_file + ".meta.json"
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                self.metadata_store = json.load(f)


_GLOBAL_TERRAIN_ENGINE: Optional[SpatialVectorSearchEngine] = None


def get_default_engine(index_path: str = "checkpoints/terrain_index.faiss") -> SpatialVectorSearchEngine:
    global _GLOBAL_TERRAIN_ENGINE
    if _GLOBAL_TERRAIN_ENGINE is None:
        _GLOBAL_TERRAIN_ENGINE = SpatialVectorSearchEngine(dim=1216, metric="cosine")
        if os.path.exists(index_path):
            _GLOBAL_TERRAIN_ENGINE.load_index(index_path)
    return _GLOBAL_TERRAIN_ENGINE


def query_similar_terrain(
    query_embedding: Any,
    top_k: int = 5,
    index_path: str = "checkpoints/terrain_index.faiss",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], float]:
    """
    Exposed API function for querying nearest tile indices and similarity scores.
    Args:
        query_embedding: Query vector (1216,) or (1, 1216)
        top_k: Number of nearest neighbors to retrieve
        index_path: Path to persisted FAISS index file
    Returns:
        scores: Array of similarity scores (top_k,)
        indices: Array of tile integer indices (top_k,)
        metadata: List of metadata dicts for nearest tiles
        latency_ms: Search latency in milliseconds
    """
    engine = get_default_engine(index_path)
    scores, indices, results, latency_ms = engine.search(query_embedding, top_k=top_k)
    return scores[0], indices[0], results[0], latency_ms


def build_faiss_index_from_dataloader(
    model: torch.nn.Module,
    dataloader: Any,
    mask: torch.Tensor,
    device: torch.device,
    index_save_path: str = "checkpoints/terrain_index.faiss",
    dim: int = 1216,
) -> Tuple[str, int]:
    """
    Extracts 1216-dim embeddings across a DataLoader and builds/persists the FAISS index.
    """
    model.eval()
    mask = mask.to(device)
    engine = SpatialVectorSearchEngine(dim=dim, metric="cosine")

    all_embeddings = []
    all_metadata = []
    tile_counter = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            logits, embs = model(images, mask=mask, return_embedding=True)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            embs_np = embs.cpu().numpy()

            for i in range(images.size(0)):
                all_embeddings.append(embs_np[i])
                meta_item = {
                    "tile_id": f"TILE_{tile_counter:04d}",
                    "true_class": int(targets[i].item()),
                    "predicted_class": int(preds[i]),
                }
                # If dataset has records with extra columns, include them
                if hasattr(dataloader.dataset, "records") and tile_counter < len(dataloader.dataset.records):
                    row = dataloader.dataset.records.iloc[tile_counter]
                    if "location_id" in row:
                        meta_item["location_id"] = str(row["location_id"])
                    if "tile_path" in row:
                        meta_item["tile_path"] = str(row["tile_path"])

                all_metadata.append(meta_item)
                tile_counter += 1

    if len(all_embeddings) == 0:
        raise ValueError("DataLoader yielded 0 tiles; cannot build FAISS index.")

    embeddings_arr = np.array(all_embeddings, dtype=np.float32)
    indexed_count = engine.add_embeddings(embeddings_arr, all_metadata)
    engine.save_index(index_save_path)

    return index_save_path, indexed_count


if __name__ == "__main__":
    import argparse
    from inference import load_model

    parser = argparse.ArgumentParser(description="VOXEL Module 3: FAISS Vector Retrieval Engine")
    parser.add_argument("--index_path", type=str, default="checkpoints/terrain_index.faiss", help="Path to index file")
    parser.add_argument("--query_tile", type=str, default=None, help="Path to query tile tensor")
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest tiles")
    args = parser.parse_args()

    if os.path.exists(args.index_path):
        engine = get_default_engine(args.index_path)
        print(f"[FAISS] Loaded existing index: {args.index_path} ({engine.index.ntotal} vectors indexed)")
    else:
        print(f"[FAISS] Index not found at {args.index_path}. Build it via build_explainability.py.")
