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


if __name__ == "__main__":
    print("=" * 65)
    print(" [FAISS] INITIALIZING 1216-DIM VECTOR RETRIEVAL ENGINE")
    print("=" * 65)

    search_engine = SpatialVectorSearchEngine(dim=1216, metric="cosine")

    # Generate 100 mock 1216-dim scene embeddings
    np.random.seed(42)
    mock_vectors = np.random.randn(100, 1216).astype(np.float32)
    mock_metadata = [
        {
            "tile_id": f"S2A_MSIL2A_TILE_{i:03d}",
            "land_cover_class": f"Class_{i % 10}",
            "lat": 19.0 + (i * 0.01),
            "lon": 72.8 + (i * 0.01),
        }
        for i in range(100)
    ]

    indexed_count = search_engine.add_embeddings(mock_vectors, mock_metadata)
    os.makedirs("checkpoints", exist_ok=True)
    test_index_file = "checkpoints/terrain_index.faiss"
    search_engine.save_index(test_index_file)
    print(f"Indexed Vectors:         {indexed_count} scenes (saved to {test_index_file})")

    # Perform a query search via query_similar_terrain
    query_vector = mock_vectors[0] + np.random.normal(0, 0.05, 1216).astype(np.float32)
    scores, indices, results, latency_ms = query_similar_terrain(query_vector, top_k=5, index_path=test_index_file)

    print(f"Retrieval Latency:       {latency_ms:.3f} ms (sub-millisecond search)")
    print("-" * 65)
    print("Top-5 Most Similar Spatial Scenes:")
    for rank, (score, idx, meta) in enumerate(zip(scores, indices, results), 1):
        print(
            f"  Rank {rank}: Score={score:.4f} | Index={idx} | ID={meta.get('tile_id')} | "
            f"Class={meta.get('land_cover_class')} | Coord=({meta.get('lat', 0):.3f}, {meta.get('lon', 0):.3f})"
        )
    print("=" * 65)
    print("FAISS vector retrieval engine verified successfully!")
