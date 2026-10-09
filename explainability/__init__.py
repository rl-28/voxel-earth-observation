"""
VOXEL Module 3: Explainability and Vector Search Engine
"""

from explainability.faiss_search import SpatialVectorSearchEngine, query_similar_terrain
from explainability.shap_engine import (
    compute_permutation_shap_attributions,
    plot_band_importance,
    run_explainability_pipeline,
)

__all__ = [
    "SpatialVectorSearchEngine",
    "query_similar_terrain",
    "compute_permutation_shap_attributions",
    "plot_band_importance",
    "run_explainability_pipeline",
]
