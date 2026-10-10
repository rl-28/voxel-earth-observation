import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np

from .config import (
    EVAL_DIR, SERBIA_EVAL_DIR, BALKANS_EVAL_DIR, 
    SENTINEL_2_L2A_SPECTRAL_BANDS, EXPECTED_SHAPE
)
from .pipeline import Module1Pipeline, Module1Result

logger = logging.getLogger(__name__)

@dataclass
class EvaluationRegion:
    region_id: str
    display_name: str
    output_directory: Path
    description: str
    aoi_geojson: Optional[Path] = None
    stac_query: Optional[Dict[str, Any]] = None

EVALUATION_REGIONS = {
    "serbia": EvaluationRegion(
        region_id="serbia",
        display_name="Serbia (Ext SR)",
        output_directory=SERBIA_EVAL_DIR,
        description="Out-of-domain evaluation tiles for Serbia",
    ),
    "western_balkans": EvaluationRegion(
        region_id="western_balkans",
        display_name="Western Balkans (Ext WB)",
        output_directory=BALKANS_EVAL_DIR,
        description="Out-of-domain evaluation tiles for Western Balkans",
    )
}

@dataclass
class ProjectionVerificationResult:
    passed: bool
    output_crs: Optional[str]
    transform: Optional[tuple]
    bounds: Optional[tuple]
    width: int
    height: int
    messages: List[str]

class EvaluationPipeline:
    def __init__(self, region_id: str):
        if region_id not in EVALUATION_REGIONS:
            raise ValueError(f"Unsupported evaluation region: {region_id}")
        self.region = EVALUATION_REGIONS[region_id]
        self.pipeline = Module1Pipeline()
        
    def _verify_projection(self, result: Module1Result) -> ProjectionVerificationResult:
        messages = []
        passed = True
        
        crs = result.metadata.get("crs")
        if not crs:
            messages.append("Missing output CRS metadata.")
            passed = False
            
        transform = result.metadata.get("transform")
        if not transform:
            messages.append("Missing output transform metadata.")
            passed = False
            
        H, W = result.output_shape[:2]
        
        return ProjectionVerificationResult(
            passed=passed,
            output_crs=crs,
            transform=tuple(transform) if transform else None,
            bounds=result.metadata.get("bounds"),
            width=W,
            height=H,
            messages=messages
        )

    def process_local_source(self, band_paths: Dict[str, str], reference_band_key: str = "B04"):
        if not band_paths:
            raise ValueError(f"{self.region.display_name} source is not configured.")
            
        out_dir = self.region.output_directory
        out_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = out_dir / "manifest.json"
        
        manifest = {
            "region_identity": self.region.region_id,
            "display_name": self.region.display_name,
            "preprocessing_version": "1.0",
            "tiles": []
        }
        
        for res in self.pipeline.process_scene(band_paths, reference_band_key):
            proj_res = self._verify_projection(res)
            if not proj_res.passed:
                raise ValueError(f"Projection verification failed for {res.tile_id}: {proj_res.messages}")
            
            tile_filename = f"{self.region.region_id}_{res.tile_id}.npy"
            tile_path = out_dir / tile_filename
            
            np.save(tile_path, res.processed_tensor)
            
            tile_meta = {
                "tile_id": res.tile_id,
                "filename": tile_filename,
                "shape": res.output_shape,
                "dtype": str(res.processed_tensor.dtype),
                "output_crs": proj_res.output_crs,
                "bounds": proj_res.bounds,
                "warnings": proj_res.messages,
                "diagnostics": res.diagnostics
            }
            manifest["tiles"].append(tile_meta)
            
        manifest["number_of_tiles"] = len(manifest["tiles"])
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            
        return manifest

def main():
    parser = argparse.ArgumentParser(description="Out-of-domain evaluation data pipeline")
    parser.add_argument("--region", type=str, required=True, choices=list(EVALUATION_REGIONS.keys()), help="Evaluation region ID")
    parser.add_argument("--input", type=str, help="Path to source directory containing band GeoTIFFs (if running local mode)")
    
    args = parser.parse_args()
    
    pipeline = EvaluationPipeline(args.region)
    
    if args.input:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            raise ValueError(f"Input path {input_dir} is not a directory. Provide a directory containing band GeoTIFFs.")
            
        # Discover band paths in directory assuming format matches `SENTINEL_2_L2A_SPECTRAL_BANDS`
        band_paths = {}
        for b in SENTINEL_2_L2A_SPECTRAL_BANDS:
            # naive match for files containing the band name (e.g. B04.tif or T34UEC_..._B04_10m.jp2)
            matches = list(input_dir.glob(f"*{b}*.*"))
            if matches:
                band_paths[b] = str(matches[0])
                
        pipeline.process_local_source(band_paths)
    else:
        raise ValueError("AOI/source is not configured. Supply --input for local mode.")

if __name__ == "__main__":
    main()
