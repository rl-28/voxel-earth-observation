import streamlit as st
import tempfile
from pathlib import Path
import os
import traceback
import zipfile

from module1_ingestion.config import SENTINEL_2_L2A_SPECTRAL_BANDS
from module1_ingestion.pipeline import Module1Pipeline
from module1_ingestion.evaluation_pipeline import EvaluationPipeline, EVALUATION_REGIONS

st.set_page_config(page_title="Voxel Module 1 Pipeline", layout="wide")

st.title("VOXEL: Module 1 — Ingestion & Signal Processing")
st.markdown("Phase 10: Streamlit Frontend Integration")

st.header("Input Section")
mode = st.radio("Select Processing Mode", ["Local Directory / Upload", "Evaluation Region (Ext SR / Ext WB)"])

def process_bands(band_paths: dict):
    st.subheader("Processing Diagnostics")
    pipeline = Module1Pipeline()
    
    try:
        results = []
        for res in pipeline.process_scene(band_paths):
            results.append(res)
            st.write(f"Processed `{res.tile_id}` successfully.")
            with st.expander(f"Diagnostics: {res.tile_id}"):
                st.json(res.diagnostics)
                st.write(f"**Contract Type:** `{res.diagnostics.get('contract_type')}`")
                st.write(f"**Shape:** `{res.output_shape}`")
                
                # Check 12 vs 13 bands and display warning if necessary
                if res.output_shape[2] == 12:
                    st.warning("MODULE 2 HANDOFF: Output has 12 channels (Sentinel-2 L2A intermediate). Project-level 13th channel mapping remains unresolved.")
                
        st.success("Module 1 Processing Complete.")
        return results
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        return None

if mode == "Local Directory / Upload":
    uploaded_file = st.file_uploader("Upload a ZIP file containing Sentinel-2 L2A band GeoTIFFs (.tif or .jp2)", type="zip")
    if uploaded_file and st.button("Run Module 1 Processing"):
        with st.spinner("Extracting uploaded file and processing..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "upload.zip"
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.read())
                    
                extract_dir = Path(tmpdir) / "extracted"
                extract_dir.mkdir()
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                except zipfile.BadZipFile:
                    st.error("Uploaded file is not a valid ZIP archive.")
                    st.stop()
                    
                # Discover bands
                band_paths = {}
                for b in SENTINEL_2_L2A_SPECTRAL_BANDS:
                    matches = list(extract_dir.rglob(f"*{b}*.*"))
                    if matches:
                        band_paths[b] = str(matches[0])
                        
                if not band_paths:
                    st.error("No valid Sentinel-2 bands found in the uploaded ZIP file.")
                else:
                    st.write("Found bands:", list(band_paths.keys()))
                    process_bands(band_paths)

elif mode == "Evaluation Region (Ext SR / Ext WB)":
    region_id = st.selectbox("Select Region", list(EVALUATION_REGIONS.keys()), format_func=lambda x: EVALUATION_REGIONS[x].display_name)
    eval_input_dir = st.text_input("Local Source Directory Path for Evaluation Region (Optional)", "")
    
    if st.button("Run Module 1 Evaluation Processing"):
        try:
            eval_pipeline = EvaluationPipeline(region_id)
            if eval_input_dir:
                input_dir = Path(eval_input_dir)
                if not input_dir.is_dir():
                    st.error(f"Directory `{input_dir}` does not exist.")
                else:
                    band_paths = {}
                    for b in SENTINEL_2_L2A_SPECTRAL_BANDS:
                        matches = list(input_dir.rglob(f"*{b}*.*"))
                        if matches:
                            band_paths[b] = str(matches[0])
                            
                    if not band_paths:
                        st.error("No valid Sentinel-2 bands found in the provided directory.")
                    else:
                        with st.spinner(f"Processing {EVALUATION_REGIONS[region_id].display_name}..."):
                            manifest = eval_pipeline.process_local_source(band_paths)
                            st.success(f"Generated {manifest['number_of_tiles']} evaluation tiles.")
                            st.json(manifest)
            else:
                # Deliberately fail clearly as required by the specifications
                eval_pipeline.process_local_source({})
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
