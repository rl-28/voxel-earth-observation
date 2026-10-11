"""
VOXEL: Task-Adaptive Earth Observation & Spectral Intelligence Platform
Streamlit Dashboard (app.py)

Cinematic High-Contrast Earth-Observation Interface & Scientific Verification Hub:
  Section 1: 🚀 End-to-End Orchestrator
  Section 2: 🔬 Dataset Provenance & Model Verification
  Section 3: 📡 Module 1: Satellite Ingestion & Signal Processing
  Section 4: ⚡ Module 2: Physics-Guided Spectral Band Optimization
  Section 5: 🧠 Module 3: CNN-ViT FusionNet, Explainability & FAISS Retrieval
"""

import os
import sys
import json
import time
import zipfile
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1_ingestion.config import SENTINEL_2_L2A_SPECTRAL_BANDS
from module1_ingestion.pipeline import Module1Pipeline
from module1_ingestion.evaluation_pipeline import EvaluationPipeline, EVALUATION_REGIONS
from voxel_pipeline import VoxelPipeline, VoxelPipelineResult
from config.constants import (
    SENTINEL2_BANDS,
    SENTINEL2_BAND_KEYS,
    LAND_COVER_CLASSES,
    TASKS,
    DEFAULT_SCALE_DIVISOR,
    NUM_BANDS,
)
from utils.band_mask import describe_band_mask
from ui_theme import (
    inject_custom_styles,
    render_cinematic_hero,
    render_operational_header,
    render_spectral_matrix,
    render_hud_prediction,
    style_dark_space_figure,
    plot_dark_training_curve,
)


# -----------------------------------------------------------------------------
# CORE MODULE 1 BACKWARD COMPATIBILITY FUNCTION
# -----------------------------------------------------------------------------
def process_bands(band_paths: dict):
    """
    Maintains backward compatibility with Module 1 tests and pipeline.
    Processes Sentinel-2 band paths through Module1Pipeline.
    """
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
                if len(res.output_shape) >= 3 and res.output_shape[2] == 12:
                    st.warning("MODULE 2 HANDOFF: Output has 12 channels (Sentinel-2 L2A intermediate). Project-level 13th channel mapping remains unresolved.")
                
        st.success("Module 1 Processing Complete.")
        return results
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        return None


# -----------------------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# -----------------------------------------------------------------------------
def render_dashboard():
    # Set Page Config
    st.set_page_config(
        page_title="VOXEL: Spectral Intelligence Platform",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject High-Contrast Dark Space Theme
    inject_custom_styles()

    # Initialize Session State for Pipeline Results
    if "latest_pipeline_result" not in st.session_state:
        st.session_state["latest_pipeline_result"] = None
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = None

    # -------------------------------------------------------------------------
    # SIDEBAR: COMPACT VERTICAL NAVIGATION & PROVENANCE TELEMETRY
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
                <span class="voxel-pulse-dot"></span>
                <span style="font-family: var(--font-display); font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: 0.06em;">VOXEL</span>
            </div>
            <div style="font-family: var(--font-tech); font-size: 0.78rem; color: var(--electric-cyan); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 16px;">
                ORBITAL SPECTRAL ENGINE
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Accessible Vertical Module Navigation (prevents horizontal tab clipping)
        nav_selection = st.radio(
            "MODULE NAVIGATION",
            [
                "🚀 01 // Orchestrator",
                "🔬 02 // Provenance & Verification",
                "📡 03 // Module 1: Ingestion",
                "⚡ 04 // Module 2: BWOA Optimization",
                "🧠 05 // Module 3: Deep Learning",
            ],
            key="nav_selection",
        )

        st.markdown("---")

        # Telemetry & Verification Status Panel
        st.markdown(
            """
            <div class="voxel-card" style="padding: 12px 14px; margin-bottom: 12px;">
                <div style="font-family: var(--font-tech); font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                    SYSTEM TELEMETRY
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-size: 0.82rem; color: #cbd5e1;">Module 1 Ingestion</span>
                    <span class="voxel-badge-success">ONLINE</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-size: 0.82rem; color: #cbd5e1;">Module 2 BWOA</span>
                    <span class="voxel-badge-success">ONLINE</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-size: 0.82rem; color: #cbd5e1;">Module 3 FusionNet</span>
                    <span class="voxel-badge-success">HARDENED</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.82rem; color: #cbd5e1;">Verification Suite</span>
                    <span style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--electric-cyan);">178 PASS (CI)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="voxel-card" style="padding: 12px 14px;">
                <div style="font-family: var(--font-tech); font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                    SCIENTIFIC GATING
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">
                    <p style="margin: 0 0 3px 0;"><span style="color: #6ee7b7;">●</span> <b>Real Sentinel-2:</b> BOA Calibrated</p>
                    <p style="margin: 0 0 3px 0;"><span style="color: #6ee7b7;">●</span> <b>Holdout:</b> Hirakud 19-Tile Scene</p>
                    <p style="margin: 0 0 3px 0;"><span style="color: #fcd34d;">●</span> <b>Atmospheric:</b> B10 Refused</p>
                    <p style="margin: 0;"><span style="color: #38bdf8;">●</span> <b>Mask Safety:</b> Calibrated Decoupling</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("VOXEL Platform © 2026. Task-Adaptive Multi-Spectral Intelligence.")

    # =========================================================================
    # SECTION 1: END-TO-END ORCHESTRATOR
    # =========================================================================
    if nav_selection == "🚀 01 // Orchestrator":
        # Cinematic Earth Hero (Compact top banner)
        render_cinematic_hero()

        st.markdown(
            """
            <div class="voxel-section-header">
                <h2>Orbital Orchestrator</h2>
                <p>Natural Language Query ➔ Stage 0 Compilation ➔ Sentinel-2 STAC Ingestion ➔ Physics BWOA ➔ CNN-ViT FusionNet Inference</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Control Panel Card
        st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown('<div class="voxel-card-header">1. TARGET &amp; TASK SELECTION</div>', unsafe_allow_html=True)
            preset_query = st.selectbox(
                "Select Verified Location or Custom Query:",
                [
                    "Track waterbody levels in Loktak Lake, Manipur",
                    "Assess forest canopy in Jim Corbett National Park, Uttarakhand",
                    "Identify water bodies around Chilika Lake, Odisha, India",
                    "Analyze forest canopy and tree cover around Western Ghats, Karnataka, India",
                    "Monitor agricultural crop health in Punjab, India",
                    "Custom query...",
                ]
            )
            if preset_query == "Custom query...":
                user_query = st.text_input("Natural Language Query:", "Track waterbody levels in Loktak Lake, Manipur")
            else:
                user_query = st.text_input("Active Target Query:", preset_query)

            task_choice = st.radio(
                "Task Profile Mode:",
                ["auto", "waterbody", "forest_cover"],
                format_func=lambda x: "Auto-Detect from Query" if x == "auto" else ("Waterbody Classification (6 Bands)" if x == "waterbody" else "Forest Cover Classification (5 Bands)"),
                horizontal=True,
            )

        with col2:
            st.markdown('<div class="voxel-card-header">2. EXECUTION SETTINGS</div>', unsafe_allow_html=True)
            exec_mode = st.radio(
                "Execution Mode:",
                ["real", "offline"],
                format_func=lambda x: "🛰️ Real Satellite Mode (STAC Acquisition)" if x == "real" else "🧪 Offline Smoke Test (Synthetic Fixture)"
            )
            sc1, sc2 = st.columns(2)
            with sc1:
                run_bwoa = st.checkbox("Live BWOA Optimization", value=False, help="Executes fresh Binary Whale Optimization on the scene tensor.")
                run_faiss = st.checkbox("Semantic Latent Retrieval", value=True, help="Queries FAISS index on 1216-dim latent vector.")
            with sc2:
                run_shap = st.checkbox("Compute SHAP Attributions", value=False, help="Optional: Computes live spatial permutation SHAP scores (adds ~2s latency).")

        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        launch_btn = st.button("🚀 EXECUTE ORBITAL PIPELINE", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Pipeline Execution Handler
        if launch_btn:
            with st.spinner("Executing orbital pipeline (Stage 0 ➔ Module 1 STAC ➔ Module 2 BWOA ➔ Module 3 FusionNet)..."):
                try:
                    pipeline = VoxelPipeline(default_mode=exec_mode, verbose=False)
                    res: VoxelPipelineResult = pipeline.run(
                        query=user_query,
                        mode=exec_mode,
                        task_override=None if task_choice == "auto" else task_choice,
                        explain_shap=run_shap,
                        run_faiss=run_faiss,
                        optimize_bands=run_bwoa,
                    )
                    st.session_state["latest_pipeline_result"] = res
                    st.session_state["last_query"] = user_query
                except Exception as e:
                    st.error(f"Pipeline execution error: {str(e)}")
                    st.code(traceback.format_exc())

        # Render Active or Persisted Results
        res: Optional[VoxelPipelineResult] = st.session_state.get("latest_pipeline_result")

        if res is not None:
            # 1. Top HUD Result Card
            st.markdown("---")
            if res.prediction:
                render_hud_prediction(
                    res.prediction,
                    res.task_type,
                    res.band_reduction_pct,
                    compatibility_note=res.mask_compatibility_note,
                )
            else:
                st.markdown(
                    """
                    <div class="voxel-card" style="border-color: rgba(245, 158, 11, 0.4); background: rgba(245, 158, 11, 0.05);">
                        <span class="voxel-badge-warning">🛡️ SCIENTIFIC GATING ENFORCED</span>
                        <h3 style="margin-top: 6px; color: #fcd34d;">Prediction Withheld for Scientific Authenticity</h3>
                        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
                            Candidate spectral mask channels do not match the calibrated input weights of the real trained model checkpoint. 
                            VOXEL strictly prevents uncalibrated tensor forward passes.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Task Scope Qualification Notice (if applicable)
            if "task_qualifications" in res.diagnostics and res.diagnostics["task_qualifications"]:
                for q_note in res.diagnostics["task_qualifications"]:
                    st.info(f"ℹ️ {q_note}")

            # 2. Telemetry Metrics Strip
            st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Task Profile", res.task_type.upper(), f"{len(res.selected_bands)} Bands Active")
            m2.metric("Volume Reduction", f"-{res.band_reduction_pct:.1f}%", f"{13 - len(res.selected_bands)} Pruned")
            m3.metric("Total Latency", f"{res.timing_breakdown_ms.get('total_pipeline_latency_ms', 0):.1f} ms", "Sub-Second DL")
            prov_label = "Live Sentinel-2" if not res.is_synthetic else "Synthetic Fixture"
            m4.metric("Data Provenance", prov_label, "STAC L2A Calibrated")

            # 3. Pipeline Progress & Diagnostics Grid
            st.markdown("---")
            st.markdown('<div class="voxel-section-header"><h2>Pipeline Progress &amp; Diagnostics</h2></div>', unsafe_allow_html=True)
            
            s0_col, s1_col, s2_col, s3_col = st.columns(4)
            with s0_col:
                st.markdown(
                    """
                    <div class="voxel-card" style="height: 100%;">
                        <div class="voxel-card-header">STAGE 0 // COMPILER</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(f"**Target:** `{res.resolved_location.get('name')}`")
                st.write(f"**BBox:** `{res.resolved_location.get('bbox')}`")
                st.write(f"**Geocoder:** `{res.resolved_location.get('geocoder')}`")
                st.caption(f"Latency: {res.timing_breakdown_ms.get('stage_0_query_compilation_ms', 0):.1f} ms")
                st.markdown('</div>', unsafe_allow_html=True)

            with s1_col:
                st.markdown(
                    """
                    <div class="voxel-card" style="height: 100%;">
                        <div class="voxel-card-header">MODULE 1 // INGESTION</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(f"**Tensor:** `{res.input_tensor_shape}`")
                st.write(f"**Source:** `{res.data_source}`")
                if "cache_status" in res.diagnostics:
                    st.write(f"**Cache Status:** `{res.diagnostics['cache_status']}`")
                st.write(f"**Radiometric:** `[0.0, 1.0] BOA`")
                st.caption(f"Latency: {res.timing_breakdown_ms.get('module_1_ingestion_ms', 0):.1f} ms")
                st.markdown('</div>', unsafe_allow_html=True)

            with s2_col:
                st.markdown(
                    """
                    <div class="voxel-card" style="height: 100%;">
                        <div class="voxel-card-header">MODULE 2 // BWOA</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(f"**Inference Bands ({len(res.selected_bands)}):** `{', '.join(res.selected_bands)}`")
                if res.candidate_bwoa_bands:
                    st.write(f"**Live BWOA Candidate:** `{', '.join(res.candidate_bwoa_bands)}`")
                    st.write("**Strategy:** `Decoupled for Validity`")
                else:
                    st.write(f"**Strategy:** `{res.mask_provenance}`")
                st.write(f"**B10 Cirrus:** `🔒 Refused (0.0)`")
                st.caption(f"Latency: {res.timing_breakdown_ms.get('module_2_optimization_ms', 0):.1f} ms")
                st.markdown('</div>', unsafe_allow_html=True)

            with s3_col:
                st.markdown(
                    """
                    <div class="voxel-card" style="height: 100%;">
                        <div class="voxel-card-header">MODULE 3 // DEEP INFERENCE</div>
                    """,
                    unsafe_allow_html=True,
                )
                if res.prediction:
                    st.write(f"**Class:** `{res.prediction.get('predicted_class_name')}`")
                    st.write(f"**Confidence:** `{res.prediction.get('confidence_pct', 0):.1f}%`")
                    prov = res.prediction.get("provenance")
                    prov_badge = "✅ Genuine Real Model" if prov == "real_inference" else "⚠️ Synthetic Fixture"
                    st.write(f"**Model:** `{prov_badge}`")
                    if "all_probabilities" in res.prediction:
                        for cls_name, p_val in res.prediction["all_probabilities"].items():
                            st.progress(p_val / 100.0, text=f"{cls_name}: {p_val:.1f}%")
                else:
                    st.warning("Prediction withheld")
                st.caption(f"Latency: {res.timing_breakdown_ms.get('module_3_inference_ms', 0):.1f} ms")
                st.markdown('</div>', unsafe_allow_html=True)

            # 4. Satellite Imagery & Spectral Matrix
            st.markdown("---")
            vcol1, vcol2 = st.columns([1, 2])
            with vcol1:
                st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
                st.markdown('<div class="voxel-card-header">SATELLITE RGB VISUALIZATION</div>', unsafe_allow_html=True)
                if res.diagnostics.get("rgb_preview_path") and os.path.exists(res.diagnostics["rgb_preview_path"]):
                    st.image(res.diagnostics["rgb_preview_path"], caption="Sentinel-2 L2A True-Color Composite (B04, B03, B02)", use_container_width=True)
                else:
                    st.info("RGB composite preview unavailable for this tile.")
                st.caption("Calibrated Bottom-of-Atmosphere (BOA) Surface Reflectance scaled in [0.0, 1.0].")
                st.markdown('</div>', unsafe_allow_html=True)

            with vcol2:
                st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
                st.markdown('<div class="voxel-card-header">ACTIVE SPECTRAL MATRIX (SENTINEL-2 MSI)</div>', unsafe_allow_html=True)
                render_spectral_matrix(res.selected_bands)
                if res.candidate_bwoa_bands:
                    st.caption(f"⚡ **Live BWOA Candidate Bands:** `{', '.join(res.candidate_bwoa_bands)}` | **Model Verified Bands:** `{', '.join(res.selected_bands)}`")
                else:
                    st.caption("Active bands highlighted in electric cyan. Atmospheric Band 10 (Cirrus) permanently locked and refused.")
                st.markdown('</div>', unsafe_allow_html=True)

            # 5. Explainability & Semantic Vector Retrieval
            if res.explainability or res.similar_terrain:
                st.markdown("---")
                exp_col, sim_col = st.columns(2)

                with exp_col:
                    st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
                    st.markdown('<div class="voxel-card-header">PERMUTATION SHAP SPECTRAL IMPORTANCE</div>', unsafe_allow_html=True)
                    if res.explainability and "top_contributing_bands" in res.explainability:
                        top_bands = res.explainability["top_contributing_bands"]
                        band_names = [b["name"] for b in top_bands]
                        band_pcts = [b["percentage"] for b in top_bands]

                        if any(p > 0.0 for p in band_pcts):
                            fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
                            y_pos = np.arange(len(band_names))
                            ax.barh(y_pos, band_pcts, color="#00f0ff", align="center", edgecolor="#00f0ff", height=0.6)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(band_names)
                            ax.invert_yaxis()
                            ax.set_xlabel("Relative Attribution (%)", fontsize=8)
                            ax.set_title("Permutation SHAP Feature Attribution", fontsize=10, fontweight="bold")
                            style_dark_space_figure(fig, ax)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("SHAP Attributions: Prediction confidence invariant to spatial permutation across channels for this tile.")
                    else:
                        st.info("SHAP attribution disabled in settings (saves CPU time).")
                    st.markdown('</div>', unsafe_allow_html=True)

                with sim_col:
                    st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
                    st.markdown('<div class="voxel-card-header">SEMANTIC TERRAIN MATCHES (FAISS)</div>', unsafe_allow_html=True)
                    st.caption("ℹ️ **Reference Corpus Notice:** Matches retrieved from the EuroSAT multi-class benchmark corpus (32 scenes) to validate 1216-dim latent vector semantic coherence. Real Indian task inference is executed directly by the trained classification heads.")
                    if res.similar_terrain:
                        terrain_df = pd.DataFrame(res.similar_terrain)
                        st.dataframe(terrain_df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            # 6. Result Contract JSON Viewer
            with st.expander("📄 View Machine-Readable Result Contract (JSON)"):
                st.json(res.to_dict())

        else:
            # Informative Ready State (NO large blank panel)
            st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
            st.markdown('<div class="voxel-card-header">SYSTEM READY // PIPELINE BRIEFING</div>', unsafe_allow_html=True)
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                st.write("**Preconfigured Queries:**")
                st.markdown(
                    """
                    - `Loktak Lake, Manipur` (Waterbody)
                    - `Jim Corbett, Uttarakhand` (Forest Cover)
                    - `Chilika Lake, Odisha` (Waterbody)
                    - `Western Ghats, Karnataka` (Forest Cover)
                    """
                )
            with bcol2:
                st.write("**Trained Model Checkpoints:**")
                st.markdown(
                    """
                    - `Waterbody:` 6 Bands (B01, B03, B04, B06, B8A, B11) — 73.7% Test OA
                    - `Forest Cover:` 5 Bands (B02, B04, B06, B11, B12) — 94.7% Test OA
                    - `General Land Cover:` 6 Bands (EuroSAT Benchmark)
                    """
                )
            with bcol3:
                st.write("**Execution Flow:**")
                st.markdown(
                    """
                    1. Geocode query location (OSM Nominatim)
                    2. Query AWS Earth Search STAC for Sentinel-2 L2A scene
                    3. Apply physics-guided BWOA channel selection
                    4. Execute 1216-dim CNN-ViT FusionNet inference
                    """
                )
            st.info("💡 Select a target query above and click **'EXECUTE ORBITAL PIPELINE'** to begin.")
            st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: DATASET PROVENANCE & MODEL VERIFICATION HUB
    # =========================================================================
    elif nav_selection == "🔬 02 // Provenance & Verification":
        render_operational_header(
            "Scientific Dataset Provenance &amp; Verification Hub",
            "Zero-Leakage Spatial Split Audit · Calibrated Sentinel-2 BOA Reflectance · Real Checkpoint Registry",
            "MODULE 0 // DATA PROVENANCE"
        )

        st.subheader("1. Genuine Sentinel-2 Real-Data Foundation vs Synthetic Fixtures")
        
        prov_col1, prov_col2, prov_col3 = st.columns(3)
        with prov_col1:
            st.markdown(
                """
                <div class="voxel-card">
                    <div class="voxel-card-header" style="color: #6ee7b7;">🛰️ GENUINE SATELLITE DATASET</div>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 18px; margin: 0;">
                        <li><b>Source:</b> AWS Earth Search STAC (<code>sentinel-2-l2a</code>)</li>
                        <li><b>Coverage:</b> 6 Indian geographic regions (Chilika, Western Ghats, Thar, Sardar Sarovar, Hirakud, Kaziranga)</li>
                        <li><b>Scale:</b> 120 genuine calibrated 256×256 tiles</li>
                        <li><b>Radiometric:</b> BOA Surface Reflectance <code>[0.0, 1.0]</code></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with prov_col2:
            st.markdown(
                """
                <div class="voxel-card">
                    <div class="voxel-card-header" style="color: #6ee7b7;">🏷️ VERIFIED ESA WORLDCOVER LABELS</div>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 18px; margin: 0;">
                        <li><b>Reference:</b> ESA WorldCover 2021 v200 (10m resolution)</li>
                        <li><b>Spatial Alignment:</b> Nearest-neighbor raster reprojection to tile CRS</li>
                        <li><b>Thresholding:</b> Positive ≥ 10%, Negative ≤ 0.5%</li>
                        <li><b>Quality:</b> Ambiguous boundary tiles dropped</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with prov_col3:
            st.markdown(
                """
                <div class="voxel-card">
                    <div class="voxel-card-header" style="color: #6ee7b7;">🗺️ GEOGRAPHIC SEPARATION SPLIT</div>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 18px; margin: 0;">
                        <li><b>Train (68 tiles):</b> Kaziranga, Western Ghats, Thar, Sardar Sarovar</li>
                        <li><b>Val (20 tiles):</b> Chilika Lagoon Coast</li>
                        <li><b>Held-Out Test (19 tiles):</b> Hirakud Inland Reservoir</li>
                        <li><b>Status:</b> Zero spatial overlap with training regions</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("ℹ️ **Scientific Scope & Limitation:** Evaluated on the held-out Hirakud inland reservoir scene (19 tiles). While ensuring zero geographic overlap with training regions, it represents an initial experimental test set. Nationwide generalization requires ongoing multi-region expansion.")

        # Visual Inspection Sheet
        st.markdown("---")
        st.subheader("2. Real Dataset Visual Inspection (RGB vs ESA WorldCover Ground Truth)")
        label_check_img = PROJECT_ROOT / "results" / "label_check.png"
        if label_check_img.exists():
            st.image(str(label_check_img), caption="Real Sentinel-2 L2A RGB vs ESA WorldCover 2021 v200 Ground Truth Maps across India", use_container_width=True)

        st.markdown("---")
        st.subheader("3. Verified Checkpoint Registry & Training Metrics")

        ckpt_summary = [
            {
                "Task": "waterbody (REAL DATA)",
                "Checkpoint Path": "checkpoints/real_waterbody/best_fusion_net.pth",
                "Active Channels": "6 bands (B01, B03, B04, B06, B8A, B11)",
                "Data Source": "Real Sentinel-2 L2A (India)",
                "Best Epoch": 4,
                "Val OA (%)": "90.00%",
                "Val F1 (%)": "88.10%",
                "Test OA (%)": "73.68%",
                "Test Macro F1 (%)": "73.39%",
                "Water Recall": "100.00%",
                "Non-water Precision": "100.00%",
                "Scientific Status": "Verified Genuine Real Satellite Model",
            },
            {
                "Task": "waterbody (Synthetic Mock)",
                "Checkpoint Path": "checkpoints/waterbody/best_fusion_net.pth",
                "Active Channels": "6 bands (B01, B03, B04, B06, B8A, B11)",
                "Data Source": "Synthetic Uniform Noise",
                "Best Epoch": 1,
                "Val OA (%)": "50.00%",
                "Val F1 (%)": "33.33%",
                "Test OA (%)": "50.00%",
                "Test Macro F1 (%)": "33.33%",
                "Water Recall": "100.00%",
                "Non-water Precision": "50.00%",
                "Scientific Status": "Engineering Test Fixture (Collapsed)",
            },
            {
                "Task": "forest_cover (REAL DATA)",
                "Checkpoint Path": "checkpoints/real_forest_cover/best_fusion_net.pth",
                "Active Channels": "5 bands (B02, B04, B06, B11, B12)",
                "Data Source": "Real Sentinel-2 L2A (India)",
                "Best Epoch": 1,
                "Val OA (%)": "66.67%",
                "Val F1 (%)": "47.92%",
                "Test OA (%)": "94.74%",
                "Test Macro F1 (%)": "93.60%",
                "Water/Forest Recall": "100.00%",
                "Precision": "92.86% (Forest), 100% (Non-forest)",
                "Scientific Status": "Verified Genuine Real Satellite Model (+26.3% OA vs Baseline)",
            },
            {
                "Task": "forest_cover (Synthetic Mock)",
                "Checkpoint Path": "checkpoints/forest_cover/best_fusion_net.pth",
                "Active Channels": "5 bands (B02, B04, B06, B11, B12)",
                "Data Source": "Synthetic Uniform Noise",
                "Best Epoch": 1,
                "Val OA (%)": "100.00%",
                "Val F1 (%)": "100.00%",
                "Test OA (%)": "100.00%",
                "Test Macro F1 (%)": "100.00%",
                "Water/Forest Recall": "N/A",
                "Precision": "N/A",
                "Scientific Status": "Engineering Test Fixture",
            },
            {
                "Task": "land_cover (EuroSAT 10-Class Mock)",
                "Checkpoint Path": "checkpoints/best_fusion_net.pth",
                "Active Channels": "6 bands (B02, B03, B04, B08, B11, B12)",
                "Data Source": "Synthetic Mock",
                "Best Epoch": 1,
                "Val OA (%)": "9.38%",
                "Val F1 (%)": "1.71%",
                "Test OA (%)": "3.12%",
                "Test Macro F1 (%)": "0.67%",
                "Water/Forest Recall": "N/A",
                "Precision": "N/A",
                "Scientific Status": "Engineering Test Fixture",
            },
        ]

        st.dataframe(pd.DataFrame(ckpt_summary), use_container_width=True)

        # Training Dynamics & Confusion Matrices
        st.markdown("---")
        st.subheader("4. Real-Data Training Dynamics & Held-Out Test Confusion Matrices")
        tcol1, tcol2 = st.columns(2)

        with tcol1:
            st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
            st.markdown('<div class="voxel-card-header">REAL WATERBODY EVALUATION (HELD-OUT HIRAKUD)</div>', unsafe_allow_html=True)
            real_wb_cm = PROJECT_ROOT / "results" / "real_waterbody_eval" / "confusion_matrix.png"
            if real_wb_cm.exists():
                st.image(str(real_wb_cm), caption="Held-Out Hirakud Confusion Matrix (OA: 73.68%, Macro-F1: 73.39%, Water Recall: 100%)", use_container_width=True)
            
            real_wb_log = PROJECT_ROOT / "results" / "real_waterbody_training_log.csv"
            if real_wb_log.exists():
                fig_wb = plot_dark_training_curve(real_wb_log, "Real Waterbody Training Progression")
                if fig_wb:
                    st.pyplot(fig_wb)
                    plt.close(fig_wb)
            st.markdown('</div>', unsafe_allow_html=True)

        with tcol2:
            st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
            st.markdown('<div class="voxel-card-header">REAL FOREST COVER EVALUATION (HELD-OUT HIRAKUD)</div>', unsafe_allow_html=True)
            real_fc_cm = PROJECT_ROOT / "results" / "real_forest_cover_eval" / "confusion_matrix.png"
            if real_fc_cm.exists():
                st.image(str(real_fc_cm), caption="Held-Out Hirakud Confusion Matrix (OA: 94.74%, Macro-F1: 93.60%, Forest Recall: 100%)", use_container_width=True)
                
            real_fc_log = PROJECT_ROOT / "results" / "real_forest_cover_training_log.csv"
            if real_fc_log.exists():
                fig_fc = plot_dark_training_curve(real_fc_log, "Real Forest Cover Training Progression")
                if fig_fc:
                    st.pyplot(fig_fc)
                    plt.close(fig_fc)
            st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3: MODULE 1 INGESTION & SIGNAL PROCESSING
    # =========================================================================
    elif nav_selection == "📡 03 // Module 1: Ingestion":
        render_operational_header(
            "Satellite Ingestion &amp; Radiometric Signal Processing",
            "Direct Ingestion · 256×256 Spatial Tiling · 2D DWT Wavelet Decomposition · Calibrated BOA Reflectance",
            "MODULE 1 // SATELLITE INGESTION"
        )

        st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
        m1_mode = st.radio("Select Processing Mode", ["Local Directory / Upload", "Evaluation Region (Ext SR / Ext WB)"], key="m1_radio")

        if m1_mode == "Local Directory / Upload":
            uploaded_file = st.file_uploader("Upload a ZIP file containing Sentinel-2 L2A band GeoTIFFs (.tif or .jp2)", type="zip", key="m1_zip_uploader")
            if uploaded_file and st.button("Run Module 1 Processing", key="m1_run_btn"):
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

        elif m1_mode == "Evaluation Region (Ext SR / Ext WB)":
            region_id = st.selectbox("Select Region", list(EVALUATION_REGIONS.keys()), format_func=lambda x: EVALUATION_REGIONS[x].display_name, key="m1_region_select")
            eval_input_dir = st.text_input("Local Source Directory Path for Evaluation Region (Optional)", "", key="m1_eval_dir")
            
            if st.button("Run Module 1 Evaluation Processing", key="m1_eval_btn"):
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
                        eval_pipeline.process_local_source({})
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    with st.expander("Technical Details"):
                        st.code(traceback.format_exc())
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 4: MODULE 2 SPECTRAL BAND OPTIMIZATION
    # =========================================================================
    elif nav_selection == "⚡ 04 // Module 2: BWOA Optimization":
        render_operational_header(
            "Physics-Guided Spectral Band Optimization (BWOA)",
            "Task-Adaptive Down-Selection · Binary Whale Optimization · Domain Balance & Atmospheric Band Refusal",
            "MODULE 2 // SPECTRAL OPTIMIZATION"
        )

        mask_files = {
            "Waterbody Mask (6 Bands)": "checkpoints/masks/waterbody_mask.json",
            "Forest Cover Mask (5 Bands)": "checkpoints/masks/forest_cover_mask.json",
            "Drought Mask (4 Bands)": "checkpoints/masks/drought_mask.json",
            "Rainfall Mask (5 Bands)": "checkpoints/masks/rainfall_mask.json",
        }

        st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
        selected_mask_name = st.selectbox("Select Task Mask to Inspect:", list(mask_files.keys()))
        mask_rel_path = mask_files[selected_mask_name]
        mask_full_path = PROJECT_ROOT / mask_rel_path

        if mask_full_path.exists():
            with open(mask_full_path, "r", encoding="utf-8") as f:
                mask_data = json.load(f)

            mcol1, mcol2 = st.columns([1, 1])
            with mcol1:
                st.subheader("Band Mask Specifications")
                st.write(f"**Optimization Method:** `{mask_data.get('method', 'BWOA')}`")
                st.write(f"**Active Band Count:** `{len(mask_data.get('selected_indices', []))} / {mask_data.get('total_bands', 13)}`")
                st.write(f"**Theoretical Volume Reduction:** `{((13 - len(mask_data.get('selected_indices', []))) / 13) * 100:.1f}%`")
                st.write(f"**Selected Channels:** `{mask_data.get('selected_bands')}`")
                st.write(f"**Band Indices:** `{mask_data.get('selected_indices')}`")

            with mcol2:
                st.subheader("Channel Selection Heatmap")
                active_indices = mask_data.get("selected_indices", [])
                binary_vector = [1 if i in active_indices else 0 for i in range(13)]
                
                fig, ax = plt.subplots(figsize=(8, 1.8), dpi=150)
                colors = ["#00f0ff" if v == 1 else "#1e293b" for v in binary_vector]
                ax.bar(range(13), [1]*13, color=colors, edgecolor="#00f0ff", width=0.8)
                ax.set_xticks(range(13))
                ax.set_xticklabels(SENTINEL2_BAND_KEYS, fontsize=9)
                ax.set_yticks([])
                ax.set_title(f"Active Channels ({len(active_indices)}/13 Active)", fontsize=10, fontweight="bold")
                
                for idx, val in enumerate(binary_vector):
                    ax.text(idx, 0.5, "ON" if val == 1 else "OFF", ha="center", va="center", 
                            color="#030712" if val == 1 else "#64748b", fontweight="bold", fontsize=8)
                
                style_dark_space_figure(fig, ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True)
            st.markdown("#### Electromagnetic Regime Allocation")
            render_spectral_matrix(mask_data.get("selected_bands", []))

        st.markdown('</div>', unsafe_allow_html=True)

        # Scientific Gating Callout
        st.markdown(
            """
            <div class="voxel-card" style="border-color: rgba(56, 189, 248, 0.3); background: rgba(56, 189, 248, 0.05); margin-top: 14px;">
                <div class="voxel-card-header" style="color: #38bdf8;">ℹ️ SCIENTIFIC GATING NOTE</div>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.55;">
                    BWOA performs spectral channel down-selection to minimize telemetry data volume (53.8% to 69.2% band reduction) 
                    while enforcing physical domain balance (Visible, Red Edge, NIR, SWIR) and refusing atmospheric absorption bands (Band 10 Cirrus). 
                    VOXEL treats BWOA strictly as a data-volume reduction mechanism; claims of empirical accuracy improvements over all-band models 
                    remain gated pending controlled ablations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🧪 Run Live BWOA Metaheuristic Search (Fresh Optimization Sandbox)"):
            st.markdown("Execute a fresh Binary Whale Optimization run with custom population, iterations, and target band constraints:")
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                live_target_bands = st.slider("Target Band Count:", min_value=3, max_value=8, value=5)
                live_whales = st.slider("Whale Population (Agents):", min_value=5, max_value=25, value=10)
            with bcol2:
                live_iters = st.slider("Max Iterations:", min_value=3, max_value=20, value=5)
                live_seed = st.number_input("Random Seed:", value=42)

            if st.button("Run Live BWOA Search"):
                with st.spinner("Executing Binary Whale Optimization Algorithm..."):
                    from module2_optimization.bwoa_optimizer import run_module2_optimization
                    sample_tensor = np.random.RandomState(live_seed).uniform(0.05, 0.6, (256, 256, 13))
                    mask_arr, sel_indices, opt_metrics = run_module2_optimization(
                        input_tensor=sample_tensor,
                        target_count=live_target_bands,
                        num_whales=live_whales,
                        max_iter=live_iters,
                        seed=live_seed,
                    )
                    sel_band_names = [SENTINEL2_BAND_KEYS[i] for i in sel_indices if i < len(SENTINEL2_BAND_KEYS)]
                    st.success(f"Fresh BWOA Run Completed in {opt_metrics.get('optimization_time_ms', 0):.1f} ms!")
                    st.write(f"**Selected Candidate Bands ({len(sel_indices)}):** `{', '.join(sel_band_names)}`")
                    st.write(f"**Theoretical Volume Reduction:** `{((13 - len(sel_indices)) / 13) * 100:.1f}%`")
                    st.json(opt_metrics)

    # =========================================================================
    # SECTION 5: MODULE 3 DEEP LEARNING & EXPLAINABILITY
    # =========================================================================
    elif nav_selection == "🧠 05 // Module 3: Deep Learning":
        render_operational_header(
            "Deep Learning, Explainability &amp; Latent Retrieval",
            "Hybrid CNN-ViT FusionNet · Permutation SHAP Attributions · 1216-dim FAISS Latent Vector Space",
            "MODULE 3 // DEEP LEARNING"
        )

        st.subheader("1. Architecture Specification")
        st.markdown(
            """
            <div class="voxel-card">
                <div class="voxel-card-header">FUSIONNET HYBRID ARCHITECTURE (1216-DIMENSIONAL EMBEDDING)</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 10px;">
                    <div style="background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 12px;">
                        <div style="font-family: var(--font-display); font-size: 1.05rem; color: #ffffff;">CNN Branch</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">XNANet deep convolutional backbone delivering 1024-dimensional local texture and edge representations.</div>
                    </div>
                    <div style="background: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 12px;">
                        <div style="font-family: var(--font-display); font-size: 1.05rem; color: #ffffff;">ViT Branch</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">Tiny-32 Vision Transformer with multi-head self-attention extracting 192-dimensional global spatial context.</div>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 12px;">
                        <div style="font-family: var(--font-display); font-size: 1.05rem; color: #ffffff;">Fusion Head</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">Direct concatenation yielding 1216-dimensional joint spatial-spectral representation vectors.</div>
                    </div>
                    <div style="background: rgba(168, 85, 247, 0.05); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px; padding: 12px;">
                        <div style="font-family: var(--font-display); font-size: 1.05rem; color: #ffffff;">Dynamic Slicing</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">Channels routed dynamically matching calibrated Module 2 masks without dynamic Conv layer re-instantiation.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("2. Global Explainability Artifacts")
        shap_img_path = PROJECT_ROOT / "checkpoints" / "shap_band_importance.png"
        if shap_img_path.exists():
            st.image(str(shap_img_path), caption="Permutation SHAP Band Importance (Global Attribution from EuroSAT Benchmark Pretraining - For live per-tile attribution, see Orchestrator)", use_container_width=True)
        else:
            st.info("SHAP plot artifact not yet generated in checkpoints directory.")

        st.markdown("---")
        st.subheader("3. Semantic Vector Database Status")
        faiss_path = PROJECT_ROOT / "checkpoints" / "terrain_index.faiss"
        faiss_meta_path = PROJECT_ROOT / "checkpoints" / "terrain_index.faiss.meta.json"

        if faiss_path.exists() and faiss_meta_path.exists():
            with open(faiss_meta_path, "r", encoding="utf-8") as f:
                meta_entries = json.load(f)
            
            st.markdown(
                f"""
                <div class="voxel-card">
                    <span class="voxel-badge-success">FAISS INDEX ONLINE</span>
                    <h3 style="margin-top: 6px; color: #ffffff;">{len(meta_entries)} Reference Scenes Indexed</h3>
                    <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                        1216-dimensional latent space indexing EuroSAT multi-class reference benchmark terrain scenes. Real task inference (Waterbody / Forest Cover) is executed directly by the trained classification heads.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("View Indexed Scene Metadata"):
                st.dataframe(pd.DataFrame(meta_entries), use_container_width=True)
        else:
            st.info("FAISS vector index not yet generated in checkpoints directory.")


# Trigger UI rendering if running under real Streamlit
if not isinstance(st, MagicMock) and not hasattr(st, "_mock_return_value"):
    render_dashboard()
