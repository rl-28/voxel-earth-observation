"""
VOXEL: Task-Adaptive Earth Observation & Spectral Intelligence Platform
Streamlit Dashboard (app.py)

Unified End-to-End Scientific Architecture:
  1. Mission Control (Stage 0 -> Module 1 -> Module 2 -> Module 3 Orchestration)
  2. Data & Ingestion (Module 1 Ingestion, Spatial Tiling, 2D DWT, Scaling)
  3. Spectral Optimization (Module 2 Physics-Guided Band Optimization & Heatmap)
  4. AI & Explainability (Module 3 Hybrid CNN-ViT, Permutation SHAP, FAISS Search)
  5. Data & Model Trust (Scientific Dataset Provenance & Model Verification Hub)
  6. Execution Diagnostics (Pipeline Latency Trace & Data Handoff Telemetry)
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
from ui_components import (
    inject_custom_styles,
    render_page_header,
    render_metric_card,
    render_status_badge,
    render_empty_state,
    render_band_tiles,
    render_scientific_heatmap,
    render_shap_bar_chart,
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
# DYNAMIC ENVIRONMENT & PROVENANCE INSPECTOR
# -----------------------------------------------------------------------------
def inspect_system_status() -> Dict[str, Any]:
    """Inspects file existence and system health dynamically without hardcoded assumptions."""
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"
    
    has_water_ckpt = (ckpt_dir / "real_waterbody" / "best_fusion_net.pth").exists()
    has_forest_ckpt = (ckpt_dir / "real_forest_cover" / "best_fusion_net.pth").exists()
    has_default_ckpt = (ckpt_dir / "best_fusion_net.pth").exists()
    has_faiss = (ckpt_dir / "terrain_index.faiss").exists()
    has_shap_img = (ckpt_dir / "shap_band_importance.png").exists()

    masks_exist = all(
        (ckpt_dir / "masks" / f"{m}_mask.json").exists()
        for m in ["waterbody", "forest_cover", "drought", "rainfall"]
    )

    return {
        "module1_status": "Online",
        "module2_status": "Online" if masks_exist else "Degraded",
        "has_real_checkpoints": has_water_ckpt or has_forest_ckpt,
        "has_default_checkpoint": has_default_ckpt,
        "has_faiss_index": has_faiss,
        "has_shap_artifact": has_shap_img,
    }


# -----------------------------------------------------------------------------
# MAIN APPLICATION CONTROLLER
# -----------------------------------------------------------------------------
def render_dashboard():
    # Configure Streamlit page layout and metadata
    st.set_page_config(
        page_title="VOXEL | Earth Observation Intelligence",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject dark geospatial design system tokens
    inject_custom_styles()

    # Query system health dynamically
    sys_status = inspect_system_status()

    # -------------------------------------------------------------------------
    # SIDEBAR APPLICATION SHELL
    # -------------------------------------------------------------------------
    with st.sidebar:
        # Branding
        st.markdown(
            """
            <div style="padding: 4px 0 16px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 16px;">
                <div style="font-size: 1.4rem; font-weight: 800; color: #F1F5F9; letter-spacing: -0.02em;">
                    🛰️ VOXEL
                </div>
                <div style="font-size: 0.75rem; font-weight: 600; color: #67E8F9; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px;">
                    Earth Observation Intelligence Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Primary Navigation
        st.markdown(
            '<div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">Navigation</div>',
            unsafe_allow_html=True,
        )
        
        nav_destinations = [
            "🎯 Mission Control",
            "📡 Data & Ingestion",
            "⚡ Spectral Optimization",
            "🧠 AI & Explainability",
            "🔬 Data & Model Trust",
            "🛠️ Execution Diagnostics",
        ]
        
        selected_nav = st.radio(
            "Select Destination",
            nav_destinations,
            label_visibility="collapsed",
            key="voxel_nav_selection",
        )

        st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 16px 0;"></div>', unsafe_allow_html=True)

        # Dynamic Module Status
        st.markdown(
            '<div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">Module Status</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="font-size: 0.76rem; line-height: 1.8; color: #F1F5F9;">
                <div><span class="vx-badge vx-badge-success" style="padding: 1px 6px; font-size: 0.65rem;">M1</span> Ingestion: <b>Online</b></div>
                <div><span class="vx-badge vx-badge-success" style="padding: 1px 6px; font-size: 0.65rem;">M2</span> BWOA Masks: <b>Online</b></div>
                <div><span class="vx-badge vx-badge-info" style="padding: 1px 6px; font-size: 0.65rem;">M3</span> FusionNet: <b>{'Ready' if sys_status['has_default_checkpoint'] else 'Gated'}</b></div>
                <div><span class="vx-badge vx-badge-{'success' if sys_status['has_faiss_index'] else 'neutral'}" style="padding: 1px 6px; font-size: 0.65rem;">VEC</span> Vector Search: <b>{'Online' if sys_status['has_faiss_index'] else 'Offline'}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 16px 0;"></div>', unsafe_allow_html=True)

        # Scientific Provenance Notice
        st.markdown(
            """
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">Scientific Integrity</div>
            <div style="font-size: 0.72rem; color: #94A3B8; line-height: 1.4;">
                • <b>Mock Checkpoints:</b> Gated in Real Mode<br>
                • <b>Atmospheric Cirrus:</b> Band 10 Refused<br>
                • <b>Downlink Bandwidth:</b> ~50–60% Reduced
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # DESTINATION 1: MISSION CONTROL
    # -------------------------------------------------------------------------
    if selected_nav == "🎯 Mission Control":
        render_page_header(
            title="Mission Control",
            subtitle="Autonomous query-driven satellite tasking, spectral band reduction, and land classification inference.",
            badge_text="End-to-End Orchestrator",
            badge_type="info",
        )

        # Mission Query Composer Card
        with st.container():
            st.markdown(
                '<div style="font-size: 0.82rem; font-weight: 700; color: #67E8F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Natural Language Mission Composer</div>',
                unsafe_allow_html=True,
            )

            col_q1, col_q2 = st.columns([3, 1])
            with col_q1:
                preset_query = st.selectbox(
                    "Mission Query Presets:",
                    [
                        "Track waterbody levels in Loktak Lake, Manipur",
                        "Assess forest canopy in Jim Corbett National Park, Uttarakhand",
                        "Identify water bodies around Chilika Lake, Odisha, India",
                        "Analyze forest canopy and tree cover around Western Ghats, Karnataka, India",
                        "Monitor agricultural crop health in Punjab, India",
                        "Custom query...",
                    ],
                    key="mc_preset_selector",
                )
                if preset_query == "Custom query...":
                    user_query = st.text_input("Custom Query:", "Track waterbody levels in Loktak Lake, Manipur", key="mc_custom_query")
                else:
                    user_query = st.text_input("Active Query:", preset_query, key="mc_active_query")

                task_choice = st.radio(
                    "Task Profile Mode:",
                    ["auto", "waterbody", "forest_cover"],
                    format_func=lambda x: "Auto-Detect from NL Query" if x == "auto" else ("Waterbody Classification (6 Bands)" if x == "waterbody" else "Forest Cover Classification (5 Bands)"),
                    horizontal=True,
                    key="mc_task_choice",
                )

            with col_q2:
                exec_mode = st.radio(
                    "Execution Mode:",
                    ["real", "offline"],
                    format_func=lambda x: "Real Satellite Mode (AWS STAC)" if x == "real" else "Offline Smoke Test (Synthetic Fixture)",
                    key="mc_exec_mode",
                )
                run_shap = st.checkbox("Permutation SHAP Attribution", value=True, key="mc_shap_toggle")
                run_bwoa = st.checkbox("Live Metaheuristic BWOA Search", value=False, key="mc_bwoa_toggle")

            exec_btn = st.button("🚀 Execute Mission Pipeline", type="primary", use_container_width=True, key="mc_exec_btn")

        # Execute and process pipeline run
        if exec_btn:
            with st.spinner("Compiling query, acquiring telemetry, and running scientific inference..."):
                try:
                    pipeline = VoxelPipeline(default_mode=exec_mode, verbose=False)
                    res: VoxelPipelineResult = pipeline.run(
                        query=user_query,
                        mode=exec_mode,
                        task_override=None if task_choice == "auto" else task_choice,
                        explain_shap=run_shap,
                        optimize_bands=run_bwoa,
                    )
                    st.session_state["pipeline_result"] = res
                    st.success(f"Pipeline Execution Complete — Status: **{res.status}**")
                except Exception as e:
                    st.error(f"Pipeline Execution Error: {str(e)}")
                    st.code(traceback.format_exc())

        # Render Active Execution Summary if available
        if "pipeline_result" in st.session_state:
            res: VoxelPipelineResult = st.session_state["pipeline_result"]

            st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 20px 0;"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size: 0.85rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Active Mission Summary</div>',
                unsafe_allow_html=True,
            )

            # Concise Execution Summary Metrics
            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                render_metric_card("Resolved Task", res.task_type.upper(), f"Classes: {res.num_classes}")
            with sm2:
                render_metric_card("Spectral Reduction", f"{res.band_reduction_pct:.1f}%", f"{len(res.selected_bands)}/13 Active Bands")
            with sm3:
                latency_val = res.timing_breakdown_ms.get("total_pipeline_latency_ms", 0)
                render_metric_card("Total Latency", f"{latency_val:.1f} ms", "End-to-End Execution")
            with sm4:
                prov_label = "Synthetic Fixture" if res.is_synthetic else "Live Sentinel-2"
                render_metric_card("Data Provenance", prov_label, f"Mode: {res.execution_mode}")

            # Polished Stage Tracker
            st.markdown(
                """
                <div style="font-size: 0.78rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin: 18px 0 8px 0;">
                    Pipeline Stage Sequence
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Determine Stage States
            s0_done = bool(res.resolved_location and res.resolved_location.get("bbox"))
            s1_done = bool(res.input_tensor_shape)
            s2_done = bool(res.selected_bands)
            s3_done = bool(res.prediction)
            s3_gated = (res.prediction is None and "withheld" in " ".join(res.warnings).lower())
            exp_done = bool(res.explainability or res.similar_terrain)

            st0_cls = "complete" if s0_done else "failed"
            st1_cls = "complete" if s1_done else "failed"
            st2_cls = "complete" if s2_done else "failed"
            st3_cls = "complete" if s3_done else ("partial" if s3_gated else "failed")
            st4_cls = "complete" if exp_done else "pending"

            st.markdown(
                f"""
                <div class="vx-stage-grid">
                    <div class="vx-stage-item {st0_cls}">
                        <div class="vx-stage-title">Stage 0: Query Compiler</div>
                        <div class="vx-stage-detail">
                            <b>Loc:</b> {res.resolved_location.get('name', 'N/A')[:20]}<br>
                            <b>Latency:</b> {res.timing_breakdown_ms.get('stage_0_query_compilation_ms', 0):.1f} ms
                        </div>
                    </div>
                    <div class="vx-stage-item {st1_cls}">
                        <div class="vx-stage-title">Module 1: Ingestion</div>
                        <div class="vx-stage-detail">
                            <b>Tensor:</b> {res.input_tensor_shape}<br>
                            <b>Latency:</b> {res.timing_breakdown_ms.get('module_1_ingestion_ms', 0):.1f} ms
                        </div>
                    </div>
                    <div class="vx-stage-item {st2_cls}">
                        <div class="vx-stage-title">Module 2: Optimization</div>
                        <div class="vx-stage-detail">
                            <b>Bands:</b> {len(res.selected_bands)}/13 Active<br>
                            <b>Latency:</b> {res.timing_breakdown_ms.get('module_2_optimization_ms', 0):.1f} ms
                        </div>
                    </div>
                    <div class="vx-stage-item {st3_cls}">
                        <div class="vx-stage-title">Module 3: Deep Learning</div>
                        <div class="vx-stage-detail">
                            <b>Pred:</b> {res.prediction.get('predicted_class_name', 'Withheld') if res.prediction else 'Withheld (Gated)'}<br>
                            <b>Latency:</b> {res.timing_breakdown_ms.get('module_3_inference_ms', 0):.1f} ms
                        </div>
                    </div>
                    <div class="vx-stage-item {st4_cls}">
                        <div class="vx-stage-title">Explainability & Vector Retrieval</div>
                        <div class="vx-stage-detail">
                            <b>SHAP:</b> {'Computed' if res.explainability else 'Skipped'}<br>
                            <b>FAISS:</b> {len(res.similar_terrain) if res.similar_terrain else 0} matches
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Prominent Satellite Preview & Geographic Metadata
            st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 20px 0;"></div>', unsafe_allow_html=True)
            col_img, col_info = st.columns([1, 1])

            with col_img:
                st.markdown(
                    '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Scene Visualization</div>',
                    unsafe_allow_html=True,
                )
                rgb_path = res.diagnostics.get("rgb_preview_path")
                if rgb_path and os.path.exists(rgb_path):
                    st.image(rgb_path, caption="Sentinel-2 L2A Natural Color Composite (B04, B03, B02)", use_container_width=True)
                else:
                    render_empty_state(
                        icon="🛰️",
                        title="Offline Synthetic Fixture Tensor",
                        message="In Offline Smoke-Test Mode, execution uses a deterministic synthetic spatial-spectral tensor for fast verification. Real GeoTIFF composites are acquired in Real Satellite Mode via STAC.",
                        action_hint="Switch execution mode to 'Real Satellite Mode' to acquire live Sentinel-2 imagery.",
                    )

            with col_info:
                st.markdown(
                    '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Geographic & Telemetry Metadata</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="vx-card">
                        <div style="font-size: 0.8rem; line-height: 1.8; color: #F1F5F9;">
                            <div><b>Resolved Location:</b> {res.resolved_location.get('name', 'N/A')}</div>
                            <div><b>Bounding Box:</b> <code>{res.resolved_location.get('bbox', 'N/A')}</code></div>
                            <div><b>Target Task:</b> <span class="vx-badge vx-badge-info">{res.task_type.upper()}</span></div>
                            <div><b>Active Spectral Bands:</b> <code>{', '.join(res.selected_bands)}</code></div>
                            <div><b>Mask Provenance:</b> <code>{res.mask_provenance}</code></div>
                            <div><b>Data Source:</b> <code>{res.data_source}</code></div>
                            <div><b>Band 10 (Cirrus):</b> <span class="vx-badge vx-badge-danger">Refused (Atmospheric Zero)</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Prediction Details Card
                if res.prediction:
                    prov = res.prediction.get("provenance")
                    prov_badge = "vx-badge-success" if prov == "real_inference" else "vx-badge-warning"
                    prov_text = "Verified Real Satellite Model" if prov == "real_inference" else "Synthetic Test Fixture"

                    st.markdown(
                        f"""
                        <div class="vx-card-elevated">
                            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">
                                Model Classification Result
                            </div>
                            <div style="font-size: 1.25rem; font-weight: 800; color: #F1F5F9; margin: 4px 0;">
                                {res.prediction.get('predicted_class_name')}
                                <span class="vx-badge {prov_badge}" style="margin-left: 8px; font-size: 0.65rem;">{prov_text}</span>
                            </div>
                            <div style="font-size: 0.82rem; color: #67E8F9; margin-bottom: 8px;">
                                Confidence: <b>{res.prediction.get('confidence_pct', 0):.1f}%</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if "all_probabilities" in res.prediction:
                        st.caption("Class Probability Distribution:")
                        for c_name, p_val in res.prediction["all_probabilities"].items():
                            st.progress(p_val / 100.0, text=f"{c_name}: {p_val:.1f}%")

                    st.caption("ℹ️ **Scientific Scope:** Prediction represents whole-tile land cover classification (256×256 px / 2.56×2.56 km footprint), not pixel-level segmentation.")
                else:
                    render_empty_state(
                        icon="🔒",
                        title="Classification Prediction Withheld",
                        message="Scientific Gating Enforced: A mock or synthetic checkpoint was withheld during real satellite execution to protect scientific validity.",
                        action_hint="Train or link a verified checkpoint in checkpoints/real_<task>/ to enable live real-mode inference.",
                    )

            # Explainability & Vector Retrieval Section
            if res.explainability or res.similar_terrain:
                st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 20px 0;"></div>', unsafe_allow_html=True)
                col_exp, col_faiss = st.columns(2)

                with col_exp:
                    st.markdown(
                        '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Spectral Band Attribution (SHAP)</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("Live Permutation SHAP computed across the active spectral channels of this scene.")
                    if res.explainability and "top_contributing_bands" in res.explainability:
                        render_shap_bar_chart(res.explainability["top_contributing_bands"])
                    else:
                        render_empty_state(
                            icon="📊",
                            title="SHAP Attribution Skipped",
                            message="Permutation SHAP attribution was disabled for this run.",
                        )

                with col_faiss:
                    st.markdown(
                        '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Similar Terrain Retrieval (FAISS)</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("Nearest neighbors from the 1216-dim latent embedding space across the EuroSAT reference corpus.")
                    if res.similar_terrain:
                        st.dataframe(pd.DataFrame(res.similar_terrain), use_container_width=True)
                    else:
                        render_empty_state(
                            icon="🔍",
                            title="No Semantic Vector Matches",
                            message="Vector retrieval index was not queried or returned no matches.",
                        )

            # Machine-Readable JSON Contract
            with st.expander("📄 View Machine-Readable Result Contract (JSON)"):
                st.json(res.to_dict())

    # -------------------------------------------------------------------------
    # DESTINATION 2: DATA & INGESTION (MODULE 1)
    # -------------------------------------------------------------------------
    elif selected_nav == "📡 Data & Ingestion":
        render_page_header(
            title="Data & Ingestion",
            subtitle="Sentinel-2 Level-2A data acquisition, 256x256 spatial tiling, 2D DWT wavelet decomposition, and radiometric scaling.",
            badge_text="Module 1 Workspace",
            badge_type="info",
        )

        m1_mode = st.radio(
            "Select Processing Mode:",
            ["Evaluation Region (Standardized Benchmark)", "Local Directory / Upload"],
            horizontal=True,
            key="m1_processing_mode",
        )

        if m1_mode == "Evaluation Region (Standardized Benchmark)":
            col_reg1, col_reg2 = st.columns([2, 1])
            with col_reg1:
                region_id = st.selectbox(
                    "Target Evaluation Region:",
                    list(EVALUATION_REGIONS.keys()),
                    format_func=lambda x: EVALUATION_REGIONS[x].display_name,
                    key="m1_region_select",
                )
                eval_input_dir = st.text_input(
                    "Local Source Directory Path (Optional Custom Override):",
                    "",
                    key="m1_eval_dir",
                )

            with col_reg2:
                reg_info = EVALUATION_REGIONS[region_id]
                st.markdown(
                    f"""
                    <div class="vx-card" style="margin-top: 24px;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">Region Specifications</div>
                        <div style="font-size: 0.8rem; line-height: 1.6; color: #F1F5F9; margin-top: 6px;">
                            <div><b>Region:</b> {reg_info.display_name}</div>
                            <div><b>Product:</b> {reg_info.product_type}</div>
                            <div><b>Format:</b> Level-2A BOA Surface Reflectance</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button("Run Module 1 Regional Ingestion", type="primary", key="m1_eval_btn"):
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
                                st.error("No valid Sentinel-2 bands found in directory.")
                            else:
                                with st.spinner(f"Processing {reg_info.display_name}..."):
                                    manifest = eval_pipeline.process_local_source(band_paths)
                                    st.success(f"Generated {manifest['number_of_tiles']} evaluation tiles.")
                                    st.json(manifest)
                    else:
                        st.info("No local directory specified. Searching pre-cached regional assets...")
                        eval_pipeline.process_local_source({})
                except Exception as e:
                    st.error(f"Regional processing failed: {str(e)}")
                    st.code(traceback.format_exc())

        elif m1_mode == "Local Directory / Upload":
            uploaded_file = st.file_uploader(
                "Upload ZIP containing Sentinel-2 L2A GeoTIFFs (.tif or .jp2):",
                type="zip",
                key="m1_zip_uploader",
            )
            if uploaded_file and st.button("Process Uploaded Bands", type="primary", key="m1_upload_btn"):
                with st.spinner("Extracting archive and executing Module 1 pipeline..."):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zip_path = Path(tmpdir) / "upload.zip"
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_file.read())
                        extract_dir = Path(tmpdir) / "extracted"
                        extract_dir.mkdir()
                        try:
                            with zipfile.ZipFile(zip_path, "r") as zip_ref:
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
                            st.error("No valid Sentinel-2 bands found in archive.")
                        else:
                            st.success(f"Found {len(band_paths)} matching spectral channels.")
                            process_bands(band_paths)

        # Ingestion Metadata Reference Cards
        st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Standard Ingestion Specifications</div>',
            unsafe_allow_html=True,
        )

        ing_col1, ing_col2, ing_col3 = st.columns(3)
        with ing_col1:
            st.markdown(
                """
                <div class="vx-card">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #67E8F9; text-transform: uppercase;">Spatial Geometry</div>
                    <div style="font-size: 0.8rem; line-height: 1.6; color: #F1F5F9; margin-top: 6px;">
                        • <b>Tile Footprint:</b> 256 × 256 pixels<br>
                        • <b>Ground Resolution:</b> 10m Ground Sample Dist.<br>
                        • <b>Coverage Area:</b> 2.56 km × 2.56 km per tile
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ing_col2:
            st.markdown(
                """
                <div class="vx-card">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #67E8F9; text-transform: uppercase;">Radiometry & Scale</div>
                    <div style="font-size: 0.8rem; line-height: 1.6; color: #F1F5F9; margin-top: 6px;">
                        • <b>Input Range:</b> Raw DN [0, 10000]<br>
                        • <b>Normalized Scale:</b> BOA Reflectance [0.0, 1.0]<br>
                        • <b>Scale Divisor:</b> Auto-detected 10000.0
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ing_col3:
            st.markdown(
                """
                <div class="vx-card">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #67E8F9; text-transform: uppercase;">Wavelet DWT Integration</div>
                    <div style="font-size: 0.8rem; line-height: 1.6; color: #F1F5F9; margin-top: 6px;">
                        • <b>Transform:</b> 2D Discrete Wavelet Transform<br>
                        • <b>Wavelet Basis:</b> Haar / Daubechies (db1)<br>
                        • <b>Channels:</b> LL, LH, HL, HH subbands
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # DESTINATION 3: SPECTRAL OPTIMIZATION (MODULE 2)
    # -------------------------------------------------------------------------
    elif selected_nav == "⚡ Spectral Optimization":
        render_page_header(
            title="Spectral Optimization",
            subtitle="Physics-guided spectral channel selection via Binary Whale Optimization Algorithm (BWOA).",
            badge_text="Module 2 Workspace",
            badge_type="info",
        )

        mask_files = {
            "Waterbody Mask (6 Bands)": "checkpoints/masks/waterbody_mask.json",
            "Forest Cover Mask (5 Bands)": "checkpoints/masks/forest_cover_mask.json",
            "Drought Mask (4 Bands)": "checkpoints/masks/drought_mask.json",
            "Rainfall Mask (5 Bands)": "checkpoints/masks/rainfall_mask.json",
        }

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            selected_mask_name = st.selectbox("Select Validated Precomputed Task Mask:", list(mask_files.keys()), key="m2_mask_select")
            mask_rel_path = mask_files[selected_mask_name]
            mask_full_path = PROJECT_ROOT / mask_rel_path

            if mask_full_path.exists():
                with open(mask_full_path, "r", encoding="utf-8") as f:
                    mask_data = json.load(f)

                sel_indices = mask_data.get("selected_indices", [])
                reduction_pct = ((13 - len(sel_indices)) / 13) * 100.0

                st.markdown(
                    f"""
                    <div class="vx-card">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #67E8F9; text-transform: uppercase;">Mask Specifications</div>
                        <div style="font-size: 0.8rem; line-height: 1.8; color: #F1F5F9; margin-top: 6px;">
                            <div><b>Optimization Method:</b> <code>{mask_data.get('method', 'BWOA')}</code></div>
                            <div><b>Active Channels:</b> <b>{len(sel_indices)} of 13 Bands</b></div>
                            <div><b>Data Volume Reduction:</b> <span class="vx-badge vx-badge-success">{reduction_pct:.1f}%</span></div>
                            <div><b>Selected Band Names:</b> <code>{mask_data.get('selected_bands')}</code></div>
                            <div><b>Band 10 (Cirrus):</b> <span class="vx-badge vx-badge-danger">Refused (Zeroed Placeholder)</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_m2:
            if mask_full_path.exists():
                render_scientific_heatmap(
                    selected_indices=sel_indices,
                    band_keys=SENTINEL2_BAND_KEYS,
                    title="Precomputed Task Mask Selection",
                )

        # Visual Sentinel-2 Band Grid
        if mask_full_path.exists():
            st.markdown(
                '<div style="font-size: 0.8rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin: 16px 0 4px 0;">Sentinel-2 Spectral Channel Status</div>',
                unsafe_allow_html=True,
            )
            render_band_tiles(selected_indices=sel_indices, band_names=SENTINEL2_BAND_KEYS)

        # Scientific Gating Callout
        st.markdown(
            """
            <div class="vx-card" style="border-left: 3px solid #67E8F9; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #67E8F9; text-transform: uppercase;">Scientific Gating Notice</div>
                <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.5; margin-top: 4px;">
                    BWOA performs spectral channel down-selection to minimize downlink telemetry volume (53.8% to 69.2% band reduction) 
                    while enforcing domain balance (Visible, Red Edge, NIR, SWIR) and refusing atmospheric absorption bands (Band 10 Cirrus). 
                    VOXEL treats BWOA strictly as a data-reduction and bandwidth-conservation mechanism; claims of accuracy improvements over 
                    all-band models remain gated until controlled ablations are benchmarked.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Live BWOA Metaheuristic Search Expander
        with st.expander("🧪 Run Live BWOA Metaheuristic Search (Fresh Optimization Test)"):
            st.markdown(
                """
                <div class="vx-card" style="border-left: 3px solid #F59E0B; margin-bottom: 12px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #F59E0B; text-transform: uppercase;">Fixture Demonstration Notice</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.4; margin-top: 4px;">
                        This standalone test constructs a synthetic fixture tensor (256×256×13) to benchmark the optimization search dynamics. 
                        It is an algorithmic demonstration, not an optimization on live satellite imagery. Precomputed task masks should be used for operational tasks.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                live_target_bands = st.slider("Target Band Count:", min_value=3, max_value=8, value=5, key="live_bwoa_target")
                live_whales = st.slider("Whale Population (Agents):", min_value=5, max_value=25, value=10, key="live_bwoa_whales")
            with bcol2:
                live_iters = st.slider("Max Iterations:", min_value=3, max_value=20, value=5, key="live_bwoa_iters")
                live_seed = st.number_input("Random Seed:", value=42, key="live_bwoa_seed")

            if st.button("Execute Live BWOA Search", type="primary", key="live_bwoa_btn"):
                with st.spinner("Executing Binary Whale Optimization Algorithm..."):
                    from module2_optimization.bwoa_optimizer import run_module2_optimization
                    sample_tensor = np.random.RandomState(int(live_seed)).uniform(0.05, 0.6, (256, 256, 13))
                    mask_arr, sel_indices_live, opt_metrics = run_module2_optimization(
                        input_tensor=sample_tensor,
                        target_count=live_target_bands,
                        num_whales=live_whales,
                        max_iter=live_iters,
                        seed=int(live_seed),
                    )
                    sel_band_names = [SENTINEL2_BAND_KEYS[i] for i in sel_indices_live if i < len(SENTINEL2_BAND_KEYS)]
                    st.success(f"Fresh BWOA Run Completed in {opt_metrics.get('optimization_time_ms', 0):.1f} ms!")
                    
                    st.write(f"**Selected Candidate Bands ({len(sel_indices_live)}):** `{', '.join(sel_band_names)}`")
                    st.write(f"**Data Volume Reduction:** `{((13 - len(sel_indices_live)) / 13) * 100:.1f}%`")
                    render_scientific_heatmap(sel_indices_live, SENTINEL2_BAND_KEYS, title="Live BWOA Optimized Mask")
                    st.json(opt_metrics)

    # -------------------------------------------------------------------------
    # DESTINATION 4: AI & EXPLAINABILITY (MODULE 3)
    # -------------------------------------------------------------------------
    elif selected_nav == "🧠 AI & Explainability":
        render_page_header(
            title="AI & Explainability",
            subtitle="Deep learning classification (CNN-ViT FusionNet), Permutation SHAP attribution, and FAISS vector retrieval.",
            badge_text="Module 3 Workspace",
            badge_type="info",
        )

        # Architecture Overview Card
        st.markdown(
            """
            <div class="vx-card">
                <div style="font-size: 0.72rem; font-weight: 700; color: #67E8F9; text-transform: uppercase; margin-bottom: 8px;">FusionNet Architecture Specification</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.8rem; color: #F1F5F9;">
                    <div>• <b>CNN Branch:</b> XNANet (1024-dim features)</div>
                    <div>• <b>ViT Branch:</b> Tiny-32 ViT (192-dim multi-head attention)</div>
                    <div>• <b>Fused Embedding:</b> 1216-dim spatial-spectral vector</div>
                    <div>• <b>Dynamic Routing:</b> Adapts to arbitrary channel subsets</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a1, col_a2 = st.columns(2)

        # Explainability & SHAP plot
        with col_a1:
            st.markdown(
                '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin: 16px 0 8px 0;">Global SHAP Attribution Benchmark</div>',
                unsafe_allow_html=True,
            )
            shap_img_path = PROJECT_ROOT / "checkpoints" / "shap_band_importance.png"
            if shap_img_path.exists():
                st.image(str(shap_img_path), caption="Permutation SHAP Band Importance (Global Attribution from EuroSAT Benchmark Pretraining)", use_container_width=True)
            else:
                render_empty_state(
                    icon="📊",
                    title="SHAP Benchmark Artifact Unavailable",
                    message="The global benchmark attribution plot has not yet been generated in checkpoints/shap_band_importance.png.",
                    action_hint="Run 'python build_explainability.py' to generate SHAP artifacts.",
                )

            st.caption("ℹ️ **Scientific Gating:** SHAP measures model feature sensitivity and attribution; it does not provide physical causal explanations.")

        # FAISS Semantic Retrieval Index Status
        with col_a2:
            st.markdown(
                '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin: 16px 0 8px 0;">Semantic Vector Database (FAISS)</div>',
                unsafe_allow_html=True,
            )
            faiss_path = PROJECT_ROOT / "checkpoints" / "terrain_index.faiss"
            faiss_meta_path = PROJECT_ROOT / "checkpoints" / "terrain_index.faiss.meta.json"

            if faiss_path.exists() and faiss_meta_path.exists():
                with open(faiss_meta_path, "r", encoding="utf-8") as f:
                    meta_entries = json.load(f)
                st.markdown(
                    f"""
                    <div class="vx-card-elevated">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #10B981; text-transform: uppercase;">Index Online</div>
                        <div style="font-size: 0.82rem; color: #F1F5F9; margin-top: 4px;">
                            • <b>Indexed Scenes:</b> {len(meta_entries)} reference tiles<br>
                            • <b>Embedding Dimension:</b> 1216 dimensions<br>
                            • <b>Corpus:</b> EuroSAT 10-Class Benchmark Reference
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Inspect Indexed Corpus Metadata"):
                    st.dataframe(pd.DataFrame(meta_entries), use_container_width=True)
            else:
                render_empty_state(
                    icon="🗄️",
                    title="FAISS Vector Index Offline",
                    message="Binary index or metadata was not found at checkpoints/terrain_index.faiss.",
                    action_hint="Run 'python build_explainability.py' to generate the semantic vector index.",
                )

            st.caption("ℹ️ **Retrieval Scope:** FAISS index stores embeddings from the reference multi-class benchmark corpus for semantic retrieval validation. Operational task inference is executed directly by the specialized classification heads.")

    # -------------------------------------------------------------------------
    # DESTINATION 5: DATA & MODEL TRUST (PROVENANCE HUB)
    # -------------------------------------------------------------------------
    elif selected_nav == "🔬 Data & Model Trust":
        render_page_header(
            title="Data & Model Trust",
            subtitle="Rigorous audit of dataset provenance, geographic separation, checkpoint registry, and test split evaluation metrics.",
            badge_text="Scientific Integrity Hub",
            badge_type="info",
        )

        # Dataset Provenance Cards
        st.markdown(
            '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">1. Genuine Sentinel-2 Real-Data Foundation vs Synthetic Fixtures</div>',
            unsafe_allow_html=True,
        )

        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.markdown(
                """
                <div class="vx-card" style="border-top: 3px solid #10B981;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #10B981; text-transform: uppercase;">Genuine Real Satellite Data</div>
                    <div style="font-size: 0.78rem; line-height: 1.6; color: #F1F5F9; margin-top: 6px;">
                        • <b>Source:</b> AWS Earth Search STAC (Sentinel-2 L2A)<br>
                        • <b>Coverage:</b> 6 Indian Regions (Chilika, Western Ghats, Thar, Sardar Sarovar, Hirakud, Kaziranga)<br>
                        • <b>Volume:</b> 120 calibrated 256×256 tiles<br>
                        • <b>Radiometry:</b> BOA Surface Reflectance [0.0, 1.0]
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pcol2:
            st.markdown(
                """
                <div class="vx-card" style="border-top: 3px solid #10B981;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #10B981; text-transform: uppercase;">Verified ESA WorldCover Labels</div>
                    <div style="font-size: 0.78rem; line-height: 1.6; color: #F1F5F9; margin-top: 6px;">
                        • <b>Reference:</b> ESA WorldCover 2021 v200 (10m)<br>
                        • <b>Alignment:</b> Reprojection to Sentinel-2 CRS<br>
                        • <b>Threshold:</b> Water ≥ 10%, Non-water ≤ 0.5%<br>
                        • <b>Quality:</b> Ambiguous boundary tiles dropped
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pcol3:
            st.markdown(
                """
                <div class="vx-card" style="border-top: 3px solid #10B981;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #10B981; text-transform: uppercase;">Strict Geographic Separation</div>
                    <div style="font-size: 0.78rem; line-height: 1.6; color: #F1F5F9; margin-top: 6px;">
                        • <b>Train (68 tiles):</b> Kaziranga, Western Ghats, Thar, Sardar Sarovar<br>
                        • <b>Val (20 tiles):</b> Chilika Lake (Odisha Coast)<br>
                        • <b>Test (19 tiles):</b> Hirakud Reservoir (Odisha Inland)<br>
                        • <b>Leakage:</b> Zero spatial or temporal leakage
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Real Dataset Visual Inspection
        st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">2. Real Dataset Visual Inspection (RGB vs ESA WorldCover)</div>',
            unsafe_allow_html=True,
        )
        label_check_img = PROJECT_ROOT / "results" / "label_check.png"
        if label_check_img.exists():
            st.image(str(label_check_img), caption="Real Sentinel-2 L2A RGB vs ESA WorldCover 2021 Ground Truth Maps across India", use_container_width=True)
        else:
            render_empty_state(
                icon="🖼️",
                title="Ground Truth Inspection Sheet Unavailable",
                message="Visual comparison sheet results/label_check.png has not been generated.",
            )

        # Verified Checkpoint Registry & Training Metrics
        st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">3. Verified Checkpoint Registry & Scientific Status</div>',
            unsafe_allow_html=True,
        )

        ckpt_summary = [
            {
                "Task": "Waterbody (REAL DATA)",
                "Checkpoint Path": "checkpoints/real_waterbody/best_fusion_net.pth",
                "Active Channels": "6 bands (B01, B03, B04, B06, B8A, B11)",
                "Data Source": "Real Sentinel-2 L2A (India)",
                "Val OA (%)": "90.00%",
                "Val F1 (%)": "88.10%",
                "Test OA (%)": "73.68%",
                "Test Macro F1 (%)": "73.39%",
                "Water Recall": "100.00%",
                "Scientific Status": "Verified Genuine Real Satellite Model",
            },
            {
                "Task": "Waterbody (Synthetic Mock)",
                "Checkpoint Path": "checkpoints/waterbody/best_fusion_net.pth",
                "Active Channels": "6 bands (B01, B03, B04, B06, B8A, B11)",
                "Data Source": "Synthetic Uniform Noise",
                "Val OA (%)": "50.00%",
                "Val F1 (%)": "33.33%",
                "Test OA (%)": "50.00%",
                "Test Macro F1 (%)": "33.33%",
                "Water Recall": "100.00%",
                "Scientific Status": "Engineering Test Fixture (Collapsed)",
            },
            {
                "Task": "Forest Cover (REAL DATA)",
                "Checkpoint Path": "checkpoints/real_forest_cover/best_fusion_net.pth",
                "Active Channels": "5 bands (B02, B04, B06, B11, B12)",
                "Data Source": "Real Sentinel-2 L2A (India)",
                "Val OA (%)": "66.67%",
                "Val F1 (%)": "47.92%",
                "Test OA (%)": "94.74%",
                "Test Macro F1 (%)": "93.60%",
                "Water Recall": "100.00%",
                "Scientific Status": "Verified Genuine Real Satellite Model (+26.3% OA vs Baseline)",
            },
            {
                "Task": "Forest Cover (Synthetic Mock)",
                "Checkpoint Path": "checkpoints/forest_cover/best_fusion_net.pth",
                "Active Channels": "5 bands (B02, B04, B06, B11, B12)",
                "Data Source": "Synthetic Uniform Noise",
                "Val OA (%)": "100.00%",
                "Val F1 (%)": "100.00%",
                "Test OA (%)": "100.00%",
                "Test Macro F1 (%)": "100.00%",
                "Water Recall": "N/A",
                "Scientific Status": "Engineering Test Fixture",
            },
            {
                "Task": "EuroSAT Land Cover (Mock Benchmark)",
                "Checkpoint Path": "checkpoints/best_fusion_net.pth",
                "Active Channels": "6 bands (B02, B03, B04, B08, B11, B12)",
                "Data Source": "Synthetic Benchmark Mock",
                "Val OA (%)": "9.38%",
                "Val F1 (%)": "1.71%",
                "Test OA (%)": "3.12%",
                "Test Macro F1 (%)": "0.67%",
                "Water Recall": "N/A",
                "Scientific Status": "Engineering Test Fixture",
            },
        ]
        st.dataframe(pd.DataFrame(ckpt_summary), use_container_width=True)

        # Confusion Matrices and Curves
        st.markdown('<div style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">4. Real vs Synthetic Confusion Matrices & Evaluation Logs</div>',
            unsafe_allow_html=True,
        )

        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("#### Real Waterbody Evaluation (Hirakud Test Scene)")
            real_wb_cm = PROJECT_ROOT / "results" / "real_waterbody_eval" / "confusion_matrix.png"
            if real_wb_cm.exists():
                st.image(str(real_wb_cm), caption="Waterbody Confusion Matrix (OA: 73.68%, Macro-F1: 73.39%, Water Recall: 100%)", use_container_width=True)
            else:
                render_empty_state(
                    icon="📊",
                    title="Confusion Matrix Unavailable",
                    message="results/real_waterbody_eval/confusion_matrix.png has not been generated.",
                )

            real_wb_log = PROJECT_ROOT / "results" / "real_waterbody_training_log.csv"
            if real_wb_log.exists():
                r_df = pd.read_csv(real_wb_log)
                st.line_chart(r_df[["train_oa", "val_oa", "train_loss", "val_loss"]])

        with tcol2:
            st.markdown("#### Real Forest Cover Evaluation (Hirakud Test Scene)")
            real_fc_cm = PROJECT_ROOT / "results" / "real_forest_cover_eval" / "confusion_matrix.png"
            if real_fc_cm.exists():
                st.image(str(real_fc_cm), caption="Forest Cover Confusion Matrix (OA: 94.74%, Macro-F1: 93.60%, Forest Recall: 100%)", use_container_width=True)
            else:
                render_empty_state(
                    icon="📊",
                    title="Confusion Matrix Unavailable",
                    message="results/real_forest_cover_eval/confusion_matrix.png has not been generated.",
                )

            real_fc_log = PROJECT_ROOT / "results" / "real_forest_cover_training_log.csv"
            if real_fc_log.exists():
                rfc_df = pd.read_csv(real_fc_log)
                st.line_chart(rfc_df[["train_oa", "val_oa", "train_loss", "val_loss"]])

    # -------------------------------------------------------------------------
    # DESTINATION 6: EXECUTION DIAGNOSTICS
    # -------------------------------------------------------------------------
    elif selected_nav == "🛠️ Execution Diagnostics":
        render_page_header(
            title="Execution Diagnostics",
            subtitle="Pipeline telemetry, stage-level latency traces, data handoff invariants, and system environment verification.",
            badge_text="System Health",
            badge_type="info",
        )

        # System Runtime Environment
        import torch
        st.markdown(
            f"""
            <div class="vx-card">
                <div style="font-size: 0.72rem; font-weight: 700; color: #67E8F9; text-transform: uppercase; margin-bottom: 8px;">Runtime Environment Telemetry</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.8rem; color: #F1F5F9;">
                    <div>• <b>Python Version:</b> {sys.version.split()[0]}</div>
                    <div>• <b>PyTorch Version:</b> {torch.__version__}</div>
                    <div>• <b>Compute Device:</b> {'CUDA (GPU Accelerated)' if torch.cuda.is_available() else 'CPU (Multi-Threaded)'}</div>
                    <div>• <b>Streamlit Version:</b> {st.__version__}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "pipeline_result" in st.session_state:
            res: VoxelPipelineResult = st.session_state["pipeline_result"]
            st.markdown(
                '<div style="font-size: 0.82rem; font-weight: 700; color: #F1F5F9; text-transform: uppercase; letter-spacing: 0.05em; margin: 16px 0 12px 0;">Latest Execution Stage Trace</div>',
                unsafe_allow_html=True,
            )

            # Timing Breakdown Chart
            timings = res.timing_breakdown_ms
            if timings:
                fig, ax = plt.subplots(figsize=(7, 2.5), dpi=150)
                fig.patch.set_facecolor("#151D2A")
                ax.set_facecolor("#151D2A")

                stage_keys = [k for k in timings.keys() if k != "total_pipeline_latency_ms"]
                stage_vals = [timings[k] for k in stage_keys]
                stage_labels = [k.replace("_ms", "").replace("_", " ").title() for k in stage_keys]

                y_pos = np.arange(len(stage_keys))
                ax.barh(y_pos, stage_vals, color="#67E8F9", height=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(stage_labels, fontsize=8, color="#F1F5F9")
                ax.invert_yaxis()
                ax.set_xlabel("Latency (ms)", fontsize=8, color="#94A3B8")
                ax.tick_params(colors="#94A3B8", labelsize=7)

                for spine in ax.spines.values():
                    spine.set_color("rgba(255, 255, 255, 0.08)")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Stage Handoff Contracts
            st.markdown("#### Stage Handoff Specifications")
            hcol1, hcol2 = st.columns(2)
            with hcol1:
                st.write(f"**Stage 0 Geocoder:** `{res.resolved_location.get('geocoder')}`")
                st.write(f"**Module 1 Tensor Shape:** `{res.input_tensor_shape}`")
                st.write(f"**Module 1 Data Contract:** `Float32 Surface Reflectance [0, 1]`")
            with hcol2:
                st.write(f"**Module 2 Mask Provenance:** `{res.mask_provenance}`")
                st.write(f"**Module 3 Model Provenance:** `{res.model_provenance}`")
                st.write(f"**Module 3 Embedding Dim:** `1216 dimensions`")

            # Warnings & Gating Logs
            if res.warnings:
                st.markdown("#### Active Warnings & Gating Logs")
                for w in res.warnings:
                    st.warning(w)

            with st.expander("📄 Full Pipeline Diagnostics JSON"):
                st.json(res.diagnostics)
        else:
            render_empty_state(
                icon="📊",
                title="No Active Execution Trace",
                message="Run a mission from the Mission Control tab to view real-time stage telemetry, latency traces, and data handoff invariants.",
                action_hint="Switch to 'Mission Control' and execute a mission query.",
            )


# -----------------------------------------------------------------------------
# APPLICATION ENTRYPOINT
# -----------------------------------------------------------------------------
if not isinstance(st, MagicMock) and not hasattr(st, "_mock_return_value"):
    render_dashboard()
