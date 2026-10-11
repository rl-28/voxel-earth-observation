"""
VOXEL: Task-Adaptive Earth Observation & Spectral Intelligence Platform
Streamlit Dashboard (app.py)

Minimal, Professional Light-Theme Research Dashboard:
  Section 1: 🛰️ Satellite Analysis (Primary Screen: Natural Language Query, Real STAC Ingestion, RGB Composite, CNN-ViT Classification)
  Section 2: 📊 Research & Evaluation (Dataset Provenance, Zero-Leakage Spatial Split, Checkpoint Registry, Confusion Matrices)
  Section 3: ⚡ Spectral Optimization (Module 2: Physics-Guided Band Selection, Calibrated BWOA Masks, Live Search)
  Section 4: 📡 Satellite Ingestion (Module 1: STAC Tiling, Radiometric Normalization, BOA Calibration)
  Section 5: 🧠 Deep Architecture (Module 3: Hybrid FusionNet CNN-ViT, Latent Vector Space, FAISS Retrieval)
"""

import os
import sys
import json
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
    render_academic_header,
    render_operational_header,
    render_spectral_matrix,
    render_hud_prediction,
    style_light_figure,
    plot_light_training_curve,
    get_verified_runtime_status,
    get_system_telemetry_status,
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
            st.write(f"Processed tile `{res.tile_id}` successfully.")
            with st.expander(f"Diagnostics: {res.tile_id}"):
                st.json(res.diagnostics)
                st.write(f"**Contract Type:** `{res.diagnostics.get('contract_type')}`")
                st.write(f"**Shape:** `{res.output_shape}`")

                if len(res.output_shape) >= 3 and res.output_shape[2] == 12:
                    st.warning("Module 2 Handoff: Output has 12 channels (Sentinel-2 L2A intermediate).")

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
        page_title="VOXEL | Earth Observation Research Platform",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject Light Theme Academic Design System
    inject_custom_styles()

    # Initialize Session State
    if "latest_pipeline_result" not in st.session_state:
        st.session_state["latest_pipeline_result"] = None
    if "active_query" not in st.session_state:
        st.session_state["active_query"] = "Track waterbody levels in Loktak Lake, Manipur"

    # Inspect genuine filesystem & module verification status (lightweight, zero expensive IO)
    telemetry = get_system_telemetry_status(PROJECT_ROOT)
    m1 = telemetry["module1"]
    m2 = telemetry["module2"]
    m3 = telemetry["module3"]
    ver = telemetry["verification"]

    # -------------------------------------------------------------------------
    # SIDEBAR: NAVIGATION & VERIFIED RUNTIME TELEMETRY
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 14px;">
                <div style="font-size: 1.35rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                    🛰️ VOXEL
                </div>
                <div style="font-size: 0.80rem; font-weight: 600; color: #2563EB; text-transform: uppercase; letter-spacing: 0.05em;">
                    Earth Observation Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clear, accessible navigation radio
        nav_selection = st.radio(
            "RESEARCH MODULES",
            [
                "🛰️ Satellite Analysis",
                "📊 Research & Evaluation",
                "⚡ Spectral Optimization (Module 2)",
                "📡 Satellite Ingestion (Module 1)",
                "🧠 Deep Architecture (Module 3)",
            ],
            key="main_nav_radio",
        )

        st.markdown("<hr style='margin: 14px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        # Genuine System Telemetry & Module Health Checks
        st.markdown(
            f"""
            <div class="voxel-card" style="padding: 12px 14px; margin-bottom: 12px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                    SYSTEM TELEMETRY &amp; HEALTH
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.82rem; color: #334155;">{m1['label']}</span>
                    <span class="{m1['class']}">{m1['status']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.82rem; color: #334155;">{m2['label']}</span>
                    <span class="{m2['class']}">{m2['status']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.82rem; color: #334155;">{m3['label']}</span>
                    <span class="{m3['class']}">{m3['status']}</span>
                </div>
                <div style="border-top: 1px dashed #E2E8F0; margin-top: 8px; padding-top: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.82rem; color: #334155;">{ver['label']}</span>
                        <span class="{ver['class']}">{ver['status']}</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 3px;">
                        {f"⏱️ {ver['timestamp']} · {ver['source']}" if ver['timestamp'] else f"⚠️ {ver['source']}"}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Scientific Scope Disclaimers
        st.markdown(
            """
            <div class="voxel-card" style="padding: 12px 14px; background: #F8FAFC;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                    SCIENTIFIC INTEGRITY GATES
                </div>
                <div style="font-size: 0.78rem; color: #475569; line-height: 1.45;">
                    <p style="margin: 0 0 3px 0;">● <b>Surface Reflectance:</b> Sentinel-2 BOA [0.0, 1.0]</p>
                    <p style="margin: 0 0 3px 0;">● <b>Spatial Footprint:</b> 256×256 px (2.56 km²)</p>
                    <p style="margin: 0 0 3px 0;">● <b>Atmospheric Cirrus:</b> Band 10 Refused</p>
                    <p style="margin: 0;">● <b>Model Compatibility:</b> Mask Decoupled</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("VOXEL Platform · Sentinel-2 MSI Multi-Spectral Intelligence")

    # =========================================================================
    # SECTION 1: PRIMARY SCREEN - SATELLITE ANALYSIS
    # =========================================================================
    if nav_selection == "🛰️ Satellite Analysis":
        render_academic_header(
            "Task-Adaptive Satellite Analysis",
            "Natural Language Query Compilation ➔ STAC Ingestion ➔ Physics-Guided Band Optimization ➔ CNN-ViT FusionNet",
            "VOXEL ORCHESTRATOR"
        )

        # Primary Query Input Card
        st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
        
        # Suggested location chips for one-click testing
        st.markdown('<div style="font-size: 0.82rem; font-weight: 600; color: #64748B; margin-bottom: 6px;">VERIFIED INDIAN RESEARCH TARGETS:</div>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        if col_c1.button("📍 Loktak Lake (Water)", use_container_width=True):
            st.session_state["active_query"] = "Track waterbody levels in Loktak Lake, Manipur"
        if col_c2.button("📍 Jim Corbett (Forest)", use_container_width=True):
            st.session_state["active_query"] = "Assess forest canopy in Jim Corbett National Park, Uttarakhand"
        if col_c3.button("📍 Chilika Lake (Water)", use_container_width=True):
            st.session_state["active_query"] = "Identify water bodies around Chilika Lake, Odisha, India"
        if col_c4.button("📍 Western Ghats (Forest)", use_container_width=True):
            st.session_state["active_query"] = "Analyze forest canopy and tree cover around Western Ghats, Karnataka, India"

        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        query_input = st.text_input(
            "Earth Observation Target Query:",
            value=st.session_state.get("active_query", "Track waterbody levels in Loktak Lake, Manipur"),
            placeholder="e.g., Track waterbody levels in Loktak Lake, Manipur",
            help="Enter a geographic region and natural language task objective."
        )

        # Task Selection & Primary Execution
        q_col1, q_col2 = st.columns([2, 1])
        with q_col1:
            task_mode = st.radio(
                "Task Profile Mode:",
                ["auto", "waterbody", "forest_cover"],
                format_func=lambda x: "Auto-Detect from Query" if x == "auto" else ("Waterbody Classification (6 Bands)" if x == "waterbody" else "Forest Cover Classification (5 Bands)"),
                horizontal=True,
                help="Automatically routes to the appropriate physics domain and verified checkpoint."
            )
        with q_col2:
            st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
            launch_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

        # Advanced Controls inside a clean collapsible expander
        with st.expander("⚙️ Advanced Analysis & Optimization Settings", expanded=False):
            ac1, ac2 = st.columns(2)
            with ac1:
                exec_mode = st.radio(
                    "Acquisition Pipeline:",
                    ["real", "offline"],
                    format_func=lambda x: "🛰️ Real Satellite Mode (STAC Acquisition / Cached Tiles)" if x == "real" else "🧪 Offline Smoke Test (Synthetic Fixture)",
                    horizontal=True,
                )
                run_bwoa = st.checkbox(
                    "Execute Live BWOA Band Search",
                    value=False,
                    help="Runs fresh Binary Whale Optimization on the scene tensor to discover an exploratory spectral subset. Validated model applies verified calibration mask."
                )
            with ac2:
                run_shap = st.checkbox(
                    "Compute Permutation SHAP Attributions",
                    value=False,
                    help="Computes live spatial-pixel permutation feature attributions across active channels (~2s extra compute)."
                )
                run_faiss = st.checkbox(
                    "Query FAISS Latent Vector Index",
                    value=True,
                    help="Queries nearest-neighbor latent representations (1216-dim) in the EuroSAT benchmark corpus."
                )

        st.markdown('</div>', unsafe_allow_html=True)

        # Pipeline Execution Handler
        if launch_btn:
            with st.spinner("Executing VOXEL Earth Observation Pipeline (Geocoding ➔ STAC ➔ Optimization ➔ FusionNet)..."):
                try:
                    pipeline = VoxelPipeline(default_mode=exec_mode, verbose=False)
                    res: VoxelPipelineResult = pipeline.run(
                        query=query_input,
                        mode=exec_mode,
                        task_override=None if task_mode == "auto" else task_mode,
                        explain_shap=run_shap,
                        run_faiss=run_faiss,
                        optimize_bands=run_bwoa,
                    )
                    st.session_state["latest_pipeline_result"] = res
                    st.session_state["active_query"] = query_input
                except Exception as e:
                    st.error(f"Pipeline Execution Failed: {str(e)}")
                    with st.expander("Diagnostic Traceback"):
                        st.code(traceback.format_exc())

        # Render Active or Persisted Results
        res: Optional[VoxelPipelineResult] = st.session_state.get("latest_pipeline_result")

        if res is not None:
            # 1. Prediction & Provenance Banner
            if res.prediction:
                render_hud_prediction(
                    res.prediction,
                    res.task_type,
                    res.band_reduction_pct,
                    compatibility_note=res.mask_compatibility_note,
                )
            else:
                st.warning("⚠️ Prediction withheld: Spectral mask channels do not match calibrated input weights of the real trained model.")

            # Task Scope Qualifications Notice (if applicable)
            if "task_qualifications" in res.diagnostics and res.diagnostics["task_qualifications"]:
                for q_note in res.diagnostics["task_qualifications"]:
                    st.info(f"ℹ️ **Research Qualification:** {q_note}")

            # 2. Key Telemetry Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Active Bands", f"{len(res.selected_bands)} / 13", f"-{res.band_reduction_pct:.1f}% Volume")
            m2.metric("Total Latency", f"{res.timing_breakdown_ms.get('total_pipeline_latency_ms', 0):.0f} ms", "Sub-second Pipeline")
            prov_label = "Live Sentinel-2" if not res.is_synthetic else "Synthetic Fixture"
            m3.metric("Data Source", prov_label, res.resolved_location.get("name", "India"))
            m4.metric("Surface Radiometry", "BOA [0.0, 1.0]", "L2A Surface Reflectance")

            # 3. Prominent Visual Evidence (RGB Satellite Composite + Spectral Matrix)
            st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True)
            vcol1, vcol2 = st.columns([1, 1.25])

            with vcol1:
                st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
                st.markdown('<div class="voxel-card-header">SATELLITE RGB COMPOSITE</div>', unsafe_allow_html=True)
                if res.diagnostics.get("rgb_preview_path") and os.path.exists(res.diagnostics["rgb_preview_path"]):
                    st.image(
                        res.diagnostics["rgb_preview_path"],
                        caption="Sentinel-2 L2A True-Color Composite (B04-Red, B03-Green, B02-Blue)",
                        use_container_width=True
                    )
                else:
                    st.info("True-color composite preview unavailable for this tile.")
                
                # Spatial & Acquisition Metadata
                st.markdown(
                    f"""
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 8px; line-height: 1.5;">
                        <div><b>Location:</b> {res.resolved_location.get('name', 'N/A')}</div>
                        <div><b>Coordinates:</b> <code>{res.resolved_location.get('bbox', 'N/A')}</code></div>
                        <div><b>Input Tensor:</b> <code>{res.input_tensor_shape}</code> (10m/px)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with vcol2:
                st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
                st.markdown('<div class="voxel-card-header">SELECTED SPECTRAL CHANNELS (SENTINEL-2 MSI)</div>', unsafe_allow_html=True)
                render_spectral_matrix(res.selected_bands)

                if res.candidate_bwoa_bands:
                    st.caption(f"⚡ **Live BWOA Candidate:** `{', '.join(res.candidate_bwoa_bands)}` | **Model Verified Bands:** `{', '.join(res.selected_bands)}` (Decoupled for Mathematical Integrity)")
                else:
                    st.caption("Active bands highlighted in blue. Atmospheric Cirrus (Band 10) is refused to preserve surface reflectance accuracy.")
                
                # Class Probability Distribution
                if res.prediction and "all_probabilities" in res.prediction:
                    st.markdown('<div style="margin-top: 12px; font-size: 0.82rem; font-weight: 600; color: #475569;">CLASS PROBABILITY DENSITY:</div>', unsafe_allow_html=True)
                    for cls_name, p_val in res.prediction["all_probabilities"].items():
                        st.progress(p_val / 100.0, text=f"{cls_name}: {p_val:.1f}%")

                st.markdown('</div>', unsafe_allow_html=True)

            # 4. Optional Advanced Diagnostics, Explainability & Verification
            with st.expander("🔍 Detailed Pipeline Telemetry & Timing Breakdown", expanded=False):
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Stage 0 Compiler", f"{res.timing_breakdown_ms.get('stage_0_query_compilation_ms', 0):.1f} ms")
                t2.metric("Module 1 Ingestion", f"{res.timing_breakdown_ms.get('module_1_ingestion_ms', 0):.1f} ms")
                t3.metric("Module 2 BWOA", f"{res.timing_breakdown_ms.get('module_2_optimization_ms', 0):.1f} ms")
                t4.metric("Module 3 FusionNet", f"{res.timing_breakdown_ms.get('module_3_inference_ms', 0):.1f} ms")

                st.write("**Resolved Geographic Contract:**")
                st.json(res.resolved_location)

            if res.explainability and "top_contributing_bands" in res.explainability:
                with st.expander("📈 Permutation SHAP Spectral Feature Attribution", expanded=True):
                    top_bands = res.explainability["top_contributing_bands"]
                    band_names = [b["name"] for b in top_bands]
                    band_pcts = [b["percentage"] for b in top_bands]

                    if any(p > 0.0 for p in band_pcts):
                        fig, ax = plt.subplots(figsize=(7, 2.8), dpi=150)
                        y_pos = np.arange(len(band_names))
                        ax.barh(y_pos, band_pcts, color="#2563EB", align="center", edgecolor="#1D4ED8", height=0.55)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(band_names)
                        ax.invert_yaxis()
                        ax.set_xlabel("Relative Sensitivity Attribution (%)")
                        ax.set_title("Single-Tile Spatial Permutation SHAP Attribution")
                        style_light_figure(fig, ax)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Attribution: Prediction confidence is invariant to spatial pixel permutation across channels on this tile.")

            if res.similar_terrain:
                with st.expander("🌐 Semantic Terrain Nearest Neighbors (FAISS Vector Index)", expanded=False):
                    st.caption("Reference Corpus: EuroSAT multi-class benchmark scenes indexed in 1216-dim latent vector space to evaluate representation coherence.")
                    st.dataframe(pd.DataFrame(res.similar_terrain), use_container_width=True)

            with st.expander("📄 View Machine-Readable Result Contract (JSON)", expanded=False):
                st.json(res.to_dict())

        else:
            # Clean Research Briefing Card for Ready State
            st.markdown(
                """
                <div class="voxel-card" style="margin-top: 14px;">
                    <div class="voxel-card-header">SYSTEM READY · SCIENTIFIC BRIEFING</div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 10px;">
                        <div>
                            <div style="font-weight: 600; color: #0F172A; font-size: 0.90rem; margin-bottom: 4px;">Verified Test Targets</div>
                            <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; margin: 0;">
                                <li><code>Loktak Lake, Manipur</code> (Waterbody Classification)</li>
                                <li><code>Jim Corbett, Uttarakhand</code> (Forest Cover Classification)</li>
                                <li><code>Chilika Lake, Odisha</code> (Coastal Water Lagoon)</li>
                                <li><code>Western Ghats, Karnataka</code> (Tropical Montane Forest)</li>
                            </ul>
                        </div>
                        <div>
                            <div style="font-weight: 600; color: #0F172A; font-size: 0.90rem; margin-bottom: 4px;">Calibrated Model Checkpoints</div>
                            <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; margin: 0;">
                                <li><b>Waterbody:</b> 6 Bands (B01, B03, B04, B06, B8A, B11) — 73.7% Test OA</li>
                                <li><b>Forest Cover:</b> 5 Bands (B02, B04, B06, B11, B12) — 94.7% Test OA</li>
                                <li><b>Architecture:</b> Multimodal Hybrid FusionNet (CNN + ViT)</li>
                            </ul>
                        </div>
                        <div>
                            <div style="font-weight: 600; color: #0F172A; font-size: 0.90rem; margin-bottom: 4px;">Standard Processing Pipeline</div>
                            <ol style="color: #475569; font-size: 0.85rem; padding-left: 18px; margin: 0;">
                                <li>Stage 0 Natural Language Query Geocoding</li>
                                <li>Module 1 Sentinel-2 L2A STAC Acquisition</li>
                                <li>Module 2 Physics Spectral Band Slicing</li>
                                <li>Module 3 1216-dim FusionNet Inference</li>
                            </ol>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # SECTION 2: RESEARCH & EVALUATION SECTION
    # =========================================================================
    elif nav_selection == "📊 Research & Evaluation":
        render_operational_header(
            "Scientific Provenance & Model Evaluation Hub",
            "Zero-Leakage Spatial Split · Calibrated BOA Reflectance · Real Checkpoint Registry · Held-Out Test Metrics",
            "RESEARCH & EVALUATION"
        )

        st.subheader("1. Genuine Sentinel-2 Indian Dataset vs Synthetic Fixtures")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.markdown(
                """
                <div class="voxel-card">
                    <div class="voxel-card-header" style="color: #059669;">🛰️ SENTINEL-2 L2A GROUND TRUTH</div>
                    <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; margin: 0;">
                        <li><b>Source:</b> AWS Earth Search STAC (<code>sentinel-2-l2a</code>)</li>
                        <li><b>Geographies:</b> 6 Indian regions (Chilika, Western Ghats, Thar, Sardar Sarovar, Hirakud, Kaziranga)</li>
                        <li><b>Tiles:</b> 120 calibrated 256×256 surface reflectance tiles</li>
                        <li><b>Radiometry:</b> BOA Surface Reflectance <code>[0.0, 1.0]</code></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pcol2:
            st.markdown(
                """
                <div class="voxel-card">
                    <div class="voxel-card-header" style="color: #059669;">🏷️ ESA WORLDCOVER REFERENCE</div>
                    <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; margin: 0;">
                        <li><b>Reference:</b> ESA WorldCover 2021 v200 (10m global raster)</li>
                        <li><b>Alignment:</b> Raster reprojection to exact tile CRS & bounds</li>
                        <li><b>Thresholding:</b> Positive ≥ 10%, Negative ≤ 0.5%</li>
                        <li><b>Quality:</b> Ambiguous boundary tiles filtered</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pcol3:
            st.markdown(
                """
                <div class="voxel-card">
                    <div class="voxel-card-header" style="color: #059669;">🗺️ ZERO-LEAKAGE SPATIAL SPLIT</div>
                    <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; margin: 0;">
                        <li><b>Train (68 tiles):</b> Kaziranga, Western Ghats, Thar, Sardar Sarovar</li>
                        <li><b>Val (20 tiles):</b> Chilika Coastal Lagoon</li>
                        <li><b>Held-Out Test (19 tiles):</b> Hirakud Inland Reservoir</li>
                        <li><b>Integrity:</b> Zero tile overlap between train and test regions</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("ℹ️ **Scientific Scope & Generalization Disclaimer:** Evaluated on a 19-tile held-out set from Hirakud inland reservoir. While ensuring strict geographic isolation from training regions, it represents an experimental benchmark; nationwide generalization across all agro-ecological zones requires ongoing multi-region expansion.")

        # Ground Truth Visual Alignment Inspection Sheet
        st.markdown("<hr style='margin: 18px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
        st.subheader("2. Ground Truth Visual Alignment (Sentinel-2 RGB vs ESA WorldCover)")
        label_check_img = PROJECT_ROOT / "results" / "label_check.png"
        if label_check_img.exists():
            st.image(str(label_check_img), caption="Sentinel-2 L2A RGB vs ESA WorldCover 2021 v200 Ground Truth Across India", use_container_width=True)

        # Checkpoint Registry Table
        st.markdown("<hr style='margin: 18px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
        st.subheader("3. Calibrated Checkpoint Registry & Performance Metrics")

        ckpt_summary = [
            {
                "Task": "Waterbody (REAL DATA)",
                "Checkpoint Path": "checkpoints/real_waterbody/best_fusion_net.pth",
                "Channels": "6 bands (B01, B03, B04, B06, B8A, B11)",
                "Val OA": "90.00%",
                "Val F1": "88.10%",
                "Test OA": "73.68%",
                "Test Macro F1": "73.39%",
                "Water Recall": "100.00%",
                "Status": "Verified Real Satellite Model",
            },
            {
                "Task": "Waterbody (Synthetic Mock)",
                "Checkpoint Path": "checkpoints/waterbody/best_fusion_net.pth",
                "Channels": "6 bands (B01, B03, B04, B06, B8A, B11)",
                "Val OA": "50.00%",
                "Val F1": "33.33%",
                "Test OA": "50.00%",
                "Test Macro F1": "33.33%",
                "Water Recall": "100.00%",
                "Status": "Engineering Test Fixture",
            },
            {
                "Task": "Forest Cover (REAL DATA)",
                "Checkpoint Path": "checkpoints/real_forest_cover/best_fusion_net.pth",
                "Channels": "5 bands (B02, B04, B06, B11, B12)",
                "Val OA": "66.67%",
                "Val F1": "47.92%",
                "Test OA": "94.74%",
                "Test Macro F1": "93.60%",
                "Forest Recall": "100.00%",
                "Status": "Verified Real Satellite Model (+26.3% OA vs Baseline)",
            },
            {
                "Task": "Forest Cover (Synthetic Mock)",
                "Checkpoint Path": "checkpoints/forest_cover/best_fusion_net.pth",
                "Channels": "5 bands (B02, B04, B06, B11, B12)",
                "Val OA": "100.00%",
                "Val F1": "100.00%",
                "Test OA": "100.00%",
                "Test Macro F1": "100.00%",
                "Forest Recall": "N/A",
                "Status": "Engineering Test Fixture",
            },
        ]
        st.dataframe(pd.DataFrame(ckpt_summary), use_container_width=True)

        # Training Dynamics Curves & Confusion Matrices
        st.markdown("<hr style='margin: 18px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
        st.subheader("4. Real-Data Training Dynamics & Held-Out Test Confusion Matrices")
        tcol1, tcol2 = st.columns(2)

        with tcol1:
            st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
            st.markdown('<div class="voxel-card-header">REAL WATERBODY EVALUATION (HIRAKUD TEST SET)</div>', unsafe_allow_html=True)
            real_wb_cm = PROJECT_ROOT / "results" / "real_waterbody_eval" / "confusion_matrix.png"
            if real_wb_cm.exists():
                st.image(str(real_wb_cm), caption="Held-Out Hirakud Confusion Matrix (OA: 73.68%, Macro-F1: 73.39%, Water Recall: 100%)", use_container_width=True)
            
            real_wb_log = PROJECT_ROOT / "results" / "real_waterbody_training_log.csv"
            if real_wb_log.exists():
                fig_wb = plot_light_training_curve(real_wb_log, "Real Waterbody Training Progression")
                if fig_wb:
                    st.pyplot(fig_wb)
                    plt.close(fig_wb)
            st.markdown('</div>', unsafe_allow_html=True)

        with tcol2:
            st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
            st.markdown('<div class="voxel-card-header">REAL FOREST COVER EVALUATION (HIRAKUD TEST SET)</div>', unsafe_allow_html=True)
            real_fc_cm = PROJECT_ROOT / "results" / "real_forest_cover_eval" / "confusion_matrix.png"
            if real_fc_cm.exists():
                st.image(str(real_fc_cm), caption="Held-Out Hirakud Confusion Matrix (OA: 94.74%, Macro-F1: 93.60%, Forest Recall: 100%)", use_container_width=True)
                
            real_fc_log = PROJECT_ROOT / "results" / "real_forest_cover_training_log.csv"
            if real_fc_log.exists():
                fig_fc = plot_light_training_curve(real_fc_log, "Real Forest Cover Training Progression")
                if fig_fc:
                    st.pyplot(fig_fc)
                    plt.close(fig_fc)
            st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3: MODULE 2 SPECTRAL BAND OPTIMIZATION
    # =========================================================================
    elif nav_selection == "⚡ Spectral Optimization (Module 2)":
        render_operational_header(
            "Physics-Guided Spectral Band Optimization (BWOA)",
            "Task-Adaptive Down-Selection · Binary Whale Optimization · Physical Domain Balance & Cirrus Refusal",
            "MODULE 2 // SPECTRAL OPTIMIZATION"
        )

        mask_files = {
            "Waterbody Mask (6 Bands)": "checkpoints/masks/waterbody_mask.json",
            "Forest Cover Mask (5 Bands)": "checkpoints/masks/forest_cover_mask.json",
        }

        st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
        selected_mask_name = st.selectbox("Inspect Calibrated Task Mask:", list(mask_files.keys()))
        mask_rel_path = mask_files[selected_mask_name]
        mask_full_path = PROJECT_ROOT / mask_rel_path

        if mask_full_path.exists():
            with open(mask_full_path, "r", encoding="utf-8") as f:
                mask_data = json.load(f)

            mcol1, mcol2 = st.columns([1, 1])
            with mcol1:
                st.markdown('<div class="voxel-card-header">BAND MASK SPECIFICATIONS</div>', unsafe_allow_html=True)
                st.write(f"**Optimization Method:** `{mask_data.get('method', 'BWOA')}`")
                st.write(f"**Active Channels:** `{len(mask_data.get('selected_indices', []))} / {mask_data.get('total_bands', 13)}`")
                st.write(f"**Data Volume Reduction:** `{((13 - len(mask_data.get('selected_indices', []))) / 13) * 100:.1f}%`")
                st.write(f"**Selected Bands:** `{mask_data.get('selected_bands')}`")
                st.write(f"**Band Indices:** `{mask_data.get('selected_indices')}`")

            with mcol2:
                st.markdown('<div class="voxel-card-header">CHANNEL SELECTION HEATMAP</div>', unsafe_allow_html=True)
                active_indices = mask_data.get("selected_indices", [])
                binary_vector = [1 if i in active_indices else 0 for i in range(13)]

                fig, ax = plt.subplots(figsize=(8, 1.8), dpi=150)
                colors = ["#2563EB" if v == 1 else "#E2E8F0" for v in binary_vector]
                ax.bar(range(13), [1]*13, color=colors, edgecolor="#CBD5E1", width=0.75)
                ax.set_xticks(range(13))
                ax.set_xticklabels(SENTINEL2_BAND_KEYS, fontsize=8.5)
                ax.set_yticks([])
                ax.set_title(f"Channel Allocation ({len(active_indices)}/13 Active)")

                for idx, val in enumerate(binary_vector):
                    ax.text(idx, 0.5, "ON" if val == 1 else "OFF", ha="center", va="center",
                            color="#FFFFFF" if val == 1 else "#64748B", fontweight="bold", fontsize=7.5)

                style_light_figure(fig, ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True)
            st.markdown("#### Electromagnetic Regime Allocation")
            render_spectral_matrix(mask_data.get("selected_bands", []))

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🧪 Live BWOA Metaheuristic Search Sandbox", expanded=False):
            st.markdown("Execute a fresh Binary Whale Optimization run on simulated multi-spectral tensor:")
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                live_target_bands = st.slider("Target Band Count:", min_value=3, max_value=8, value=5)
                live_whales = st.slider("Whale Population (Agents):", min_value=5, max_value=25, value=10)
            with bcol2:
                live_iters = st.slider("Max Iterations:", min_value=3, max_value=20, value=5)
                live_seed = st.number_input("Random Seed:", value=42)

            if st.button("Run Live BWOA Search"):
                with st.spinner("Executing Binary Whale Optimization Algorithm..."):
                    from module2_optimization.orchestrator import run_module2_optimization
                    sample_tensor = np.random.RandomState(live_seed).uniform(0.05, 0.6, (256, 256, 13))
                    mask_arr, sel_indices, opt_metrics = run_module2_optimization(
                        input_tensor=sample_tensor,
                        target_count=live_target_bands,
                        num_whales=live_whales,
                        max_iter=live_iters,
                        seed=live_seed,
                    )
                    sel_band_names = [SENTINEL2_BAND_KEYS[i] for i in sel_indices if i < len(SENTINEL2_BAND_KEYS)]
                    st.success(f"BWOA Search Completed in {opt_metrics.get('optimization_time_ms', 0):.1f} ms")
                    st.write(f"**Discovered Candidate Bands ({len(sel_indices)}):** `{', '.join(sel_band_names)}`")
                    st.write(f"**Data Volume Reduction:** `{((13 - len(sel_indices)) / 13) * 100:.1f}%`")
                    st.json(opt_metrics)

    # =========================================================================
    # SECTION 4: MODULE 1 INGESTION & SIGNAL PROCESSING
    # =========================================================================
    elif nav_selection == "📡 Satellite Ingestion (Module 1)":
        render_operational_header(
            "Satellite Ingestion & Radiometric Processing",
            "Direct Ingestion · 256×256 Spatial Tiling · BOA Radiometric Calibration · 3D DWT Wavelets",
            "MODULE 1 // SATELLITE INGESTION"
        )

        st.markdown('<div class="voxel-card">', unsafe_allow_html=True)
        m1_mode = st.radio("Select Processing Mode:", ["Evaluation Region (Ext SR / Ext WB)", "Local Directory / Upload"], key="m1_radio")

        if m1_mode == "Local Directory / Upload":
            uploaded_file = st.file_uploader("Upload a ZIP file containing Sentinel-2 L2A band GeoTIFFs (.tif or .jp2)", type="zip", key="m1_zip_uploader")
            if uploaded_file and st.button("Run Module 1 Processing", key="m1_run_btn"):
                with st.spinner("Extracting uploaded archive and processing..."):
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
            region_id = st.selectbox(
                "Select Region:",
                list(EVALUATION_REGIONS.keys()),
                format_func=lambda x: EVALUATION_REGIONS[x].display_name,
                key="m1_region_select"
            )
            eval_input_dir = st.text_input("Local Source Directory Path for Evaluation Region (Optional):", "", key="m1_eval_dir")

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
    # SECTION 5: MODULE 3 DEEP ARCHITECTURE & LATENT RETRIEVAL
    # =========================================================================
    elif nav_selection == "🧠 Deep Architecture (Module 3)":
        render_operational_header(
            "Deep Learning Architecture & Latent Retrieval",
            "Hybrid CNN-ViT FusionNet · Permutation SHAP Channel Attributions · 1216-dim FAISS Latent Vector Space",
            "MODULE 3 // DEEP LEARNING"
        )

        st.subheader("1. FusionNet Architecture Specification")
        st.markdown(
            """
            <div class="voxel-card">
                <div class="voxel-card-header">FUSIONNET DUAL-BRANCH NEURAL TOPOLOGY</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-top: 10px;">
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px;">
                        <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">CNN Backbone (1024-dim)</div>
                        <div style="color: #475569; font-size: 0.84rem; margin-top: 4px;">XNANet deep convolutional feature extractor capturing high-resolution spatial edges and local terrain textures.</div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px;">
                        <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">ViT Transformer (192-dim)</div>
                        <div style="color: #475569; font-size: 0.84rem; margin-top: 4px;">Tiny-32 Vision Transformer with multi-head self-attention extracting global spatial-spectral context.</div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px;">
                        <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">Joint Embedding (1216-dim)</div>
                        <div style="color: #475569; font-size: 0.84rem; margin-top: 4px;">Linear projection concat head combining spatial and spectral vectors for classification and FAISS retrieval.</div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px;">
                        <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">Dynamic Spectral Slicing</div>
                        <div style="color: #475569; font-size: 0.84rem; margin-top: 4px;">Channels routed dynamically matching calibrated Module 2 masks without dynamic Conv re-instantiation.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin: 18px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
        st.subheader("2. Global Spectral Importance Artifact")
        shap_img_path = PROJECT_ROOT / "checkpoints" / "shap_band_importance.png"
        if shap_img_path.exists():
            st.image(str(shap_img_path), caption="Global Permutation SHAP Band Importance (EuroSAT Pretraining Benchmark)", use_container_width=True)
        else:
            st.info("SHAP plot artifact not yet generated in checkpoints directory.")

        st.markdown("<hr style='margin: 18px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
        st.subheader("3. Semantic Latent Vector Space (FAISS)")
        faiss_path = PROJECT_ROOT / "checkpoints" / "terrain_index.faiss"
        faiss_meta_path = PROJECT_ROOT / "checkpoints" / "terrain_index.faiss.meta.json"

        if faiss_path.exists() and faiss_meta_path.exists():
            with open(faiss_meta_path, "r", encoding="utf-8") as f:
                meta_entries = json.load(f)

            st.markdown(
                f"""
                <div class="voxel-card">
                    <span class="voxel-badge-success">FAISS INDEX ONLINE</span>
                    <h3 style="margin-top: 6px; color: #0F172A;">{len(meta_entries)} Reference Scenes Indexed</h3>
                    <p style="color: #475569; font-size: 0.88rem; margin: 0;">
                        1216-dimensional latent space indexing EuroSAT multi-class reference benchmark terrain scenes. Real Indian task inference (Waterbody / Forest Cover) is executed directly by the trained classification heads.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("View Indexed Scene Metadata"):
                st.dataframe(pd.DataFrame(meta_entries), use_container_width=True)
        else:
            st.info("FAISS vector index file not found in checkpoints directory.")


# Trigger UI rendering if running under real Streamlit
if not isinstance(st, MagicMock) and not hasattr(st, "_mock_return_value"):
    render_dashboard()
