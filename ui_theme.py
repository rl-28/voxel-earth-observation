"""
ui_theme.py - Premium High-Contrast Futuristic Earth-Observation Theme & Visual Components for VOXEL

Design Direction:
- Space navy/black palette (#030712, #070d1e, #0b152d)
- High-contrast typography with zero dark-on-dark conflicts:
  * Orbitron (futuristic display headers)
  * Space Grotesk (body text, high legibility)
  * Rajdhani (telemetry & data labels)
  * JetBrains Mono (metrics, numbers, coordinates)
- Accessible, clean input controls with explicit text/background contrast
- Compact, responsive sidebar navigation (240–260px desktop width, mobile collapsible)
- Restrained electric-cyan (#00f0ff) and orbital-blue (#38bdf8) accents without excessive glow
"""

import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st

# Electromagnetic regime definitions for Sentinel-2 MSI
SPECTRAL_REGIMES = [
    {
        "regime": "VIS",
        "name": "Visible Spectrum",
        "color": "#38bdf8",
        "bands": [
            {"id": "B01", "name": "Coastal Aerosol", "wl": "443 nm", "res": "60m"},
            {"id": "B02", "name": "Blue", "wl": "490 nm", "res": "10m"},
            {"id": "B03", "name": "Green", "wl": "560 nm", "res": "10m"},
            {"id": "B04", "name": "Red", "wl": "665 nm", "res": "10m"},
        ],
    },
    {
        "regime": "VRE",
        "name": "Vegetation Red Edge",
        "color": "#34d399",
        "bands": [
            {"id": "B05", "name": "Red Edge 1", "wl": "705 nm", "res": "20m"},
            {"id": "B06", "name": "Red Edge 2", "wl": "740 nm", "res": "20m"},
            {"id": "B07", "name": "Red Edge 3", "wl": "783 nm", "res": "20m"},
        ],
    },
    {
        "regime": "NIR",
        "name": "Near-Infrared",
        "color": "#a78bfa",
        "bands": [
            {"id": "B08", "name": "Broad NIR", "wl": "842 nm", "res": "10m"},
            {"id": "B8A", "name": "Narrow NIR", "wl": "865 nm", "res": "20m"},
        ],
    },
    {
        "regime": "SWIR",
        "name": "Shortwave Infrared",
        "color": "#fbbf24",
        "bands": [
            {"id": "B11", "name": "SWIR-1", "wl": "1610 nm", "res": "20m"},
            {"id": "B12", "name": "SWIR-2", "wl": "2190 nm", "res": "20m"},
        ],
    },
    {
        "regime": "ATM",
        "name": "Atmospheric Absorption",
        "color": "#f87171",
        "bands": [
            {"id": "B09", "name": "Water Vapour", "wl": "945 nm", "res": "60m"},
            {"id": "B10", "name": "Cirrus (Refused)", "wl": "1375 nm", "res": "60m", "refused": True},
        ],
    },
]


def get_hero_image_base64() -> Optional[str]:
    """Encodes local cinematic Earth orbital asset into base64 for zero-latency background rendering."""
    asset_path = Path(__file__).resolve().parent / "assets" / "hero_earth.jpg"
    if asset_path.exists():
        try:
            return base64.b64encode(asset_path.read_bytes()).decode("utf-8")
        except Exception:
            return None
    return None


def inject_custom_styles():
    """Injects high-contrast dark space CSS styles into Streamlit without breaking native widgets."""
    hero_b64 = get_hero_image_base64()
    hero_bg_rule = (
        f"url('data:image/jpeg;base64,{hero_b64}')"
        if hero_b64
        else "radial-gradient(ellipse at 50% 0%, #0e2a47 0%, #030712 70%)"
    )

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@600;700;800;900&family=Rajdhani:wght@500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {{
        --space-bg: #030712;
        --space-surface: rgba(8, 16, 36, 0.82);
        --space-surface-solid: #081024;
        --space-card-border: rgba(0, 240, 255, 0.22);
        --space-card-border-hover: rgba(0, 240, 255, 0.55);
        --electric-cyan: #00f0ff;
        --electric-blue: #38bdf8;
        --space-text-main: #f8fafc;
        --space-text-secondary: #cbd5e1;
        --space-text-muted: #94a3b8;
        --font-display: 'Orbitron', sans-serif;
        --font-heading: 'Space Grotesk', sans-serif;
        --font-tech: 'Rajdhani', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }}

    /* Global App Background */
    .stApp {{
        background-color: var(--space-bg) !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(14, 38, 77, 0.35) 0%, transparent 50%),
            radial-gradient(circle at 90% 90%, rgba(8, 28, 60, 0.3) 0%, transparent 55%) !important;
        background-attachment: fixed !important;
        color: var(--space-text-main) !important;
        font-family: var(--font-heading) !important;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1360px !important;
    }}

    /* Sidebar Layout & Responsive Width */
    [data-testid="stSidebar"] {{
        background: #040817 !important;
        border-right: 1px solid var(--space-card-border) !important;
        min-width: 250px !important;
        max-width: 260px !important;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding: 1.5rem 1rem !important;
    }}

    /* High-Contrast Widget Labels */
    [data-testid="stWidgetLabel"] label, 
    [data-testid="stWidgetLabel"] p, 
    label {{
        color: #f8fafc !important;
        font-family: var(--font-heading) !important;
        font-weight: 600 !important;
        font-size: 0.93rem !important;
        margin-bottom: 4px !important;
    }}

    /* High-Contrast Text Inputs */
    div[data-testid="stTextInput"] input {{
        background-color: #070e24 !important;
        border: 1px solid rgba(0, 240, 255, 0.25) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 0.95rem !important;
        font-family: var(--font-heading) !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border-color: var(--electric-cyan) !important;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.3) !important;
    }}

    /* High-Contrast Selectbox / Dropdowns */
    div[data-baseweb="select"] {{
        background-color: #070e24 !important;
        border: 1px solid rgba(0, 240, 255, 0.25) !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] * {{
        color: #ffffff !important;
        background-color: transparent !important;
        font-family: var(--font-heading) !important;
    }}
    div[data-baseweb="popover"], 
    div[data-baseweb="menu"], 
    ul[role="listbox"] {{
        background-color: #070e24 !important;
        border: 1px solid rgba(0, 240, 255, 0.35) !important;
        border-radius: 8px !important;
    }}
    li[role="option"] {{
        color: #f1f5f9 !important;
        background-color: #070e24 !important;
        font-size: 0.92rem !important;
        padding: 8px 12px !important;
    }}
    li[role="option"]:hover, 
    li[aria-selected="true"] {{
        background-color: #162a52 !important;
        color: var(--electric-cyan) !important;
    }}

    /* High-Contrast Radio Buttons */
    div[data-testid="stRadio"] label, 
    div[data-testid="stRadio"] p, 
    div[role="radiogroup"] label {{
        color: #e2e8f0 !important;
        font-family: var(--font-heading) !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stRadio"] label:hover {{
        color: var(--electric-cyan) !important;
    }}

    /* Checkboxes */
    div[data-testid="stCheckbox"] label, 
    div[data-testid="stCheckbox"] p {{
        color: #e2e8f0 !important;
        font-family: var(--font-heading) !important;
        font-size: 0.92rem !important;
    }}

    /* Metric Values & Labels */
    div[data-testid="stMetricValue"] {{
        color: var(--electric-cyan) !important;
        font-family: var(--font-mono) !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #94a3b8 !important;
        font-family: var(--font-tech) !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }}

    /* Primary Launch Button */
    .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, #1d4ed8 0%, #0284c7 60%, var(--electric-cyan) 100%) !important;
        color: #ffffff !important;
        font-family: var(--font-tech) !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.45) !important;
        transition: all 0.25s ease !important;
    }}
    .stButton button[kind="primary"]:hover {{
        box-shadow: 0 6px 24px rgba(0, 240, 255, 0.65) !important;
        transform: translateY(-1px) !important;
    }}

    /* Expanders & Accordions */
    [data-testid="stExpander"] {{
        background: rgba(7, 14, 32, 0.75) !important;
        border: 1px solid rgba(0, 240, 255, 0.18) !important;
        border-radius: 10px !important;
        margin-top: 10px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: #f1f5f9 !important;
        font-family: var(--font-heading) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }}

    /* General Markdown and Paragraphs */
    .stMarkdown p, .stMarkdown span {{
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.55;
    }}

    /* Voxel Cards */
    .voxel-card {{
        background: var(--space-surface);
        border: 1px solid var(--space-card-border);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    .voxel-card-header {{
        font-family: var(--font-tech);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--electric-cyan);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Section Subheaders */
    .voxel-section-header {{
        margin-top: 4px;
        margin-bottom: 18px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(0, 240, 255, 0.18);
    }}
    .voxel-section-header h2 {{
        font-family: var(--font-display);
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: 0.04em;
    }}
    .voxel-section-header p {{
        font-size: 0.92rem;
        color: var(--space-text-muted);
        margin: 3px 0 0 0;
    }}

    /* Badges */
    .voxel-badge-success {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
        font-family: var(--font-tech);
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }}
    .voxel-badge-warning {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fcd34d;
        font-family: var(--font-tech);
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }}
    .voxel-badge-danger {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
        font-family: var(--font-tech);
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }}

    /* Pulse Dot */
    .voxel-pulse-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--electric-cyan);
        box-shadow: 0 0 8px var(--electric-cyan);
    }}

    /* Compact Hero Banner for Landing */
    .voxel-hero-compact {{
        position: relative;
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 22px;
        border: 1px solid rgba(0, 240, 255, 0.22);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        background: 
            linear-gradient(180deg, rgba(3, 7, 18, 0.3) 0%, rgba(3, 7, 18, 0.85) 75%, #030712 100%),
            {hero_bg_rule};
        background-size: cover;
        background-position: center 30%;
        padding: 24px 28px 20px 28px;
    }}
    .voxel-hero-title {{
        font-family: var(--font-display);
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        background: linear-gradient(135deg, #ffffff 40%, var(--electric-cyan) 85%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 4px 0;
    }}
    .voxel-hero-subtitle {{
        font-family: var(--font-tech);
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 6px;
    }}
    .voxel-hero-tagline {{
        font-family: var(--font-heading);
        font-size: 0.95rem;
        color: #e2e8f0;
        max-width: 780px;
        line-height: 1.45;
        margin-bottom: 16px;
    }}

    /* Telemetry Ribbon Strip */
    .voxel-telemetry-ribbon {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
    }}
    .voxel-telemetry-pill {{
        background: rgba(7, 14, 30, 0.85);
        border: 1px solid rgba(0, 240, 255, 0.18);
        border-radius: 8px;
        padding: 8px 12px;
    }}
    .voxel-telemetry-label {{
        font-family: var(--font-tech);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--space-text-muted);
        margin-bottom: 2px;
    }}
    .voxel-telemetry-value {{
        font-family: var(--font-mono);
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
    }}

    /* Operational Minimal Header */
    .voxel-operational-header {{
        background: rgba(7, 14, 32, 0.85);
        border: 1px solid rgba(0, 240, 255, 0.18);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .voxel-operational-title {{
        font-family: var(--font-display);
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.04em;
    }}
    .voxel-operational-tag {{
        font-family: var(--font-tech);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: var(--electric-cyan);
        text-transform: uppercase;
    }}

    /* Spectral Matrix Grid */
    .spectral-matrix-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(115px, 1fr));
        gap: 8px;
        margin-top: 10px;
    }}
    .spectral-band-card {{
        background: rgba(7, 14, 30, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }}
    .spectral-band-card.active {{
        border-color: var(--electric-cyan);
        background: rgba(0, 240, 255, 0.08);
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.18);
    }}
    .spectral-band-card.refused {{
        opacity: 0.45;
        border-color: rgba(239, 68, 68, 0.4);
        background: rgba(239, 68, 68, 0.05);
    }}
    .spectral-band-id {{
        font-family: var(--font-display);
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
    }}
    .spectral-band-name {{
        font-size: 0.74rem;
        color: var(--space-text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .spectral-band-wl {{
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: #93c5fd;
        margin-top: 2px;
    }}
    .spectral-band-status {{
        display: inline-block;
        margin-top: 5px;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: var(--font-tech);
        text-transform: uppercase;
    }}
    .spectral-band-status.on {{
        background: rgba(0, 240, 255, 0.2);
        color: var(--electric-cyan);
    }}
    .spectral-band-status.off {{
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
    }}
    .spectral-band-status.lock {{
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
    }}

    /* HUD Prediction Card */
    .voxel-hud-container {{
        background: linear-gradient(135deg, rgba(8, 20, 48, 0.95) 0%, rgba(4, 11, 28, 0.98) 100%);
        border: 1px solid rgba(0, 240, 255, 0.35);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0, 240, 255, 0.12);
        margin-top: 8px;
    }}
    .voxel-hud-class {{
        font-family: var(--font-display);
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #ffffff;
        margin-bottom: 2px;
    }}
    .voxel-hud-conf {{
        font-family: var(--font-mono);
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--electric-cyan);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_cinematic_hero():
    """Renders the sleek, compact A-SPACE inspired Earth-from-orbit landing hero section."""
    hero_html = """
    <div class="voxel-hero-compact">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span class="voxel-pulse-dot"></span>
            <span style="font-family: var(--font-tech); font-size: 0.82rem; font-weight: 700; color: var(--electric-cyan); letter-spacing: 0.1em; text-transform: uppercase;">
                ORBITAL SPECTRAL INTELLIGENCE // SENTINEL-2 L2A
            </span>
        </div>
        <div class="voxel-hero-title">VOXEL</div>
        <div class="voxel-hero-subtitle">Task-Adaptive Earth Observation &amp; Spectral Intelligence</div>
        <div class="voxel-hero-tagline">
            Observe Beyond the Visible. Bridging physical multi-spectral radiometry, 
            physics-guided metaheuristic band down-selection, and deep CNN-ViT spatial-spectral representations.
        </div>
        <div class="voxel-telemetry-ribbon">
            <div class="voxel-telemetry-pill">
                <div class="voxel-telemetry-label">Constellation</div>
                <div class="voxel-telemetry-value">Sentinel-2A/B MSI</div>
            </div>
            <div class="voxel-telemetry-pill">
                <div class="voxel-telemetry-label">Down-Selection</div>
                <div class="voxel-telemetry-value">13 ➔ 5–6 Bands (-61.5%)</div>
            </div>
            <div class="voxel-telemetry-pill">
                <div class="voxel-telemetry-label">Deep Representation</div>
                <div class="voxel-telemetry-value">1216-dim FusionNet</div>
            </div>
            <div class="voxel-telemetry-pill">
                <div class="voxel-telemetry-label">Holdout Integrity</div>
                <div class="voxel-telemetry-value">Hirakud Scene (Zero Overlap)</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)


def render_operational_header(title: str, subtitle: str, module_tag: str):
    """Renders a compact operational header for inner modules (Modules 1-3, Provenance)."""
    header_html = f"""
    <div class="voxel-operational-header">
        <div>
            <div class="voxel-operational-tag">{module_tag}</div>
            <div class="voxel-operational-title">{title}</div>
            <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 2px;">{subtitle}</div>
        </div>
        <div>
            <span class="voxel-badge-success">OPERATIONAL // VERIFIED</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_spectral_matrix(selected_bands: List[str]):
    """
    Renders an interactive visual electromagnetic spectrum matrix showing all 13 bands.
    Active channels glow electric cyan, Band 10 is refused with lock icon.
    """
    norm_selected = [b.upper() for b in selected_bands]

    chips_html = '<div class="spectral-matrix-grid">'
    for group in SPECTRAL_REGIMES:
        for b in group["bands"]:
            is_refused = b.get("refused", False)
            is_active = (b["id"] in norm_selected) and not is_refused

            if is_refused:
                card_class = "spectral-band-card refused"
                status_html = '<span class="spectral-band-status lock">🔒 REFUSED</span>'
            elif is_active:
                card_class = "spectral-band-card active"
                status_html = '<span class="spectral-band-status on">ACTIVE</span>'
            else:
                card_class = "spectral-band-card"
                status_html = '<span class="spectral-band-status off">OFF</span>'

            chips_html += f"""
            <div class="{card_class}">
                <div class="spectral-band-id" style="color: {group['color']}">{b['id']}</div>
                <div class="spectral-band-name">{b['name']}</div>
                <div class="spectral-band-wl">{b['wl']} · {b['res']}</div>
                {status_html}
            </div>
            """
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


def render_hud_prediction(prediction: Dict[str, Any], task_type: str, reduction_pct: float, compatibility_note: Optional[str] = None):
    """Renders the futuristic HUD result container for Module 3 classification output."""
    class_name = prediction.get("predicted_class_name", "Unknown")
    conf_pct = float(prediction.get("confidence_pct", 0.0))
    prov = prediction.get("provenance", "unknown")

    if prov == "real_inference":
        prov_badge = '<span class="voxel-badge-success">🛰️ GENUINE SATELLITE MODEL</span>'
    elif "withheld" in prov or "gated" in prov:
        prov_badge = '<span class="voxel-badge-warning">🛡️ SCIENTIFICALLY GATED</span>'
    else:
        prov_badge = '<span class="voxel-badge-warning">🧪 SYNTHETIC TEST FIXTURE</span>'

    note_html = ""
    if compatibility_note:
        note_html = f"""
        <div style="margin-top: 10px; padding: 8px 12px; background: rgba(56, 189, 248, 0.08); border-left: 3px solid var(--electric-cyan); font-size: 0.85rem; color: #cbd5e1;">
            {compatibility_note}
        </div>
        """

    hud_html = f"""
    <div class="voxel-hud-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-family: var(--font-tech); font-weight: 700; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; font-size: 0.82rem;">
                Task Profile: {task_type.upper()} // Volume Reduction: {reduction_pct:.1f}%
            </span>
            {prov_badge}
        </div>
        <div class="voxel-hud-class">{class_name}</div>
        <div style="display: flex; align-items: baseline; gap: 10px;">
            <div class="voxel-hud-conf">{conf_pct:.2f}%</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Softmax Probability Density</div>
        </div>
        {note_html}
    </div>
    """
    st.markdown(hud_html, unsafe_allow_html=True)


def style_dark_space_figure(fig, ax=None):
    """Applies deep space theme styling to a matplotlib figure and axes."""
    fig.patch.set_facecolor("#070d1e")
    if ax is not None:
        if isinstance(ax, (list, np.ndarray)):
            for a in np.array(ax).flat:
                _style_single_ax(a)
        else:
            _style_single_ax(ax)


def _style_single_ax(a):
    a.set_facecolor("#091228")
    a.tick_params(colors="#94a3b8", labelsize=9)
    for spine in a.spines.values():
        spine.set_color("#1e3a5f")
        spine.set_linewidth(1)
    if a.xaxis.label:
        a.xaxis.label.set_color("#cbd5e1")
    if a.yaxis.label:
        a.yaxis.label.set_color("#cbd5e1")
    if a.title:
        a.title.set_color("#f8fafc")
    a.grid(color="#172e4c", linestyle="--", linewidth=0.5)


def plot_dark_training_curve(csv_path: Path, title: str):
    """Renders dual-panel dark space training curve figure (Loss & Accuracy)."""
    if not csv_path.exists():
        return None
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2), dpi=150)

    # Panel 1: Loss
    epochs = df["epoch"]
    ax1.plot(epochs, df["train_loss"], marker="o", color="#00f0ff", label="Train Loss", linewidth=2, markersize=4)
    ax1.plot(epochs, df["val_loss"], marker="s", color="#f59e0b", label="Val Loss", linewidth=2, linestyle="--", markersize=4)
    ax1.set_title("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=8)
    ax1.set_ylabel("Loss", fontsize=8)
    ax1.legend(facecolor="#091228", edgecolor="#1e3a5f", fontsize=8)

    # Panel 2: Accuracy
    ax2.plot(epochs, df["train_oa"], marker="o", color="#10b981", label="Train OA (%)", linewidth=2, markersize=4)
    ax2.plot(epochs, df["val_oa"], marker="s", color="#38bdf8", label="Val OA (%)", linewidth=2, linestyle="--", markersize=4)
    ax2.set_title("Overall Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=8)
    ax2.set_ylabel("OA (%)", fontsize=8)
    ax2.legend(facecolor="#091228", edgecolor="#1e3a5f", fontsize=8)

    style_dark_space_figure(fig, [ax1, ax2])
    fig.suptitle(title, fontsize=11, fontweight="bold", color="#ffffff", y=1.02)
    fig.tight_layout()
    return fig
