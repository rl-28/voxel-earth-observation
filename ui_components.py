"""
VOXEL Earth Observation Intelligence Platform
UI Components, Design System Tokens & Scientific Visualization Utilities

Implements the unified dark geospatial intelligence design system:
- Background: #0B0F17
- Primary Surfaces: #151D2A
- Elevated Surfaces: #202A39
- Borders: Muted Slate (rgba(255, 255, 255, 0.08) / #263345)
- Text Primary: #F1F5F9
- Text Secondary: #94A3B8
- Spectral Data Accent: #67E8F9
- Primary Action Accent: #FF555D
- Success (Verified): #10B981
- Warning / Gated: #F59E0B
"""

from typing import Dict, List, Optional, Any
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# CSS DESIGN SYSTEM INJECTION
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Base Dark Geospatial Theme Variables */
:root {
    --bg-base: #0B0F17;
    --surface-primary: #151D2A;
    --surface-elevated: #202A39;
    --border-muted: #263345;
    --border-subtle: rgba(255, 255, 255, 0.07);
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --accent-cyan: #67E8F9;
    --accent-coral: #FF555D;
    --accent-green: #10B981;
    --accent-amber: #F59E0B;
}

/* App Background & Typography */
.stApp {
    background-color: #0B0F17 !important;
    color: #F1F5F9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #111722 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Main Content Surface container */
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    border-color: rgba(255, 255, 255, 0.08) !important;
}

/* Metric Cards & Summary Boxes */
.vx-card {
    background-color: #151D2A;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.vx-card-elevated {
    background-color: #202A39;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* Metric Display Unit */
.vx-metric {
    background: #151D2A;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
}

.vx-metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 4px;
}

.vx-metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.2;
}

.vx-metric-delta {
    font-size: 0.75rem;
    color: #67E8F9;
    margin-top: 2px;
}

/* Badges & Pills */
.vx-badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 4px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.vx-badge-success {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.vx-badge-warning {
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.vx-badge-info {
    background-color: rgba(103, 232, 249, 0.12);
    color: #67E8F9;
    border: 1px solid rgba(103, 232, 249, 0.25);
}

.vx-badge-danger {
    background-color: rgba(255, 85, 93, 0.15);
    color: #FF555D;
    border: 1px solid rgba(255, 85, 93, 0.3);
}

.vx-badge-neutral {
    background-color: rgba(148, 163, 184, 0.12);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

/* Stage Tracker Grid */
.vx-stage-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 16px 0;
}

.vx-stage-item {
    background-color: #151D2A;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 12px;
    position: relative;
}

.vx-stage-item.complete {
    border-left: 3px solid #10B981;
}

.vx-stage-item.partial {
    border-left: 3px solid #F59E0B;
}

.vx-stage-item.failed {
    border-left: 3px solid #FF555D;
}

.vx-stage-item.pending {
    border-left: 3px solid #64748B;
    opacity: 0.7;
}

.vx-stage-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 4px;
}

.vx-stage-detail {
    font-size: 0.72rem;
    color: #94A3B8;
    line-height: 1.4;
}

/* Spectral Band Grid */
.vx-band-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(76px, 1fr));
    gap: 8px;
    margin: 12px 0;
}

.vx-band-tile {
    text-align: center;
    padding: 8px 4px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    transition: all 0.15s ease;
}

.vx-band-tile.active {
    background-color: rgba(103, 232, 249, 0.15);
    border: 1px solid #67E8F9;
    color: #67E8F9;
}

.vx-band-tile.inactive {
    background-color: #151D2A;
    border: 1px solid rgba(255, 255, 255, 0.05);
    color: #64748B;
}

.vx-band-tile.refused {
    background-color: rgba(255, 85, 93, 0.1);
    border: 1px dashed rgba(255, 85, 93, 0.4);
    color: #FF555D;
}

.vx-band-name {
    font-size: 0.82rem;
    font-weight: 800;
    display: block;
}

.vx-band-role {
    font-size: 0.65rem;
    color: #94A3B8;
    display: block;
    margin-top: 2px;
}

/* Buttons */
button[kind="primary"] {
    background-color: #FF555D !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

button[kind="primary"]:hover {
    background-color: #E0444C !important;
}

/* Empty State Container */
.vx-empty-state {
    text-align: center;
    padding: 36px 20px;
    background-color: #151D2A;
    border: 1px dashed rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    margin: 12px 0;
}

.vx-empty-state-icon {
    font-size: 2rem;
    margin-bottom: 8px;
    opacity: 0.8;
}

.vx-empty-state-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #F1F5F9;
    margin-bottom: 4px;
}

.vx-empty-state-message {
    font-size: 0.8rem;
    color: #94A3B8;
    max-width: 460px;
    margin: 0 auto;
    line-height: 1.4;
}

/* Header Banner */
.vx-header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.vx-header-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}

.vx-header-subtitle {
    font-size: 0.85rem;
    color: #94A3B8;
    margin: 0;
}
</style>
"""


def inject_custom_styles():
    """Inject VOXEL custom dark geospatial CSS styles into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# REUSABLE UI COMPONENT RENDERERS
# -----------------------------------------------------------------------------

def render_page_header(title: str, subtitle: str, badge_text: Optional[str] = None, badge_type: str = "info"):
    """Renders a standard compact page header with optional status badge."""
    badge_html = f'<span class="vx-badge vx-badge-{badge_type}" style="margin-left: 12px; vertical-align: middle;">{badge_text}</span>' if badge_text else ""
    st.markdown(
        f"""
        <div class="vx-header">
            <h2 class="vx-header-title">{title}{badge_html}</h2>
            <p class="vx-header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, delta: Optional[str] = None):
    """Renders an individual metric card component."""
    delta_html = f'<div class="vx-metric-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="vx-metric">
            <div class="vx-metric-label">{label}</div>
            <div class="vx-metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(text: str, badge_type: str = "info") -> str:
    """Returns HTML for an inline status badge."""
    return f'<span class="vx-badge vx-badge-{badge_type}">{text}</span>'


def render_empty_state(icon: str, title: str, message: str, action_hint: Optional[str] = None):
    """Renders an elegant empty state placeholder for missing artifacts or non-executed stages."""
    hint_html = f'<div style="margin-top: 8px; font-size: 0.75rem; color: #67E8F9;">👉 {action_hint}</div>' if action_hint else ""
    st.markdown(
        f"""
        <div class="vx-empty-state">
            <div class="vx-empty-state-icon">{icon}</div>
            <div class="vx-empty-state-title">{title}</div>
            <div class="vx-empty-state-message">{message}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_band_tiles(selected_indices: List[int], band_names: List[str]):
    """
    Renders visual Sentinel-2 band selection tiles.
    Band 10 (Cirrus) is permanently styled with the refusal border.
    """
    band_roles = {
        "B01": "Coastal Aerosol (60m)",
        "B02": "Blue Visible (10m)",
        "B03": "Green Visible (10m)",
        "B04": "Red Visible (10m)",
        "B05": "Red Edge VRE 1 (20m)",
        "B06": "Red Edge VRE 2 (20m)",
        "B07": "Red Edge VRE 3 (20m)",
        "B08": "Broad NIR (10m)",
        "B8A": "Narrow NIR (20m)",
        "B09": "Water Vapour (60m)",
        "B10": "Cirrus Absorption (60m)",
        "B11": "SWIR 1 (20m)",
        "B12": "SWIR 2 (20m)",
    }

    tiles_html = ['<div class="vx-band-grid">']
    for idx, b_name in enumerate(band_names):
        is_selected = idx in selected_indices
        role = band_roles.get(b_name, "")

        if b_name == "B10":
            css_class = "refused"
            status_tag = "REFUSED"
        elif is_selected:
            css_class = "active"
            status_tag = "ACTIVE"
        else:
            css_class = "inactive"
            status_tag = "OFF"

        tiles_html.append(
            f"""
            <div class="vx-band-tile {css_class}">
                <span class="vx-band-name">{b_name}</span>
                <span style="font-size: 0.62rem; font-weight: 700;">{status_tag}</span>
                <span class="vx-band-role">{role}</span>
            </div>
            """
        )
    tiles_html.append('</div>')
    st.markdown("".join(tiles_html), unsafe_allow_html=True)


def render_scientific_heatmap(
    selected_indices: List[int],
    band_keys: List[str],
    title: str = "Spectral Channel Selection Heatmap",
):
    """
    Renders a clean, publication-ready spectral selection heatmap.
    Uses muted dark palette matching the interface.
    """
    binary_vector = [1 if i in selected_indices else 0 for i in range(len(band_keys))]
    
    fig, ax = plt.subplots(figsize=(9, 1.8), dpi=150)
    fig.patch.set_facecolor("#151D2A")
    ax.set_facecolor("#151D2A")

    # Dark scientific colormap (dark navy background to bright cyan active)
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "voxel_spectral",
        ["#202A39", "#67E8F9"],
        N=2,
    )

    ax.imshow([binary_vector], cmap=custom_cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(band_keys)))
    ax.set_xticklabels(band_keys, fontsize=8, color="#F1F5F9", fontweight="600")
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="#94A3B8", length=0)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color("rgba(255, 255, 255, 0.1)")

    # Tile labels
    for idx, val in enumerate(binary_vector):
        b_key = band_keys[idx] if idx < len(band_keys) else ""
        if b_key == "B10":
            text_label = "CIRRUS ∅"
            text_color = "#FF555D"
        elif val == 1:
            text_label = "ACTIVE"
            text_color = "#0B0F17"
        else:
            text_label = "EXCLUDED"
            text_color = "#64748B"

        ax.text(
            idx,
            0,
            text_label,
            ha="center",
            va="center",
            color=text_color,
            fontweight="bold",
            fontsize=7,
        )

    ax.set_title(
        f"{title} ({len(selected_indices)}/{len(band_keys)} Channels Active)",
        fontsize=9,
        fontweight="bold",
        color="#F1F5F9",
        pad=10,
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_shap_bar_chart(
    top_bands: List[Dict[str, Any]],
    title: str = "Permutation SHAP Spectral Attribution",
):
    """
    Renders horizontal bar chart of spectral attributions styled for the dark theme.
    """
    if not top_bands:
        return

    band_names = [b["name"] for b in top_bands]
    band_pcts = [b["percentage"] for b in top_bands]

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    fig.patch.set_facecolor("#151D2A")
    ax.set_facecolor("#151D2A")

    y_pos = np.arange(len(band_names))
    bars = ax.barh(y_pos, band_pcts, color="#67E8F9", height=0.6, align="center")
    
    # Highlight top contributing band with coral accent
    if len(bars) > 0:
        bars[0].set_color("#FF555D")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(band_names, fontsize=8, color="#F1F5F9", fontweight="600")
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance (%)", fontsize=8, color="#94A3B8")
    ax.tick_params(colors="#94A3B8", labelsize=7)

    for spine in ax.spines.values():
        spine.set_color("rgba(255, 255, 255, 0.08)")

    ax.grid(axis="x", color="rgba(255, 255, 255, 0.05)", linestyle="--", linewidth=0.5)

    ax.set_title(title, fontsize=9, fontweight="bold", color="#F1F5F9", pad=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
