"""
☀️ Solar Energy Analytics Dashboard — Streamlit
================================================
Production-level PV performance dashboard.
Deploy to Streamlit Cloud → connect GitHub repo → done.
"""

import warnings
warnings.filterwarnings("ignore")

import html

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="☀️ Solar PV Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  (dark theme tokens)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#080d14"
PANEL    = "#0f1621"
GRID     = "#1e2736"
TEXT     = "#dde6f0"
SUBTEXT  = "#8fa3ba"
C_TEAL   = "#00c896"
C_CYAN   = "#4fa8f0"
C_MAG    = "#e86ab0"
C_YEL    = "#f0b840"
C_ORA    = "#f07830"
C_PUR    = "#a87ae8"
C_GRN    = "#38b878"
C_RED    = "#e85050"
PALETTE  = [C_TEAL, C_CYAN, C_MAG, C_YEL, C_ORA, C_PUR, C_GRN, C_RED]

def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' — safe for all Plotly color props."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

LAYOUT = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, size=12, family="system-ui, -apple-system, sans-serif"),
    margin=dict(l=44, r=24, t=52, b=44),
    xaxis=dict(
        gridcolor="#1a2438", gridwidth=1,
        zerolinecolor="#1a2438", zerolinewidth=1,
        color=SUBTEXT, tickfont=dict(size=11),
        linecolor="#1a2438",
    ),
    yaxis=dict(
        gridcolor="#1a2438", gridwidth=1,
        zerolinecolor="#1a2438", zerolinewidth=1,
        color=SUBTEXT, tickfont=dict(size=11),
        linecolor="#1a2438",
    ),
    legend=dict(
        bgcolor="rgba(15,22,33,0.85)", bordercolor="#1e2736", borderwidth=1,
        font=dict(color=TEXT, size=11), itemsizing="constant",
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    hoverlabel=dict(bgcolor="#111a28", bordercolor="#28334a", font_color=TEXT, font_size=12),
    modebar=dict(bgcolor="transparent", color=SUBTEXT, activecolor=TEXT),
)

def _add_forecast_vline(fig, x_str, color, label="  Forecast →", size=12):
    """Datetime-safe forecast separator — avoids Plotly annotation_position bug."""
    fig.add_shape(
        type="line", x0=x_str, x1=x_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=color, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=x_str, y=0.97, xref="x", yref="paper",
        text=label, showarrow=False,
        font=dict(color=color, size=size),
        xanchor="left", yanchor="top",
        bgcolor="rgba(0,0,0,0)",
    )

def apply_layout(fig, title="", height=420):
    fig.update_layout(
        **LAYOUT,
        title=dict(text=title, font=dict(color=TEXT, size=13), x=0, xanchor="left", pad=dict(l=4)),
        height=height,
    )
    return fig


def _render_gradient_table_html(data: pd.DataFrame, max_rows: int = 200) -> str:
    """Render an original-style yellow/orange preview table without Pandas Styler.

    Streamlit Cloud can fail when st.dataframe receives a Pandas Styler object on
    newer Python/Pandas/Streamlit combinations. This keeps the visual treatment
    as plain HTML/CSS, so it does not need matplotlib and does not trigger
    Styler marshalling.
    """
    if data is None or data.empty:
        return "<div class='preview-empty'>No numeric data available to preview.</div>"

    table_df = data.tail(max_rows).copy()
    numeric_cols = set(table_df.select_dtypes(include=np.number).columns)

    stats_by_col = {}
    for col in numeric_cols:
        vals = pd.to_numeric(table_df[col], errors="coerce")
        vmin, vmax = vals.min(skipna=True), vals.max(skipna=True)
        valid = np.isfinite(vmin) and np.isfinite(vmax) and not np.isclose(vmin, vmax)
        stats_by_col[col] = (vmin, vmax, valid)

    start_rgb = np.array([255, 247, 188])  # pale yellow, close to YlOrRd low end
    end_rgb = np.array([217, 95, 14])      # orange/red, close to YlOrRd high end

    def _fmt(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):,.4g}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        return html.escape(str(value))

    def _cell_style(col, value):
        base = "border:1px solid #21262d;padding:6px 8px;text-align:right;white-space:nowrap;"
        if col not in numeric_cols or pd.isna(value):
            return base
        vmin, vmax, valid = stats_by_col[col]
        if not valid:
            return base
        try:
            t = float((value - vmin) / (vmax - vmin))
        except Exception:
            return base
        t = max(0.0, min(1.0, t))
        rgb = (start_rgb * (1 - t) + end_rgb * t).astype(int)
        text_color = "#0d1117" if t < 0.65 else "#ffffff"
        return base + f"background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]});color:{text_color};"

    # Header
    header_cells = ["<th class='preview-index-header'></th>"]
    for col in table_df.columns:
        header_cells.append(f"<th>{html.escape(str(col))}</th>")
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    # Body
    body_rows = []
    for idx, row in table_df.iterrows():
        idx_html = html.escape(str(idx))
        cells = [f"<td class='preview-index-cell'>{idx_html}</td>"]
        for col, value in row.items():
            cells.append(f"<td style='{_cell_style(col, value)}'>{_fmt(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<div class='preview-table-wrapper'>"
        "<table class='preview-table'>"
        "<thead>" + header_html + "</thead>"
        "<tbody>" + "".join(body_rows) + "</tbody>"
        "</table>"
        "</div>"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─────────────────────────────────────────────────────────────────────────────
   Solar PV Dashboard theme refresh
   Keeps the original dark dashboard look, but fixes newer Streamlit widget
   contrast issues and prevents the KPI cards from wrapping awkwardly.
   ───────────────────────────────────────────────────────────────────────────── */
:root {
    --bg: #080d14;
    --panel: #0f1621;
    --panel-2: #0a1018;
    --panel-hover: #141d2b;
    --grid: #1e2736;
    --grid-2: #28334a;
    --text: #dde6f0;
    --subtext: #8fa3ba;
    --muted: #5c7188;
    --accent: #00c896;
    --accent-dim: rgba(0,200,150,0.12);
    --accent-glow: rgba(0,200,150,0.25);
    --danger: #f05050;
}

/* ── root background + base text ── */
.stApp,
.main,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"] { background: linear-gradient(180deg,rgba(8,13,20,.95) 0%,transparent 100%) !important; backdrop-filter: blur(12px) !important; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

/* ── general typography contrast ── */
h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown li,
label, span, div, p {
    color: var(--text);
}
h1 { letter-spacing: -0.03em; }
h2 {
    color: var(--text) !important;
    border-bottom: 1px solid var(--grid);
    padding-bottom: 8px;
    margin-top: 0.75rem;
}
h3 { color: var(--text) !important; }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaptionContainer,
.caption {
    color: var(--subtext) !important;
    opacity: 1 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] strong {
    color: var(--text) !important;
    opacity: 1 !important;
}
[data-testid="stCaptionContainer"] *,
small {
    color: var(--subtext) !important;
    opacity: 1 !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0c1320 0%,#080d14 100%) !important;
    border-right: 1px solid var(--grid) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] hr { border:none !important; border-top:1px solid var(--grid) !important; margin:6px 0 !important; }
[data-testid="stSidebar"] section,
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: var(--subtext) !important; font-weight: 500 !important;
    font-size: 0.8rem !important; letter-spacing: 0.04em !important; text-transform: uppercase !important;
}

/* ── file uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel-2) !important;
    border: 1px dashed var(--grid-2) !important;
    border-radius: 10px !important; padding: 16px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--subtext) !important; }
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] [role="button"] {
    background: var(--accent) !important; color: #000 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 0.82rem !important;
}
[data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }

/* ── select / multiselect controls ── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    background-color: var(--panel-2) !important; border: 1px solid var(--grid-2) !important;
    border-radius: 8px !important; color: var(--text) !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-dim) !important;
}
[data-baseweb="select"] input,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea { color: var(--text) !important; }
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: var(--panel) !important; border: 1px solid var(--grid-2) !important;
    border-radius: 8px !important; box-shadow: 0 8px 32px rgba(0,0,0,.5) !important;
}
[data-baseweb="menu"] li, [role="option"] {
    background-color: transparent !important; color: var(--text) !important;
    border-radius: 6px !important; margin: 2px 4px !important;
}
[data-baseweb="menu"] li:hover, [role="option"]:hover { background-color: var(--panel-hover) !important; }
[data-baseweb="tag"] {
    background-color: var(--accent-dim) !important;
    border: 1px solid rgba(0,200,150,.3) !important; border-radius: 6px !important;
}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {
    color: var(--accent) !important; fill: var(--accent) !important;
}

/* ── sliders ── */
[data-testid="stSlider"] * { color: var(--text) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div { color: var(--text) !important; }
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent) !important; border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}

/* ── radio buttons ── */
[data-testid="stRadio"] label, [data-testid="stRadio"] label span,
[role="radiogroup"] label, [role="radiogroup"] label span { color: var(--text) !important; opacity: 1 !important; }
[data-testid="stRadio"] p { color: var(--subtext) !important; font-weight: 500 !important; font-size: 0.8rem !important; text-transform: uppercase !important; }
[data-testid="stRadio"] div[role="radiogroup"] label div,
[data-testid="stRadio"] div[role="radiogroup"] label p,
[data-testid="stRadio"] div[role="radiogroup"] label span { color: var(--text) !important; opacity: 1 !important; }

/* ── alerts ── */
[data-testid="stAlert"] {
    background: linear-gradient(135deg,rgba(0,200,150,.06) 0%,rgba(15,22,33,.9) 100%) !important;
    border: 1px solid rgba(0,200,150,.25) !important; border-radius: 8px !important;
    color: var(--text) !important;
}
[data-testid="stAlert"] * { color: var(--text) !important; }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(145deg,#121c2e 0%,#0a1018 100%);
    border-radius: 14px; padding: 18px 14px 16px 14px;
    border: 1px solid var(--grid); text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.04);
    min-height: 120px; height: 120px;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    overflow: hidden; position: relative;
    transition: transform .15s ease,box-shadow .15s ease;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,0,0,.4); }
.kpi-value {
    font-size: clamp(1.15rem,1.55vw,1.7rem); line-height: 1.1; font-weight: 800;
    margin: 0 0 8px 0; white-space: nowrap; letter-spacing: -0.03em;
}
.kpi-label {
    font-size: 0.68rem; color: var(--subtext) !important; text-transform: uppercase;
    letter-spacing: .08em; line-height: 1.4; max-width: 100%; font-weight: 500;
}
@media (max-width: 1350px) { .kpi-grid { grid-template-columns: repeat(2,minmax(160px,1fr)); } }
@media (max-width: 760px) { .kpi-grid { grid-template-columns: 1fr; } }

/* ── tab bar ── */
[data-baseweb="tab-list"] {
    background: var(--panel) !important; border-radius: 10px !important;
    border: 1px solid var(--grid) !important; padding: 3px !important; gap: 2px !important;
}
[data-baseweb="tab"] {
    color: var(--muted) !important; font-weight: 600 !important;
    font-size: 0.88rem !important; border-radius: 7px !important; padding: 7px 14px !important;
}
[data-baseweb="tab"]:hover { color: var(--text) !important; background: var(--panel-hover) !important; }
[data-baseweb="tab"] p, [data-baseweb="tab"] span { color: inherit !important; }
[aria-selected="true"] {
    color: var(--accent) !important; background: var(--accent-dim) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── metric widget ── */
[data-testid="metric-container"] {
    background: var(--panel) !important; border-radius: 10px !important;
    padding: 14px 18px !important; border: 1px solid var(--grid) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.2) !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--subtext) !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: .05em !important; }

/* ── buttons ── */
.stDownloadButton button, .stButton button {
    background: linear-gradient(135deg,var(--accent) 0%,#3d9fe0 100%) !important;
    color: #030810 !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 0.86rem !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
}
.stDownloadButton button:hover, .stButton button:hover { opacity: .9 !important; }

/* ── dataframes and tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--grid) !important;
    border-radius: 9px !important;
    overflow: hidden !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--grid-2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a4a60; }
.js-plotly-plot .plotly { background: transparent !important; }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* ── raw data preview table ── */
.preview-table-wrapper {
    max-height: 320px; overflow: auto;
    border: 1px solid var(--grid); border-radius: 10px; background: var(--panel);
}
.preview-table { border-collapse: collapse; min-width: 100%; font-size: 0.76rem; color: var(--text); font-variant-numeric: tabular-nums; }
.preview-table thead th {
    position: sticky; top: 0; z-index: 2; background: #111a27;
    color: var(--subtext); border-bottom: 1px solid var(--grid-2); border-right: 1px solid var(--grid);
    padding: 8px 10px; text-align: right; white-space: nowrap;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 600;
}
.preview-table tbody tr:nth-child(even) { background: rgba(255,255,255,.015); }
.preview-table tbody tr:hover { background: var(--panel-hover) !important; }
.preview-index-header, .preview-index-cell {
    position: sticky; left: 0; z-index: 3; background: #0a1018 !important;
    color: var(--muted) !important; border-right: 1px solid var(--grid-2);
    padding: 6px 10px; text-align: left; white-space: nowrap; font-size: 0.72rem;
}
.preview-empty { background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; padding: 16px; color: var(--subtext); text-align: center; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA  (load CSV or auto-generate synthetic)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATED FORECAST + YIELD PIPELINE
# Integrates logic from:
#   • irradiance LSTM notebook: lag/time features + recursive 1-year forecast
#   • temperature LSTM notebook: lag/time features + recursive 1-year forecast
#   • combined code notebook: merge-style actual/predicted yield calculations
#
# Note: the notebooks supplied model-training code, but not saved .keras/.pkl model
# weights. This dashboard therefore runs an automatic, deployable forecasting
# pipeline from the uploaded CSV itself. If the CSV already contains prediction
# columns, those are preserved; otherwise the app trains lightweight lag/time
# forecasters and then calculates all actual/predicted yield columns.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_CAPACITY_KW = 100.0
# Kept as AREA_DEFAULT_M2 for the existing formulas, but it now represents
# the 100 kW reference scale used by the original dashboard. The previous
# 40,000 value made the displayed efficiency round to 0.00%.
AREA_DEFAULT_M2 = SYSTEM_CAPACITY_KW
GAMMA_DEFAULT = -0.004
T_REF_DEFAULT_C = 25.0
DEFAULT_ETA_REF = 0.185
FORECAST_DAYS = 365

COLUMN_ALIASES = {
    "datetime": ["date_time", "timestamp", "time", "date", "datetime_utc", "datetime_local"],
    "irradiance_avg": ["avg_irradiance", "average_irradiance", "irradiance", "global_irradiance", "ghi", "actual_irradiance"],
    "avg_module_temp": ["module_temp", "module_temperature", "avg_temp", "temperature", "actual_temperature", "panel_temperature", "pv_temperature"],
    "generated_yield": ["energy_yield", "yield", "generated_energy", "energy_kwh", "actual_energy_yield", "actual_measured_energy_kwh"],
    "power_kw": ["power", "ac_power", "dc_power", "active_power", "power_output", "p_ac", "p_dc"],
    "power_analyzer": ["power_w", "power_analyser", "power_meter", "power_analyzer_w"],
    "predicted_irradiance": ["forecasted_irradiance", "irradiance_prediction", "pred_irradiance", "predicted_ghi"],
    "predicted_temperature": ["forecasted_module_temp", "temperature_prediction", "pred_temperature", "predicted_module_temp"],
    "actual_irradiance": ["measured_irradiance", "observed_irradiance"],
    "actual_temperature": ["measured_temperature", "observed_temperature", "actual_module_temp"],
}


def _normalise_column_name(name: str) -> str:
    out = str(name).strip().lower()
    for old, new in [("°", ""), ("/", "_"), ("-", "_"), (" ", "_"), ("(", ""), (")", "")]:
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalise_column_name(c) for c in df.columns]
    rename = {}
    existing = set(df.columns)
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in existing:
            continue
        for alias in aliases:
            alias_norm = _normalise_column_name(alias)
            if alias_norm in existing:
                rename[alias_norm] = canonical
                existing.add(canonical)
                break
    if rename:
        df = df.rename(columns=rename)
    # Keep first instance if aliases created duplicates.
    return df.loc[:, ~df.columns.duplicated()].copy()


def _find_datetime_column(df: pd.DataFrame) -> str | None:
    if "datetime" in df.columns:
        return "datetime"
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() >= 0.80:
            return c
    return None


def _median_interval_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    diffs = pd.Series(index).sort_values().diff().dropna().dt.total_seconds() / 3600
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1.0
    interval = float(diffs.median())
    if not np.isfinite(interval) or interval <= 0:
        return 1.0
    return interval


def _steps_per_day(index: pd.DatetimeIndex) -> int:
    return max(1, int(round(24 / _median_interval_hours(index))))


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in ["source_file", "season"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _prepare_datetime_index(raw: pd.DataFrame) -> pd.DataFrame:
    df = _apply_column_aliases(raw)
    dt_col = _find_datetime_column(df)
    if dt_col is None:
        # Last resort: create a synthetic hourly timeline so the dashboard still runs.
        df["datetime"] = pd.date_range("2022-01-01", periods=len(df), freq="h")
    elif dt_col != "datetime":
        df = df.rename(columns={dt_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return _coerce_numeric(df)


def _ensure_core_measurements(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("irr_sensor")]

    if "irradiance_avg" not in df.columns:
        if sensor_cols:
            df["irradiance_avg"] = df[sensor_cols].mean(axis=1)
        elif "actual_irradiance" in df.columns:
            df["irradiance_avg"] = df["actual_irradiance"]
        elif "predicted_irradiance" in df.columns:
            df["irradiance_avg"] = df["predicted_irradiance"]
        else:
            # Clear-sky-like fallback from time features; keeps the dashboard usable.
            hour = df.index.hour.values
            doy = df.index.dayofyear.values
            lat = np.radians(31.95)
            decl = np.radians(23.45) * np.sin(np.radians(360 / 365 * (doy - 81)))
            ha = np.radians(15 * (hour - 12))
            cos_z = np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha), 0, 1)
            df["irradiance_avg"] = 0.75 * 1361.0 * cos_z

    if "actual_irradiance" not in df.columns:
        df["actual_irradiance"] = df["irradiance_avg"]

    if "avg_module_temp" not in df.columns:
        if "actual_temperature" in df.columns:
            df["avg_module_temp"] = df["actual_temperature"]
        else:
            # Empirical module temperature fallback from irradiance and seasonality.
            hour = df.index.hour.values
            doy = df.index.dayofyear.values
            ambient = 18 + 10 * np.sin(np.radians(360 / 365 * (doy - 80))) + 5 * np.sin(np.radians(360 / 24 * (hour - 14)))
            df["avg_module_temp"] = ambient + 0.025 * df["irradiance_avg"].fillna(0)

    if "actual_temperature" not in df.columns:
        df["actual_temperature"] = df["avg_module_temp"]

    # Tilted irradiance is useful for existing dashboard panels. Use average as fallback.
    if "irradiation_tilted" not in df.columns:
        df["irradiation_tilted"] = df["irradiance_avg"]

    interval_h = _median_interval_hours(df.index)

    # Derive power and measured interval energy as robustly as possible.
    if "power_kw" not in df.columns:
        if "power_analyzer" in df.columns:
            p = pd.to_numeric(df["power_analyzer"], errors="coerce")
            # Some PV meters record exported generation as negative power.
            # Use magnitude for production dashboards, then convert W -> kW when needed.
            p_abs = p.abs()
            df["power_kw"] = p_abs / 1000.0 if p_abs.quantile(0.95) > 500 else p_abs
        elif "generated_yield" in df.columns:
            gy = df["generated_yield"].copy()
            daily_diff = gy.groupby(df.index.date).diff()
            looks_cumulative = daily_diff.dropna().ge(-1e-9).mean() > 0.80 and gy.quantile(0.95) > max(daily_diff.quantile(0.95), 1e-9) * 3
            if looks_cumulative:
                energy_interval = daily_diff.clip(lower=0).fillna(gy.clip(lower=0))
            else:
                energy_interval = gy.clip(lower=0)
            df["energy_kwh"] = energy_interval
            df["power_kw"] = energy_interval / interval_h
        else:
            irr_kw = df["irradiance_avg"].clip(lower=0) / 1000.0
            df["power_kw"] = AREA_DEFAULT_M2 * DEFAULT_ETA_REF * irr_kw * (1 + GAMMA_DEFAULT * (df["avg_module_temp"] - T_REF_DEFAULT_C))
            df["power_kw"] = df["power_kw"].clip(lower=0)

    if "energy_kwh" not in df.columns:
        df["energy_kwh"] = df["power_kw"].clip(lower=0) * interval_h

    if "generated_yield" not in df.columns:
        df["generated_yield"] = df["energy_kwh"]

    # Fill small gaps before modelling.
    for c in ["irradiance_avg", "actual_irradiance", "avg_module_temp", "actual_temperature", "power_kw", "energy_kwh"]:
        if c in df.columns:
            df[c] = df[c].interpolate(limit=4).ffill().bfill()

    return df


def _cyclical_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "hour_sin":   np.sin(2 * np.pi * index.hour / 24),
        "hour_cos":   np.cos(2 * np.pi * index.hour / 24),
        "month_sin":  np.sin(2 * np.pi * index.month / 12),
        "month_cos":  np.cos(2 * np.pi * index.month / 12),
        "minute_sin": np.sin(2 * np.pi * index.minute / 60),
        "minute_cos": np.cos(2 * np.pi * index.minute / 60),
        "doy_sin":    np.sin(2 * np.pi * index.dayofyear / 365.25),
        "doy_cos":    np.cos(2 * np.pi * index.dayofyear / 365.25),
    }, index=index)


def _supervised_matrix(df: pd.DataFrame, target_col: str, extra_cols: list[str] | None = None):
    extra_cols = extra_cols or []
    spd = _steps_per_day(df.index)
    lag_hour = max(1, int(round(1 / _median_interval_hours(df.index))))
    lags = {"lag_1": 1, "lag_1h": lag_hour, "lag_24h": spd}

    target = df[target_col].astype(float)
    X = _cyclical_features(df.index)
    for name, lag in lags.items():
        X[f"{target_col}_{name}"] = target.shift(lag)
    X[f"{target_col}_roll_day_mean"] = target.shift(1).rolling(spd, min_periods=max(2, spd // 4)).mean()
    X[f"{target_col}_roll_day_std"] = target.shift(1).rolling(spd, min_periods=max(2, spd // 4)).std().fillna(0)

    for col in extra_cols:
        if col in df.columns:
            s = df[col].astype(float)
            X[col] = s
            X[f"{col}_lag_1"] = s.shift(1)
            X[f"{col}_roll_day_mean"] = s.shift(1).rolling(spd, min_periods=max(2, spd // 4)).mean()

    y = target.rename("target")
    valid = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid], list(X.columns), lags, spd


def _fit_lag_model(df: pd.DataFrame, target_col: str, extra_cols: list[str] | None = None):
    X, y, feature_cols, lags, spd = _supervised_matrix(df, target_col, extra_cols)
    if len(X) < 80:
        return None, feature_cols, lags, spd
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X[feature_cols], y)
        return model, feature_cols, lags, spd
    except Exception:
        return None, feature_cols, lags, spd


def _seasonal_persistence(series: pd.Series, floor_zero: bool = False) -> pd.Series:
    spd = _steps_per_day(series.index)
    pred = series.shift(spd).combine_first(series.shift(1)).combine_first(series.rolling(spd, min_periods=1).mean()).combine_first(series)
    if floor_zero:
        pred = pred.clip(lower=0)
    return pred


def _auto_predict_column(df: pd.DataFrame, target_col: str, pred_col: str, extra_cols: list[str] | None = None, floor_zero: bool = False) -> pd.DataFrame:
    df = df.copy()
    # Preserve externally supplied LSTM/model predictions if present.
    if pred_col in df.columns and df[pred_col].notna().sum() >= max(10, int(0.1 * len(df))):
        df[pred_col] = df[pred_col].astype(float).interpolate(limit=4).ffill().bfill()
        if floor_zero:
            df[pred_col] = df[pred_col].clip(lower=0)
        return df

    model, feature_cols, *_ = _fit_lag_model(df, target_col, extra_cols)
    pred = _seasonal_persistence(df[target_col], floor_zero=floor_zero)

    if model is not None:
        X, _, _, _, _ = _supervised_matrix(df, target_col, extra_cols)
        try:
            model_pred = pd.Series(model.predict(X[feature_cols]), index=X.index)
            if floor_zero:
                model_pred = model_pred.clip(lower=0)
            pred.loc[model_pred.index] = model_pred
        except Exception:
            pass

    df[pred_col] = pred.astype(float).interpolate(limit=4).ffill().bfill()
    if floor_zero:
        df[pred_col] = df[pred_col].clip(lower=0)
    return df


def _feature_row_for_future(ts, history: pd.Series, feature_cols: list[str], lags: dict, spd: int, extra_history: dict[str, pd.Series] | None = None, extra_current: dict[str, float] | None = None) -> pd.DataFrame:
    extra_history = extra_history or {}
    extra_current = extra_current or {}
    row = _cyclical_features(pd.DatetimeIndex([ts])).iloc[0].to_dict()
    target_col = history.name

    for name, lag in lags.items():
        row[f"{target_col}_{name}"] = float(history.iloc[-lag]) if len(history) >= lag else float(history.iloc[-1])
    recent = history.iloc[-spd:] if len(history) >= spd else history
    row[f"{target_col}_roll_day_mean"] = float(recent.mean())
    row[f"{target_col}_roll_day_std"] = float(recent.std()) if len(recent) > 1 else 0.0

    for col, hist in extra_history.items():
        row[col] = float(extra_current.get(col, hist.iloc[-1] if len(hist) else 0.0))
        row[f"{col}_lag_1"] = float(hist.iloc[-1]) if len(hist) else row[col]
        recent_extra = hist.iloc[-spd:] if len(hist) >= spd else hist
        row[f"{col}_roll_day_mean"] = float(recent_extra.mean()) if len(recent_extra) else row[col]

    return pd.DataFrame([[row.get(c, 0.0) for c in feature_cols]], columns=feature_cols)



def _dataset_coverage_days(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    return float((index.max() - index.min()).total_seconds() / 86400.0) + _median_interval_hours(index) / 24.0


def _solar_clear_sky_irradiance(index: pd.DatetimeIndex, lat_deg: float = 31.95) -> pd.Series:
    """Simple clear-sky-shaped irradiance baseline for Amman/Jordan-like PV data."""
    lat = np.radians(lat_deg)
    doy = index.dayofyear.values.astype(float)
    hour_decimal = index.hour.values + index.minute.values / 60.0 + index.second.values / 3600.0
    decl = np.radians(23.45) * np.sin(np.radians(360.0 / 365.25 * (doy - 81.0)))
    ha = np.radians(15.0 * (hour_decimal - 12.0))
    cos_z = np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha), 0, 1)
    return pd.Series(0.75 * 1361.0 * cos_z, index=index, name="clear_sky_irradiance")


def _empirical_module_temperature(index: pd.DatetimeIndex, irradiance: pd.Series | np.ndarray) -> pd.Series:
    """Annual + diurnal module-temperature baseline, calibrated later to the uploaded CSV."""
    irr = pd.Series(np.asarray(irradiance, dtype=float), index=index).clip(lower=0)
    doy = index.dayofyear.values.astype(float)
    hour_decimal = index.hour.values + index.minute.values / 60.0 + index.second.values / 3600.0
    seasonal = 17.0 + 11.0 * np.sin(2 * np.pi * (doy - 105.0) / 365.25)
    diurnal = 4.5 * np.sin(2 * np.pi * (hour_decimal - 14.0) / 24.0)
    return pd.Series(seasonal + diurnal + 0.025 * irr.values, index=index, name="module_temp_baseline")


def _seasonal_profile_forecast(df: pd.DataFrame, target_col: str, future_index: pd.DatetimeIndex, floor_zero: bool = False) -> pd.Series:
    hist = pd.DataFrame({
        "month": df.index.month,
        "hour": df.index.hour,
        "minute": df.index.minute,
        "target": df[target_col].astype(float).values,
    })
    fut = pd.DataFrame({"month": future_index.month, "hour": future_index.hour, "minute": future_index.minute}, index=future_index)

    profile_mhm = hist.groupby(["month", "hour", "minute"])["target"].median().rename("target").reset_index()
    out = fut.reset_index(names="datetime").merge(profile_mhm, on=["month", "hour", "minute"], how="left").set_index("datetime")["target"]

    if out.isna().any():
        profile_hm = hist.groupby(["hour", "minute"])["target"].median().rename("fallback").reset_index()
        fb = fut.reset_index(names="datetime").merge(profile_hm, on=["hour", "minute"], how="left").set_index("datetime")["fallback"]
        out = out.combine_first(fb)
    if out.isna().any():
        profile_h = hist.groupby("hour")["target"].median()
        fb_hour = pd.Series([profile_h.get(ts.hour, np.nan) for ts in future_index], index=future_index)
        out = out.combine_first(fb_hour)
    out = out.fillna(df[target_col].median())
    if floor_zero:
        out = out.clip(lower=0)
    return out.astype(float)


def _solar_seasonal_baseline_forecast(df: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Used when the uploaded file is too short for a responsible full-year ML forecast.
    It uses the uploaded data to calibrate clear-sky irradiance and module temperature,
    then applies the annual solar-seasonal shape so unseen months do not become flat or jumpy.
    """
    hist_clear = _solar_clear_sky_irradiance(df.index)
    hist_irr = df["actual_irradiance"].astype(float).clip(lower=0)
    daylight = hist_clear > 50

    ratio = (hist_irr[daylight] / hist_clear[daylight]).replace([np.inf, -np.inf], np.nan).dropna().clip(0.05, 1.15)
    global_ratio = float(ratio.median()) if len(ratio) else 0.65
    if not np.isfinite(global_ratio) or global_ratio <= 0:
        global_ratio = 0.65

    ratio_df = pd.DataFrame({
        "hour": df.index[daylight].hour,
        "minute": df.index[daylight].minute,
        "ratio": ratio.reindex(df.index[daylight]).values,
    }).dropna()
    future_clear = _solar_clear_sky_irradiance(future_index)

    if not ratio_df.empty:
        time_ratio = ratio_df.groupby(["hour", "minute"])["ratio"].median().clip(0.05, 1.15).rename("ratio").reset_index()
        fut_keys = pd.DataFrame({"hour": future_index.hour, "minute": future_index.minute}, index=future_index)
        tr = fut_keys.reset_index(names="datetime").merge(time_ratio, on=["hour", "minute"], how="left").set_index("datetime")["ratio"]
        tr = tr.fillna(global_ratio)
        future_irr = future_clear * (0.70 * global_ratio + 0.30 * tr)
    else:
        future_irr = future_clear * global_ratio
    future_irr = future_irr.clip(lower=0)

    # Calibrate the temperature baseline to the uploaded file, then forecast the future.
    hist_temp = df["actual_temperature"].astype(float)
    hist_temp_baseline = _empirical_module_temperature(df.index, hist_irr)
    temp_offset = (hist_temp - hist_temp_baseline).replace([np.inf, -np.inf], np.nan).dropna().median()
    temp_offset = float(temp_offset) if np.isfinite(temp_offset) else 0.0
    future_temp = _empirical_module_temperature(future_index, future_irr) + temp_offset

    future = pd.DataFrame({
        "predicted_irradiance": future_irr.values,
        "forecasted_irradiance": future_irr.values,
        "predicted_temperature": future_temp.values,
        "forecasted_module_temp": future_temp.values,
    }, index=future_index)
    future.index.name = "datetime"
    future.attrs["forecast_method"] = "Solar-seasonal baseline forecast"
    future.attrs["forecast_warning"] = (
        "Only a short history is available, so the 1-year forecast is a calibrated solar-seasonal baseline, "
        "not a high-confidence ML forecast. Upload 6–12 months or a full year for better annual forecasting."
    )
    return future


def _batch_time_model_forecast(df: pd.DataFrame, target_col: str, future_index: pd.DatetimeIndex, extra_train: pd.Series | None = None, extra_future: pd.Series | None = None, floor_zero: bool = False) -> pd.Series:
    # Fast deployable forecast for Streamlit: learns calendar seasonality when the
    # uploaded data covers enough months. For short files, _recursive_forecast uses
    # the solar-seasonal baseline instead to avoid flat/jumpy unseen-month forecasts.
    fallback = _seasonal_profile_forecast(df, target_col, future_index, floor_zero=floor_zero)
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        X_train = _cyclical_features(df.index)
        y_train = df[target_col].astype(float)
        if extra_train is not None and extra_future is not None:
            X_train["driver"] = extra_train.astype(float).reindex(df.index).interpolate().ffill().bfill().values
        valid = X_train.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & y_train.notna()
        if valid.sum() < 80:
            return fallback
        model = ExtraTreesRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train.loc[valid], y_train.loc[valid])
        X_future = _cyclical_features(future_index)
        if extra_train is not None and extra_future is not None:
            X_future["driver"] = extra_future.astype(float).reindex(future_index).interpolate().ffill().bfill().values
        pred = pd.Series(model.predict(X_future), index=future_index)
        # Blend model and seasonal profile for stability on a full year horizon.
        pred = 0.70 * pred + 0.30 * fallback
        if floor_zero:
            pred = pred.clip(lower=0)
        return pred.astype(float)
    except Exception:
        return fallback


def _recursive_forecast(df: pd.DataFrame, days: int = FORECAST_DAYS) -> pd.DataFrame:
    if len(df) < 48:
        return pd.DataFrame()

    interval_h = _median_interval_hours(df.index)
    freq = pd.to_timedelta(interval_h, unit="h")
    n_steps = max(1, int(round(days * 24 / interval_h)))
    n_steps = min(n_steps, 365 * 24 * 4)  # safety cap for 15-min one-year forecasts
    future_index = pd.date_range(df.index.max() + freq, periods=n_steps, freq=freq)

    coverage_days = _dataset_coverage_days(df.index)
    observed_months = int(pd.Index(df.index.month).nunique())

    # A full-year ML forecast from only January/February data is misleading:
    # tree models cannot learn unseen seasons. Use a calibrated solar-seasonal
    # baseline instead for short or low-season-coverage uploads.
    if coverage_days < 180 or observed_months < 6:
        future = _solar_seasonal_baseline_forecast(df, future_index)
        future.attrs["coverage_days"] = coverage_days
        future.attrs["observed_months"] = observed_months
        return future

    future_irr = _batch_time_model_forecast(
        df,
        "actual_irradiance",
        future_index,
        floor_zero=True,
    )
    future_temp = _batch_time_model_forecast(
        df,
        "actual_temperature",
        future_index,
        extra_train=df["actual_irradiance"],
        extra_future=future_irr,
        floor_zero=False,
    )

    future = pd.DataFrame({
        "predicted_irradiance": future_irr.values,
        "forecasted_irradiance": future_irr.values,
        "predicted_temperature": future_temp.values,
        "forecasted_module_temp": future_temp.values,
    }, index=future_index)
    future.index.name = "datetime"
    future.attrs["forecast_method"] = "Calendar ML forecast"
    future.attrs["coverage_days"] = coverage_days
    future.attrs["observed_months"] = observed_months
    return future

def _compute_yield_columns(df: pd.DataFrame, future: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    df = df.copy()
    interval_h = _median_interval_hours(df.index)

    if "predicted_irradiance" not in df.columns:
        df["predicted_irradiance"] = df["actual_irradiance"]
    if "predicted_temperature" not in df.columns:
        df["predicted_temperature"] = df["actual_temperature"]

    df["irradiance_kw_m2"] = df["irradiance_avg"].clip(lower=0) / 1000.0
    denom = df["irradiance_kw_m2"] * AREA_DEFAULT_M2
    # Keep two versions:
    #   efficiency_ratio is used internally for power/yield calculations.
    #   efficiency is displayed in the dashboard as a human-readable percent.
    df["efficiency_ratio"] = np.where(denom > 0, df["power_kw"] / denom, np.nan)
    df.loc[df["irradiance_avg"] < 50, "efficiency_ratio"] = np.nan
    df["efficiency_ratio"] = pd.Series(df["efficiency_ratio"], index=df.index).clip(0, 1)
    df["efficiency"] = (df["efficiency_ratio"] * 100).clip(0, 25)

    eta_candidates = df.loc[df["irradiance_avg"] > 200, "efficiency_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    eta_ref = float(eta_candidates.median()) if len(eta_candidates) else DEFAULT_ETA_REF
    if not np.isfinite(eta_ref) or eta_ref <= 0 or eta_ref > 1:
        eta_ref = DEFAULT_ETA_REF
    df["eta_ref_used"] = eta_ref

    df["irradiance_error"] = df["actual_irradiance"] - df["predicted_irradiance"]
    df["irradiance_abs_error"] = df["irradiance_error"].abs()
    df["temp_error"] = df["actual_temperature"] - df["predicted_temperature"]
    df["temp_abs_error"] = df["temp_error"].abs()

    df["predicted_irradiance_kw_m2"] = df["predicted_irradiance"].clip(lower=0) / 1000.0
    df["actual_irradiance_kw_m2"] = df["actual_irradiance"].clip(lower=0) / 1000.0
    temp_factor_pred = (1 + GAMMA_DEFAULT * (df["predicted_temperature"] - T_REF_DEFAULT_C)).clip(lower=0)
    temp_factor_act = (1 + GAMMA_DEFAULT * (df["actual_temperature"] - T_REF_DEFAULT_C)).clip(lower=0)

    df["Predicted_Power_kW"] = (df["predicted_irradiance_kw_m2"] * AREA_DEFAULT_M2 * eta_ref * temp_factor_pred).clip(lower=0)
    df["Actual_Model_Power_kW"] = (df["actual_irradiance_kw_m2"] * AREA_DEFAULT_M2 * eta_ref * temp_factor_act).clip(lower=0)
    df["Predicted_Energy_Yield"] = df["Predicted_Power_kW"] * interval_h
    df["Actual_Energy_Yield"] = df["Actual_Model_Power_kW"] * interval_h
    df["Actual_Measured_Energy_kWh"] = df["energy_kwh"]
    df["energy_yield_error"] = df["Actual_Energy_Yield"] - df["Predicted_Energy_Yield"]
    df["energy_yield_abs_error"] = df["energy_yield_error"].abs()

    if future is not None and not future.empty:
        future = future.copy()
        future["predicted_irradiance_kw_m2"] = future["predicted_irradiance"].clip(lower=0) / 1000.0
        temp_factor_future = (1 + GAMMA_DEFAULT * (future["predicted_temperature"] - T_REF_DEFAULT_C)).clip(lower=0)
        future["forecast_power_kw"] = (future["predicted_irradiance_kw_m2"] * AREA_DEFAULT_M2 * eta_ref * temp_factor_future).clip(lower=0)
        future["Predicted_Energy_Yield"] = future["forecast_power_kw"] * interval_h
        future["eta_ref_used"] = eta_ref
    return df, future


def _prepare_uploaded_pipeline(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _prepare_datetime_index(raw)
    df = _ensure_core_measurements(df)
    _add_time_features(df)
    df = _auto_predict_column(df, "actual_irradiance", "predicted_irradiance", floor_zero=True)
    df = _auto_predict_column(df, "actual_temperature", "predicted_temperature", extra_cols=["actual_irradiance"], floor_zero=False)
    future = _recursive_forecast(df, days=FORECAST_DAYS)
    df, future = _compute_yield_columns(df, future)
    df["source_file"] = df.get("source_file", "uploaded_csv")
    if future is not None:
        df.attrs["forecast_method"] = future.attrs.get("forecast_method", "Automatic forecast")
        df.attrs["forecast_warning"] = future.attrs.get("forecast_warning", "")
        df.attrs["forecast_coverage_days"] = future.attrs.get("coverage_days", _dataset_coverage_days(df.index))
        df.attrs["forecast_observed_months"] = future.attrs.get("observed_months", int(pd.Index(df.index.month).nunique()))
    df.attrs["future_forecast"] = future
    df.attrs["auto_pipeline"] = True
    return df, future


@st.cache_data(show_spinner="🔄 Loading, forecasting, and calculating yields…")
def load_data(uploaded=None):
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        if not raw.empty:
            df, future = _prepare_uploaded_pipeline(raw)
            df.attrs["future_forecast"] = future
            return df, False

    df_syn = _generate_synthetic()
    df_syn = _ensure_core_measurements(df_syn)
    df_syn = _auto_predict_column(df_syn, "actual_irradiance", "predicted_irradiance", floor_zero=True)
    df_syn = _auto_predict_column(df_syn, "actual_temperature", "predicted_temperature", extra_cols=["actual_irradiance"], floor_zero=False)
    future_syn = _recursive_forecast(df_syn, days=FORECAST_DAYS)
    df_syn, future_syn = _compute_yield_columns(df_syn, future_syn)
    if future_syn is not None:
        df_syn.attrs["forecast_method"] = future_syn.attrs.get("forecast_method", "Automatic forecast")
        df_syn.attrs["forecast_warning"] = future_syn.attrs.get("forecast_warning", "")
        df_syn.attrs["forecast_coverage_days"] = future_syn.attrs.get("coverage_days", _dataset_coverage_days(df_syn.index))
        df_syn.attrs["forecast_observed_months"] = future_syn.attrs.get("observed_months", int(pd.Index(df_syn.index.month).nunique()))
    df_syn.attrs["future_forecast"] = future_syn
    df_syn.attrs["auto_pipeline"] = True
    return df_syn, True

def _generate_synthetic() -> tuple:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", periods=8760, freq="h")
    n   = len(idx)
    hour = idx.hour.values
    doy  = idx.dayofyear.values

    lat  = np.radians(31.95)
    decl = np.radians(23.45) * np.sin(np.radians(360 / 365 * (doy - 81)))
    ha   = np.radians(15 * (hour - 12))
    cos_z = np.clip(np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(ha), 0, 1)
    G_clear = 0.75 * 1361.0 * cos_z

    cloud    = np.repeat(rng.beta(2, 5, n // 24 + 1)[: n // 24], 24)[:n]
    irr_base = G_clear * (0.3 + 0.7 * (1 - cloud))

    sensors = {}
    for i in range(1, 7):
        off  = rng.uniform(-15, 15)
        noise = rng.normal(0, irr_base * 0.02 + 1, n)
        sensors[f"irr_sensor_{i}"] = np.clip(irr_base + off + noise, 0, 1300)

    irr_avg    = np.mean([sensors[k] for k in sensors], axis=0)
    irr_tilted = np.clip(irr_avg * 1.08 * rng.normal(1, 0.01, n), 0, 1350)

    T_amb = (10 + 10 * np.sin(np.radians(360 / 365 * (doy - 15)))
             + 6 * np.sin(np.radians(360 / 24 * (hour - 14)))
             + rng.normal(0, 2, n))
    T_mod = T_amb + 0.03 * irr_avg + rng.normal(0, 1, n)

    eta_stc, temp_coeff = 0.185, -0.0045
    irr_kw   = irr_avg / 1000.0
    eta      = eta_stc * (1 + temp_coeff * (T_mod - 25))
    power_kw = np.clip(100.0 * irr_kw * eta, 0, None)

    yield_arr, daily_cum = np.zeros(n), 0.0
    for i in range(n):
        if i > 0 and idx[i].date() != idx[i - 1].date():
            daily_cum = 0.0
        daily_cum    += power_kw[i]
        yield_arr[i]  = daily_cum

    pred_irr  = irr_avg * rng.normal(1.0, 0.05, n)
    pred_temp = T_mod   + rng.normal(0, 1.5, n)

    df = pd.DataFrame({
        **sensors,
        "irradiation_tilted":         irr_tilted,
        "power_analyzer":             np.clip(power_kw * 1000, 0, None),
        "generated_yield":            yield_arr,
        "avg_module_temp":            T_mod,
        "irradiance_avg":             irr_avg,
        "source_file":                "synthetic_2022",
        "actual_irradiance":          irr_avg,
        "predicted_irradiance":       pred_irr,
        "actual_temperature":         T_mod,
        "predicted_temperature":      pred_temp,
        "power_kw":                   power_kw,
        "irradiance_kw_m2":           irr_kw,
        "efficiency":                 np.clip(eta * 100, 0, 25),
        "irradiance_error":           irr_avg - pred_irr,
        "irradiance_abs_error":       np.abs(irr_avg - pred_irr),
        "temp_error":                 T_mod - pred_temp,
        "temp_abs_error":             np.abs(T_mod - pred_temp),
        "predicted_irradiance_kw_m2": pred_irr / 1000.0,
        "Predicted_Energy_Yield":     np.clip(100 * (pred_irr / 1000) * eta_stc, 0, None),
    }, index=idx)

    _add_time_features(df)
    return df


def _add_time_features(df):
    df["hour"]      = df.index.hour
    df["month"]     = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["doy"]       = df.index.dayofyear
    df["season"]    = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Autumn",  10: "Autumn", 11: "Autumn",
    })


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:16px 4px 10px 4px'>"
        "<div style='font-size:1.05rem;font-weight:800;letter-spacing:-.02em;color:#dde6f0'>☀️ Solar PV Dashboard</div>"
        "<div style='font-size:.7rem;color:#5c7188;margin-top:3px;letter-spacing:.04em;text-transform:uppercase'>Amman, Jordan · 100 kW System</div>"
        "</div>", unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#00c896;padding:2px 0 6px 0'>📂 Data Source</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Schema: datetime, irr_sensor_1…6, power_kw, etc.", label_visibility="collapsed")
    st.divider()
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#00c896;padding:2px 0 6px 0'>🎛️ Filters</div>", unsafe_allow_html=True)

result = load_data(uploaded_file)
df_raw, is_synthetic = result

with st.sidebar:
    yr_min = df_raw.index.year.min()
    yr_max = df_raw.index.year.max()

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    sel_months = st.multiselect(
        "Months", options=list(range(1, 13)),
        format_func=lambda m: month_names[m - 1],
        default=list(range(1, 13)),
    )
    sel_hours = st.slider("Hours of day", 0, 23, (0, 23))
    irr_min   = st.slider(
        "Min irradiance filter (W/m²)", 0, 200, 0,
        help="Filter out low-irradiance / night records",
    )

    st.divider()
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#00c896;padding:2px 0 6px 0'>📊 Visualisation</div>", unsafe_allow_html=True)
    scatter_n = st.slider("Scatter sample size", 500, 5000, 2000, step=500)
    st.divider()
    if is_synthetic:
        st.markdown("<div style='background:rgba(240,120,48,.1);border:1px solid rgba(240,120,48,.3);border-radius:8px;padding:10px 12px;font-size:.8rem;color:#f0a070'>⚠️ Synthetic data active. Upload your CSV above.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:rgba(0,200,150,.08);border:1px solid rgba(0,200,150,.25);border-radius:8px;padding:10px 12px;font-size:.8rem;color:#00c896'>✅ Uploaded dataset loaded.</div>", unsafe_allow_html=True)

# ─── apply filters ────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_months:
    df = df[df["month"].isin(sel_months)]
df = df[(df["hour"] >= sel_hours[0]) & (df["hour"] <= sel_hours[1])]
if irr_min > 0:
    df = df[df["irradiance_avg"] >= irr_min]

df_day = df[df["irradiance_avg"] > 20].copy()
df_day["sensor_std"]      = df_day[[c for c in df_day.columns if "irr_sensor" in c]].std(axis=1)
df_day["clearness_index"] = df_day["irradiance_avg"] / df_raw["irradiance_avg"].max()

if df.empty:
    st.error("No data after filtering — please adjust the sidebar filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
_yr_start = df_raw.index.year.min()
_yr_end   = df_raw.index.year.max()
_yr_label = f"{_yr_start}" if _yr_start == _yr_end else f"{_yr_start}–{_yr_end}"
st.markdown(
    f"<div style='padding:16px 0 6px 0'>"
    f"<h1 style='color:#dde6f0;margin:0;font-size:clamp(1.4rem,2.5vw,2rem)'>☀️ Solar PV Analytics Dashboard</h1>"
    f"<p style='color:#5c7188;margin:5px 0 0 0;font-size:.85rem'>"
    f"{_yr_label}&nbsp;&middot;&nbsp;100 kW Rooftop&nbsp;&middot;&nbsp;Amman, Jordan"
    f"&nbsp;&nbsp;<span style='color:#8fa3ba'>|</span>&nbsp;&nbsp;"
    f"<b style='color:#00c896'>{len(df):,}</b> <span style='color:#5c7188'>records selected</span></p></div>",
    unsafe_allow_html=True,
)
st.markdown("<hr style='border:none;border-top:1px solid #1e2736;margin:6px 0 18px 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
total_energy   = df["energy_kwh"].sum() if "energy_kwh" in df.columns else df["power_kw"].sum()
peak_power     = df["power_kw"].max()
avg_eff        = df_day["efficiency"].mean() if not df_day.empty else 0
_peak_ref      = max(df_raw["power_kw"].max() if "power_kw" in df_raw.columns else peak_power, 0.001)
cap_factor     = df["power_kw"].mean() / _peak_ref * 100
irr_mae        = df_day["irradiance_abs_error"].mean() if not df_day.empty else 0
if "temp_abs_error" in df_raw.columns and len(df_raw) > 0:
    _raw_day   = df_raw[df_raw["irradiance_avg"] > 20] if "irradiance_avg" in df_raw.columns else df_raw
    temp_mae   = _raw_day["temp_abs_error"].mean() if not _raw_day.empty else df_raw["temp_abs_error"].mean()
else:
    temp_mae   = df_day["temp_abs_error"].mean() if not df_day.empty else 0
avg_irr        = df["irradiance_avg"].mean()
avg_temp       = df["avg_module_temp"].mean()

kpi_data = [
    ("Total Energy Yield",    f"{total_energy:,.0f} kWh",  C_TEAL, "⚡"),
    ("Peak Power Output",     f"{peak_power:.2f} kW",      C_CYAN, "🔝"),
    ("Avg Module Efficiency", f"{avg_eff:.2f}%",           C_YEL,  "📐"),
    ("Capacity Factor",       f"{cap_factor:.1f}%",        C_ORA,  "📊"),
    ("Avg Irradiance",        f"{avg_irr:.0f} W/m²",       C_PUR,  "☀️"),
    ("Avg Module Temp",       f"{avg_temp:.1f} °C",        C_MAG,  "🌡️"),
    ("Irradiance MAE",        f"{irr_mae:.1f} W/m²",       C_GRN,  "📡"),
    ("Temp MAE (full data)",  f"{temp_mae:.4f} °C",        C_RED,  "🎯"),
]

for row_start in range(0, len(kpi_data), 4):
    kpi_cols = st.columns(4)
    for col, (label, value, color, icon) in zip(kpi_cols, kpi_data[row_start:row_start + 4]):
        with col:
            card_html = (
                f"<div class='kpi-card' style='border-top:3px solid {color};'>"
                f"<div style='font-size:1rem;margin-bottom:5px;opacity:.65'>{icon}</div>"
                f"<div class='kpi-value' style='color:{color};'>{html.escape(str(value))}</div>"
                f"<div class='kpi-label'>{html.escape(str(label))}</div>"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Time Series",
    "📊 Distributions",
    "🔥 Heatmaps",
    "🔗 Correlations",
    "🎯 Model Analysis",
    "🔬 Advanced",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📈 Time-Series & Trend Analysis")

    daily = df.resample("D").agg(
        power_total=("power_kw", "sum"),
        irradiance_mean=("irradiance_avg", "mean"),
        temp_mean=("avg_module_temp", "mean"),
        efficiency_mean=("efficiency", "mean"),
    ).dropna()

    # ── Forecast data for extending charts ──────────────────────────────
    _forecast_future = df_raw.attrs.get("future_forecast")
    _has_forecast = isinstance(_forecast_future, pd.DataFrame) and not _forecast_future.empty
    if _has_forecast:
        _f_daily = _forecast_future.resample("D").agg(
            power_total=("Predicted_Energy_Yield", "sum"),
            irradiance_mean=("predicted_irradiance", "mean"),
            temp_mean=("predicted_temperature", "mean"),
        ).dropna()
        # ISO string — avoids Plotly datetime arithmetic bug in add_shape/add_annotation
        _sep = daily.index.max().isoformat()

    # ── Daily energy + rolling averages ─────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["power_total"],
        fill="tozeroy", mode="lines",
        line=dict(color=C_CYAN, width=0.9),
        fillcolor=hex_to_rgba(C_CYAN, 0.10),
        name="Daily kWh (actual)",
    ))
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["power_total"].rolling(7).mean(),
        mode="lines", line=dict(color=C_TEAL, width=2), name="7-day MA",
    ))
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["power_total"].rolling(30).mean(),
        mode="lines", line=dict(color=C_YEL, width=2.5), name="30-day MA",
    ))
    if _has_forecast and not _f_daily.empty:
        fig.add_trace(go.Scatter(
            x=_f_daily.index, y=_f_daily["power_total"],
            fill="tozeroy", mode="lines",
            line=dict(color=C_ORA, width=0.9, dash="dot"),
            fillcolor=hex_to_rgba(C_ORA, 0.07), name="Forecast kWh",
        ))
        fig.add_trace(go.Scatter(
            x=_f_daily.index, y=_f_daily["power_total"].rolling(7).mean(),
            mode="lines", line=dict(color=C_ORA, width=2, dash="dash"),
            name="Forecast 7-day MA",
        ))
        _add_forecast_vline(fig, _sep, C_ORA)
    if len(daily) > 0:
        pk = daily["power_total"].idxmax()
        fig.add_annotation(
            x=pk, y=daily.loc[pk, "power_total"],
            text=f"⚡ Peak<br>{pk.strftime('%b %d, %Y')}",
            showarrow=True, arrowhead=2, arrowcolor=C_YEL,
            font=dict(color=C_YEL, size=11), bgcolor=PANEL, bordercolor=C_YEL,
        )
    _end_yr = _f_daily.index.max().year if (_has_forecast and not _f_daily.empty) else daily.index.max().year
    apply_layout(fig, f"Daily Energy Output — {daily.index.min().year}–{_end_yr} (kWh/day)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly bars + efficiency trend ─────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        monthly_irr = df.groupby("month")["irradiance_avg"].mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[month_names[m - 1] for m in monthly_irr.index],
            y=monthly_irr.values,
            marker=dict(
                color=monthly_irr.values,
                colorscale=[[0, hex_to_rgba(C_CYAN, 0.7)], [0.5, hex_to_rgba(C_TEAL, 0.85)], [1, C_YEL]],
                showscale=False,
            ),
            opacity=0.9, name="Avg Irradiance (W/m²)",
        ))
        fig2.add_trace(go.Scatter(
            x=[month_names[m - 1] for m in monthly_irr.index],
            y=monthly_irr.values,
            mode="lines+markers", line=dict(color=C_YEL, width=2),
            marker=dict(size=7, color=C_YEL), name="Trend",
        ))
        apply_layout(fig2, "Monthly Average Irradiance (W/m²)", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        eff_roll30 = daily["efficiency_mean"].rolling(30).mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=daily.index, y=daily["efficiency_mean"],
            mode="lines", line=dict(color=C_PUR, width=0.6),
            opacity=0.35, name="Daily Efficiency",
        ))
        fig3.add_trace(go.Scatter(
            x=daily.index, y=eff_roll30,
            mode="lines", line=dict(color=C_MAG, width=2.5),
            name="30-day MA (actual)",
            fill="tonexty", fillcolor=hex_to_rgba(C_MAG, 0.07),
        ))
        if _has_forecast and not _f_daily.empty:
            _eta_v = float(df_raw["eta_ref_used"].median()) if "eta_ref_used" in df_raw.columns else 0.185
            _f_eff_ma = (_eta_v * (1 + (-0.004) * (_f_daily["temp_mean"] - 25.0)) * 100).clip(0, 25).rolling(30, min_periods=1).mean()
            fig3.add_trace(go.Scatter(
                x=_f_daily.index, y=_f_eff_ma,
                mode="lines", line=dict(color=C_ORA, width=2, dash="dash"),
                name="30-day MA (forecast)", opacity=0.8,
            ))
            _add_forecast_vline(fig3, _sep, C_ORA, size=11)
        apply_layout(fig3, "Module Efficiency Trend (%)", height=340)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Rolling volatility ───────────────────────────────────────────────
    _spd = 96  # 15-min data: 96 steps/day
    vol_7d = df["power_kw"].rolling(_spd * 7).std()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=vol_7d.index, y=vol_7d.values,
        mode="lines", line=dict(color=C_ORA, width=1.4),
        fill="tozeroy", fillcolor=hex_to_rgba(C_ORA, 0.12),
        name="7-day rolling σ (actual)",
    ))
    if _has_forecast and not _forecast_future.empty and "forecast_power_kw" in _forecast_future.columns:
        _fc_vol = _forecast_future["forecast_power_kw"].rolling(_spd * 7).std()
        fig4.add_trace(go.Scatter(
            x=_fc_vol.index, y=_fc_vol.values,
            mode="lines", line=dict(color=C_YEL, width=1.4, dash="dot"),
            fill="tozeroy", fillcolor=hex_to_rgba(C_YEL, 0.06),
            name="7-day rolling σ (forecast)",
        ))
        _add_forecast_vline(fig4, _sep, C_ORA, size=11)
    apply_layout(fig4, "Power Output — 7-Day Rolling Volatility (σ kW)", height=280)
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Feature Distributions")

    dist_cols = [
        "irradiance_avg", "power_kw", "avg_module_temp", "efficiency",
        "generated_yield", "irradiation_tilted", "irradiance_abs_error", "temp_abs_error",
    ]

    c1, c2 = st.columns(2)
    col_toggle = [c1, c2, c1, c2, c1, c2, c1, c2]

    for col_container, col_name in zip(col_toggle, dist_cols):
        with col_container:
            data_src = df_day[col_name].dropna() if col_name in df_day.columns else df[col_name].dropna()
            if len(data_src) < 10:
                continue
            color = PALETTE[dist_cols.index(col_name) % len(PALETTE)]
            sk, ku = data_src.skew(), data_src.kurt()

            fig_d = go.Figure()
            fig_d.add_trace(go.Histogram(
                x=data_src, nbinsx=60,
                marker_color=color, opacity=0.35,
                histnorm="probability density", name="Histogram",
                showlegend=False,
            ))
            # KDE
            kde = gaussian_kde(data_src)
            xs  = np.linspace(data_src.min(), data_src.max(), 300)
            fig_d.add_trace(go.Scatter(
                x=xs, y=kde(xs), mode="lines",
                line=dict(color=color, width=2.5), name="KDE",
            ))
            fig_d.add_vline(x=data_src.mean(),   line=dict(color=C_YEL, width=1.5, dash="dash"),
                            annotation_text=f"μ={data_src.mean():.1f}",
                            annotation_font_color=C_YEL)
            fig_d.add_vline(x=data_src.median(), line=dict(color=C_MAG, width=1.5, dash="dot"),
                            annotation_text=f"M={data_src.median():.1f}",
                            annotation_font_color=C_MAG)

            title_d = f"{col_name.replace('_',' ').title()} — skew={sk:.2f} | kurt={ku:.2f}"
            apply_layout(fig_d, title_d, height=280)
            st.plotly_chart(fig_d, use_container_width=True)

    # ── Boxplot panel ──────────────────────────────────────────────────────
    st.markdown("### Outlier Detection — Boxplots")
    fig_box = go.Figure()
    for i, col_name in enumerate(dist_cols):
        data_src = df_day[col_name].dropna() if col_name in df_day.columns else df[col_name].dropna()
        fig_box.add_trace(go.Box(
            y=data_src,
            name=col_name.replace("_", " "),
            marker_color=PALETTE[i % len(PALETTE)],
            line_color=PALETTE[i % len(PALETTE)],
            fillcolor="rgba(0,0,0,0)",
            boxmean=True,
        ))
    apply_layout(fig_box, "IQR Boxplots — All Key Variables", height=400)
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🔥 Seasonal Heatmaps — Production Fingerprint")
    st.caption("Hour-of-day × Month: reveals *when* energy is produced across the year.")

    hm_metric = st.radio(
        "Select metric",
        ["irradiance_avg", "power_kw", "efficiency", "avg_module_temp"],
        format_func=lambda c: c.replace("_", " ").title(),
        horizontal=True,
    )

    piv = df.pivot_table(values=hm_metric, index="hour", columns="month", aggfunc="mean")
    piv.columns = [month_names[m - 1] for m in piv.columns]

    color_scales = {
        "irradiance_avg": "plasma",
        "power_kw":       "viridis",
        "efficiency":     "magma",
        "avg_module_temp":"RdYlBu_r",
    }

    fig_hm = go.Figure(go.Heatmap(
        z=piv.values,
        x=piv.columns.tolist(),
        y=piv.index.tolist(),
        colorscale=color_scales.get(hm_metric, "plasma"),
        colorbar=dict(title=hm_metric.replace("_", " "), tickfont=dict(color=TEXT)),
        hoverongaps=False,
        hovertemplate="Month: %{x}<br>Hour: %{y}<br>Value: %{z:.2f}<extra></extra>",
    ))
    apply_layout(fig_hm,
                 f"Hour × Month Heatmap — {hm_metric.replace('_',' ').title()}",
                 height=520)
    fig_hm.update_layout(
        yaxis_title="Hour of Day", xaxis_title="Month",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Weekday × Month ────────────────────────────────────────────────────
    st.markdown("### Weekday × Month — Average Power (kW)")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    piv_wd = df.pivot_table(values="power_kw", index="dayofweek", columns="month", aggfunc="mean")
    piv_wd.index   = [day_names[i] for i in piv_wd.index]
    piv_wd.columns = [month_names[m - 1] for m in piv_wd.columns]

    fig_wd = go.Figure(go.Heatmap(
        z=piv_wd.values, x=piv_wd.columns.tolist(), y=piv_wd.index.tolist(),
        colorscale="teal",
        colorbar=dict(title="kW", tickfont=dict(color=TEXT)),
        hovertemplate="Month: %{x}<br>Day: %{y}<br>Avg Power: %{z:.2f} kW<extra></extra>",
    ))
    apply_layout(fig_wd, "Weekday × Month — Avg Power (kW)", height=380)
    st.plotly_chart(fig_wd, use_container_width=True)

    # ── Sensor bias heatmap ────────────────────────────────────────────────
    st.markdown("### Sensor Bias Matrix (Hour × Sensor)")
    sensor_cols = [c for c in df_day.columns if c.startswith("irr_sensor")]
    if sensor_cols:
        bias_data = {c: df_day[c] - df_day["irradiance_avg"] for c in sensor_cols}
        bias_df   = pd.DataFrame(bias_data)
        bias_df["hour"] = df_day["hour"]
        piv_bias = bias_df.groupby("hour")[sensor_cols].mean()

        fig_sb = go.Figure(go.Heatmap(
            z=piv_bias.values,
            x=[c.replace("irr_","") for c in sensor_cols],
            y=piv_bias.index.tolist(),
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="Bias (W/m²)", tickfont=dict(color=TEXT)),
            hovertemplate="Sensor: %{x}<br>Hour: %{y}<br>Bias: %{z:.2f} W/m²<extra></extra>",
        ))
        apply_layout(fig_sb, "Hourly Sensor Bias vs Array Average (W/m²)", height=380)
        st.plotly_chart(fig_sb, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔗 Correlation & Relationship Analysis")

    corr_cols = [
        "irradiance_avg", "irradiation_tilted", "power_kw", "generated_yield",
        "avg_module_temp", "efficiency", "irradiance_kw_m2",
        "actual_irradiance", "predicted_irradiance",
        "irradiance_abs_error", "temp_abs_error", "Predicted_Energy_Yield",
    ]
    available = [c for c in corr_cols if c in df_day.columns]
    corr = df_day[available].corr()

    # ── Heatmap ────────────────────────────────────────────────────────────
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    z_masked = corr.values.copy()
    z_masked[mask] = None

    fig_corr = go.Figure(go.Heatmap(
        z=z_masked,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, C_PUR], [0.5, PANEL], [1, C_TEAL]],
        zmid=0, zmin=-1, zmax=1,
        text=np.where(~mask, np.round(corr.values, 2).astype(str), ""),
        texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT),
        colorbar=dict(title="Pearson r", tickfont=dict(color=TEXT)),
        hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
    ))
    apply_layout(fig_corr, "Pearson Correlation Matrix — Daytime Records", height=560)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Scatter deep-dive ──────────────────────────────────────────────────
    st.markdown("### Interactive Scatter Deep-Dive")
    num_cols_list = [c for c in df_day.select_dtypes(include=np.number).columns
                     if c not in ["hour","month","dayofweek","doy"]]

    sc1, sc2, sc3 = st.columns(3)
    x_col = sc1.selectbox("X axis", num_cols_list, index=num_cols_list.index("irradiance_avg") if "irradiance_avg" in num_cols_list else 0)
    y_col = sc2.selectbox("Y axis", num_cols_list, index=num_cols_list.index("power_kw") if "power_kw" in num_cols_list else 1)
    c_col = sc3.selectbox("Colour by", ["season","month","hour"] + num_cols_list, index=0)

    # Prevent duplicate-axis selections from crashing Plotly/Narwhals.
    # This restores the original dashboard behavior: show a friendly note instead of plotting.
    if x_col == y_col:
        st.warning("You can't do that — you have to pick two different axes.")
    else:
        # Keep column names unique. This also prevents crashes when the colour variable
        # is the same as either the X or Y variable.
        scatter_cols = list(dict.fromkeys([x_col, y_col, c_col]))
        scatter_base = df_day[scatter_cols].dropna()

        if scatter_base.empty:
            st.warning("No valid rows are available for this scatter plot after filtering.")
        else:
            samp = scatter_base.sample(min(scatter_n, len(scatter_base)), random_state=0)
            x_vals = samp[x_col].to_numpy()
            y_vals = samp[y_col].to_numpy()
            color_vals = samp[c_col]

            if c_col in ["season", "month", "hour"]:
                season_cmap = {"Winter": C_CYAN, "Spring": C_GRN, "Summer": C_YEL, "Autumn": C_ORA}
                if c_col == "season":
                    colors_sc = color_vals.map(season_cmap).fillna(SUBTEXT).tolist()
                else:
                    scaled = (color_vals - color_vals.min()) / (color_vals.max() - color_vals.min() + 1e-9)
                    colors_sc = px.colors.sample_colorscale("plasma", scaled.to_numpy())

                fig_sc = go.Figure(go.Scatter(
                    x=x_vals, y=y_vals, mode="markers",
                    marker=dict(color=colors_sc, size=5, opacity=0.55),
                    text=color_vals.astype(str).to_numpy(),
                    hovertemplate=f"{x_col}=%{{x:.2f}}<br>{y_col}=%{{y:.2f}}<br>{c_col}=%{{text}}<extra></extra>",
                ))
            else:
                fig_sc = go.Figure(go.Scatter(
                    x=x_vals, y=y_vals, mode="markers",
                    marker=dict(color=color_vals.to_numpy(), colorscale="plasma", size=5, opacity=0.55,
                                colorbar=dict(title=c_col, tickfont=dict(color=TEXT))),
                    hovertemplate=f"{x_col}=%{{x:.2f}}<br>{y_col}=%{{y:.2f}}<extra></extra>",
                ))

            # regression line
            valid_sc = df_day[[x_col, y_col]].dropna()
            if len(valid_sc) > 10:
                x_reg = valid_sc[x_col].to_numpy()
                y_reg = valid_sc[y_col].to_numpy()
                m_sc, b_sc, r_sc, *_ = stats.linregress(x_reg, y_reg)
                xs_sc = np.linspace(np.nanmin(x_reg), np.nanmax(x_reg), 100)
                fig_sc.add_trace(go.Scatter(
                    x=xs_sc, y=m_sc*xs_sc+b_sc, mode="lines",
                    line=dict(color=C_RED, width=2.5, dash="dash"),
                    name=f"r = {r_sc:.3f}",
                ))

            apply_layout(fig_sc, f"{y_col} vs {x_col} (n={len(samp):,})", height=440)
            fig_sc.update_xaxes(title=x_col.replace("_", " "))
            fig_sc.update_yaxes(title=y_col.replace("_", " "))
            st.plotly_chart(fig_sc, use_container_width=True)

    # ── Top correlations bar ───────────────────────────────────────────────
    if "power_kw" in corr.columns:
        st.markdown("### Top Feature Correlations with **power_kw**")
        top_corr = corr["power_kw"].drop("power_kw").abs().sort_values(ascending=True)
        colors_bar = [C_RED if corr.loc[f, "power_kw"] < 0 else C_TEAL for f in top_corr.index]
        fig_bar = go.Figure(go.Bar(
            x=corr.loc[top_corr.index, "power_kw"].values,
            y=top_corr.index.tolist(),
            orientation="h",
            marker_color=colors_bar, opacity=0.85,
            text=[f"{v:.3f}" for v in corr.loc[top_corr.index, "power_kw"].values],
            textposition="outside", textfont=dict(color=TEXT),
        ))
        apply_layout(fig_bar, "Pearson r with power_kw", height=380)
        fig_bar.update_xaxes(title="Pearson r")
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🎯 Prediction Model Analysis")

    forecast_future = df_raw.attrs.get("future_forecast")
    if isinstance(forecast_future, pd.DataFrame) and not forecast_future.empty:
        st.success("✅ Automatic pipeline ran: CSV preprocessing → irradiance forecast → temperature forecast → actual/predicted yield calculation.")
        forecast_warning = df_raw.attrs.get("forecast_warning", "")
        forecast_method = df_raw.attrs.get("forecast_method", forecast_future.attrs.get("forecast_method", "Automatic forecast"))
        coverage_days = df_raw.attrs.get("forecast_coverage_days", None)
        observed_months = df_raw.attrs.get("forecast_observed_months", None)
        if forecast_warning:
            st.warning(f"⚠️ {forecast_warning} Detected history: {coverage_days:.0f} days across {observed_months} month(s).")
        else:
            st.caption(f"Forecast method: {forecast_method}")

        f_daily = forecast_future.resample("D").agg(
            predicted_irradiance=("predicted_irradiance", "mean"),
            predicted_temperature=("predicted_temperature", "mean"),
            predicted_energy=("Predicted_Energy_Yield", "sum"),
            forecast_power_kw=("forecast_power_kw", "max"),
        ).dropna()

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Forecast horizon", f"{forecast_future.index.min().date()} → {forecast_future.index.max().date()}")
        fc2.metric("Predicted annual yield", f"{forecast_future['Predicted_Energy_Yield'].sum():,.0f} kWh")
        _fc_daily_peak = forecast_future["forecast_power_kw"].resample("D").max()
        _peak_fc_power = float(_fc_daily_peak.quantile(0.99))
        fc3.metric("Peak forecast power (p99)", f"{_peak_fc_power:.2f} kW")
        fc4.metric("Avg forecast temp", f"{forecast_future['predicted_temperature'].mean():.1f} °C")

        fig_fc = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fc.add_trace(go.Scatter(
            x=f_daily.index, y=f_daily["predicted_irradiance"],
            mode="lines", line=dict(color=C_TEAL, width=1.7),
            name="Forecast irradiance (W/m²)",
        ), secondary_y=False)
        fig_fc.add_trace(go.Scatter(
            x=f_daily.index, y=f_daily["predicted_temperature"],
            mode="lines", line=dict(color=C_MAG, width=1.7),
            name="Forecast module temp (°C)",
        ), secondary_y=True)
        chart_title = f"{forecast_method} — Daily Irradiance and Module Temperature"
        apply_layout(fig_fc, chart_title, height=360)
        fig_fc.update_yaxes(title_text="Irradiance (W/m²)", secondary_y=False)
        fig_fc.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
        st.plotly_chart(fig_fc, use_container_width=True)

        f_monthly = forecast_future.resample("ME")["Predicted_Energy_Yield"].sum()
        fig_fm = go.Figure(go.Bar(
            x=f_monthly.index.strftime("%b %Y"),
            y=f_monthly.values,
            marker_color=C_YEL, opacity=0.85,
            name="Predicted Energy Yield",
        ))
        apply_layout(fig_fm, "Forecasted Monthly Energy Yield (kWh)", height=300)
        fig_fm.update_yaxes(title="kWh")
        st.plotly_chart(fig_fm, use_container_width=True)

        forecast_csv = forecast_future.reset_index().to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download 1-Year Forecast CSV",
            data=forecast_csv,
            file_name="solar_1year_auto_forecast.csv",
            mime="text/csv",
            key="download_auto_forecast",
        )
        st.markdown("---")

    if "actual_irradiance" not in df_day.columns or df_day.empty:
        st.warning("Prediction columns not available in this dataset.")
    else:
        c1, c2 = st.columns(2)

        # Actual vs Predicted Irradiance
        with c1:
            samp_p = df_day[["actual_irradiance","predicted_irradiance"]].dropna().sample(
                min(scatter_n, len(df_day)), random_state=1)
            r2_irr = np.corrcoef(samp_p["actual_irradiance"], samp_p["predicted_irradiance"])[0,1]**2
            mn_p, mx_p = samp_p.min().min(), samp_p.max().max()

            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=samp_p["actual_irradiance"], y=samp_p["predicted_irradiance"],
                mode="markers", marker=dict(color=C_TEAL, size=4, opacity=0.3),
                name="Samples",
            ))
            fig_avp.add_trace(go.Scatter(
                x=[mn_p, mx_p], y=[mn_p, mx_p],
                mode="lines", line=dict(color=C_YEL, width=2, dash="dash"),
                name="1:1 line",
            ))
            apply_layout(fig_avp, f"Irradiance: Actual vs Predicted  |  R²={r2_irr:.4f}", height=380)
            fig_avp.update_xaxes(title="Actual (W/m²)")
            fig_avp.update_yaxes(title="Predicted (W/m²)")
            st.plotly_chart(fig_avp, use_container_width=True)

        # Residuals Distribution
        with c2:
            resid = df_day["irradiance_error"].dropna()
            xs_n  = np.linspace(resid.min(), resid.max(), 300)
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(
                x=resid, nbinsx=80, histnorm="probability density",
                marker_color=C_CYAN, opacity=0.4, name="Residuals",
            ))
            fig_res.add_trace(go.Scatter(
                x=xs_n, y=stats.norm.pdf(xs_n, resid.mean(), resid.std()),
                mode="lines", line=dict(color=C_MAG, width=2.5), name="Normal fit",
            ))
            fig_res.add_vline(x=0, line=dict(color=C_YEL, width=1.5, dash="dash"))
            apply_layout(fig_res,
                         f"Irradiance Residuals  |  μ={resid.mean():.2f}  σ={resid.std():.2f}",
                         height=380)
            fig_res.update_xaxes(title="Error (W/m²)")
            fig_res.update_yaxes(title="Density")
            st.plotly_chart(fig_res, use_container_width=True)

        # Daily MAE over time
        daily_mae = df_day["irradiance_abs_error"].resample("D").mean()
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(
            x=daily_mae.index, y=daily_mae.values,
            mode="lines", line=dict(color=C_RED, width=0.7),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.15)", name="Daily MAE",
        ))
        fig_mae.add_trace(go.Scatter(
            x=daily_mae.index, y=daily_mae.rolling(14).mean(),
            mode="lines", line=dict(color=C_ORA, width=2.5), name="14-day MA",
        ))
        apply_layout(fig_mae, "Daily Mean Absolute Irradiance Error", height=300)
        fig_mae.update_xaxes(title="Date")
        fig_mae.update_yaxes(title="MAE (W/m²)")
        st.plotly_chart(fig_mae, use_container_width=True)

        # Q-Q Plot
        c3, c4 = st.columns(2)
        with c3:
            qq_sample = resid.sample(min(2000, len(resid)), random_state=4)
            (osm, osr), (slope_q, intercept_q, r_q) = stats.probplot(qq_sample)
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=list(osm), y=list(osr), mode="markers",
                marker=dict(color=C_CYAN, size=5, opacity=0.4), name="Residuals",
            ))
            fig_qq.add_trace(go.Scatter(
                x=list(osm), y=[slope_q*v + intercept_q for v in osm],
                mode="lines", line=dict(color=C_YEL, width=2.5, dash="dash"),
                name="Normal ref.",
            ))
            apply_layout(fig_qq, f"Q-Q Plot  |  r²={r_q**2:.4f}", height=340)
            st.plotly_chart(fig_qq, use_container_width=True)

        with c4:
            # Temperature error scatter
            samp_te = df_day[["actual_temperature","temp_error"]].dropna().sample(
                min(scatter_n, len(df_day)), random_state=5)
            fig_te = go.Figure(go.Scatter(
                x=samp_te["actual_temperature"], y=samp_te["temp_error"],
                mode="markers",
                marker=dict(color=C_PUR, size=4, opacity=0.25),
                name="Temp residuals",
            ))
            fig_te.add_hline(y=0, line=dict(color=C_YEL, width=1.5, dash="dash"))
            apply_layout(fig_te, "Temperature Residuals vs Actual Temp", height=340)
            fig_te.update_xaxes(title="Actual Temp (°C)")
            fig_te.update_yaxes(title="Error (°C)")
            st.plotly_chart(fig_te, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — ADVANCED
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 🔬 Advanced Analysis")

    # ── ACF ────────────────────────────────────────────────────────────────
    st.markdown("### Autocorrelation (ACF) — power_kw")
    hourly_pwr = df["power_kw"].dropna()
    max_lag = 72
    acf_vals = [hourly_pwr.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    ci_acf   = 1.96 / np.sqrt(len(hourly_pwr))

    fig_acf = go.Figure()
    colors_acf = [C_TEAL if v >= 0 else C_RED for v in acf_vals]
    for lag, val, color in zip(range(1, max_lag + 1), acf_vals, colors_acf):
        fig_acf.add_shape(type="line",
                          x0=lag, y0=0, x1=lag, y1=val,
                          line=dict(color=color, width=3))
    fig_acf.add_trace(go.Scatter(
        x=list(range(1, max_lag + 1)), y=acf_vals,
        mode="markers", marker=dict(color=colors_acf, size=6),
        showlegend=False,
    ))
    fig_acf.add_hline(y=ci_acf,  line=dict(color=C_YEL, width=1.5, dash="dash"), annotation_text="95% CI")
    fig_acf.add_hline(y=-ci_acf, line=dict(color=C_YEL, width=1.5, dash="dash"))
    fig_acf.add_hline(y=0, line=dict(color=SUBTEXT, width=0.8))
    apply_layout(fig_acf, "Autocorrelation Function (ACF) — power_kw | 72h", height=340)
    fig_acf.update_xaxes(title="Lag (hours)")
    fig_acf.update_yaxes(title="ACF")
    st.plotly_chart(fig_acf, use_container_width=True)

    # ── Lag-24 scatter ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        lag_df = pd.DataFrame({
            "t":    hourly_pwr.values[24:],
            "t_24": hourly_pwr.values[:-24],
        }).dropna()
        m_lg, b_lg, r_lg, *_ = stats.linregress(lag_df["t_24"], lag_df["t"])
        xs_lg = np.linspace(lag_df.min().min(), lag_df.max().max(), 100)
        samp_lag = lag_df.sample(min(3000, len(lag_df)), random_state=0)

        fig_lag = go.Figure()
        fig_lag.add_trace(go.Scatter(
            x=samp_lag["t_24"], y=samp_lag["t"],
            mode="markers", marker=dict(color=C_ORA, size=4, opacity=0.15),
            name="Samples",
        ))
        fig_lag.add_trace(go.Scatter(
            x=xs_lg, y=m_lg * xs_lg + b_lg,
            mode="lines", line=dict(color=C_RED, width=2.5, dash="dash"),
            name=f"r = {r_lg:.3f}",
        ))
        apply_layout(fig_lag, "Lag-24 Plot: power(t) vs power(t-24h)", height=360)
        fig_lag.update_xaxes(title="power at t-24 (kW)")
        fig_lag.update_yaxes(title="power at t (kW)")
        st.plotly_chart(fig_lag, use_container_width=True)

    # ── Sensor violin ──────────────────────────────────────────────────────
    with c2:
        sensor_cols = [c for c in df_day.columns if c.startswith("irr_sensor")]
        if sensor_cols:
            fig_vio = go.Figure()
            for i, sc in enumerate(sensor_cols):
                fig_vio.add_trace(go.Violin(
                    y=df_day[sc].dropna(),
                    name=sc.replace("irr_", ""),
                    box_visible=True, meanline_visible=True,
                    fillcolor=hex_to_rgba(PALETTE[i % len(PALETTE)], 0.35),
                    line_color=PALETTE[i % len(PALETTE)],
                ))
            apply_layout(fig_vio, "Sensor Distribution Comparison (Violin)", height=360)
            fig_vio.update_yaxes(title="W/m²")
            st.plotly_chart(fig_vio, use_container_width=True)

    # ── Raw data table ─────────────────────────────────────────────────────
    st.markdown("### 📋 Raw Data Preview")
    preview_df = df.select_dtypes(include=np.number).tail(200).copy()
    st.markdown(_render_gradient_table_html(preview_df), unsafe_allow_html=True)

    # ── Download ───────────────────────────────────────────────────────────
    csv_download = df.reset_index().to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Filtered Dataset (CSV)",
        data=csv_download,
        file_name="solar_filtered.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.8rem'>"
    "☀️ Solar PV Analytics Dashboard · Built with Streamlit · "
    "Dark theme · Plotly interactive charts</p>",
    unsafe_allow_html=True,
)
