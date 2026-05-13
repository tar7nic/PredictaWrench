import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

MODEL_PATH   = os.path.join(MODELS_DIR, "xgb_rul_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.csv")
TRAIN_PATH   = os.path.join(DATA_DIR,   "train_features.csv")
TEST_PATH    = os.path.join(DATA_DIR,   "test_features.csv")

NON_FEATURE_COLS = {"unit", "cycle", "RUL"}

# ── Theme constants ───────────────────────────────────────────────────────────
BG_PRIMARY   = "#0d1117"
BG_SECONDARY = "#161b22"
BG_CARD      = "#1c2128"
BG_CARD2     = "#21262d"
ACCENT       = "#f97316"
ACCENT_DIM   = "#c2611a"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"
BORDER       = "#30363d"
CRITICAL_CLR = "#ef4444"
WARNING_CLR  = "#eab308"
HEALTHY_CLR  = "#22c55e"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT_PRIMARY, size=11),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictaWrench",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Bebas+Neue&display=swap');

html, body, [class*="css"] {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: 'IBM Plex Mono', monospace;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {BG_SECONDARY} !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT_PRIMARY} !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}

/* Hide default header */
header[data-testid="stHeader"] {{ background: transparent; }}

/* Main area */
.main .block-container {{
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}}

/* Remove Streamlit branding */
#MainMenu, footer {{ visibility: hidden; }}

/* Metric cards */
[data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 1rem 1.2rem;
}}
[data-testid="stMetric"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}
[data-testid="stMetricValue"] {{
    color: {ACCENT} !important;
    font-size: 1.6rem !important;
    font-weight: 600;
}}
[data-testid="stMetricDelta"] {{ font-size: 0.75rem !important; }}

/* Selectbox */
div[data-baseweb="select"] > div {{
    background-color: {BG_CARD} !important;
    border-color: {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: {TEXT_MUTED} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 2px solid transparent !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}
div[data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {BORDER};
    gap: 0.5rem;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 6px; }}

/* Divider */
hr {{ border-color: {BORDER}; margin: 1rem 0; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {TEXT_MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ── Helper: custom card ───────────────────────────────────────────────────────
def card(content_html, padding="1.2rem 1.4rem", border_left=None):
    border_style = f"border-left: 3px solid {border_left};" if border_left else ""
    st.markdown(f"""
    <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:6px;
                padding:{padding};margin-bottom:0.8rem;{border_style}">
        {content_html}
    </div>""", unsafe_allow_html=True)


def section_header(title, subtitle=""):
    sub_html = f'<p style="color:{TEXT_MUTED};font-size:0.75rem;margin:0.2rem 0 0 0;letter-spacing:0.05em">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:1.2rem;padding-bottom:0.6rem;border-bottom:1px solid {BORDER}">
        <h2 style="font-family:\'Bebas Neue\',sans-serif;color:{ACCENT};
                   font-size:1.8rem;margin:0;letter-spacing:0.08em;line-height:1">{title}</h2>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def risk_badge(rul):
    if rul < 30:
        return f'<span style="background:{CRITICAL_CLR}22;color:{CRITICAL_CLR};border:1px solid {CRITICAL_CLR}55;padding:2px 10px;border-radius:4px;font-size:0.7rem;font-weight:600">CRITICAL</span>'
    elif rul < 80:
        return f'<span style="background:{WARNING_CLR}22;color:{WARNING_CLR};border:1px solid {WARNING_CLR}55;padding:2px 10px;border-radius:4px;font-size:0.7rem;font-weight:600">WARNING</span>'
    else:
        return f'<span style="background:{HEALTHY_CLR}22;color:{HEALTHY_CLR};border:1px solid {HEALTHY_CLR}55;padding:2px 10px;border-radius:4px;font-size:0.7rem;font-weight:600">HEALTHY</span>'


# ── Data & model loading ──────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH).iloc[0]


def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


@st.cache_data
def build_fleet_table(_model, _scaler, test_df):
    last_cycles = test_df.loc[test_df.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
    feat_cols   = get_feature_columns(last_cycles)
    X           = _scaler.transform(last_cycles[feat_cols].values)
    preds       = np.clip(_model.predict(X), 0, 125)
    last_cycles = last_cycles.copy()
    last_cycles["pred_RUL"] = preds.round(1)
    last_cycles["health_score"] = (last_cycles["pred_RUL"] / 125 * 100).clip(0, 100).round(1)
    return last_cycles[["unit", "cycle", "pred_RUL", "health_score", "RUL"]].reset_index(drop=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.5rem 0 1.5rem 0;border-bottom:1px solid {BORDER};margin-bottom:1.2rem">
        <p style="font-family:\'Bebas Neue\',sans-serif;font-size:1.9rem;
                  color:{ACCENT};margin:0;letter-spacing:0.1em;line-height:1">PREDICTA<br>WRENCH</p>
        <p style="color:{TEXT_MUTED};font-size:0.65rem;margin:0.3rem 0 0 0;
                  letter-spacing:0.12em;text-transform:uppercase">Engine Health Monitor</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Fleet Overview", "Engine Deep Dive", "Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style="color:{TEXT_MUTED};font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase">
    Dataset<br>
    <span style="color:{TEXT_PRIMARY}">NASA CMAPSS FD001</span></p>
    <p style="color:{TEXT_MUTED};font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;margin-top:0.6rem">
    Model<br>
    <span style="color:{TEXT_PRIMARY}">XGBoost Regressor</span></p>
    <p style="color:{TEXT_MUTED};font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;margin-top:0.6rem">
    Target<br>
    <span style="color:{TEXT_PRIMARY}">Remaining Useful Life</span></p>
    """, unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
model, scaler = load_model_and_scaler()
train_df, test_df = load_data()
metrics = load_metrics()
fleet   = build_fleet_table(model, scaler, test_df)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Fleet Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "Fleet Overview":
    section_header("FLEET OVERVIEW", "Real-time predictive health status across all monitored engines")

    n_total    = len(fleet)
    n_critical = (fleet["pred_RUL"] < 30).sum()
    n_warning  = ((fleet["pred_RUL"] >= 30) & (fleet["pred_RUL"] < 80)).sum()
    n_healthy  = (fleet["pred_RUL"] >= 80).sum()
    avg_rul    = fleet["pred_RUL"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Engines",    n_total)
    c2.metric("Critical",         n_critical, delta=f"{n_critical/n_total*100:.0f}% of fleet", delta_color="inverse")
    c3.metric("Warning",          n_warning,  delta=f"{n_warning/n_total*100:.0f}% of fleet",  delta_color="off")
    c4.metric("Healthy",          n_healthy,  delta=f"{n_healthy/n_total*100:.0f}% of fleet")
    c5.metric("Avg Predicted RUL", f"{avg_rul:.1f}", delta="cycles remaining")

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # Risk distribution donut
    col_chart, col_table = st.columns([1, 2], gap="large")

    with col_chart:
        st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Risk Distribution</p>", unsafe_allow_html=True)
        donut = go.Figure(go.Pie(
            labels=["Critical", "Warning", "Healthy"],
            values=[n_critical, n_warning, n_healthy],
            hole=0.62,
            marker=dict(colors=[CRITICAL_CLR, WARNING_CLR, HEALTHY_CLR],
                        line=dict(color=BG_CARD, width=2)),
            textinfo="percent",
            textfont=dict(family="IBM Plex Mono", size=11, color=TEXT_PRIMARY),
            hovertemplate="%{label}: %{value} engines<extra></extra>",
        ))
        donut.add_annotation(
            text=f"<b>{n_total}</b><br><span style='font-size:10px'>ENGINES</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="IBM Plex Mono", size=14, color=TEXT_PRIMARY),
            align="center"
        )
        donut.update_layout(**PLOTLY_LAYOUT, showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                                        xanchor="center", x=0.5,
                                        font=dict(size=10, color=TEXT_MUTED)),
                            height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(donut, use_container_width=True)

        # RUL histogram
        st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>RUL Distribution</p>", unsafe_allow_html=True)
        hist = go.Figure(go.Histogram(
            x=fleet["pred_RUL"],
            nbinsx=20,
            marker=dict(color=ACCENT, opacity=0.85, line=dict(color=BG_CARD, width=0.5)),
            hovertemplate="RUL %{x:.0f}: %{y} engines<extra></extra>",
        ))
        hist.add_vline(x=30, line_dash="dash", line_color=CRITICAL_CLR, line_width=1.2,
                       annotation_text="Critical", annotation_font=dict(color=CRITICAL_CLR, size=9))
        hist.add_vline(x=80, line_dash="dash", line_color=WARNING_CLR, line_width=1.2,
                       annotation_text="Warning",  annotation_font=dict(color=WARNING_CLR,  size=9))
        hist.update_layout(**PLOTLY_LAYOUT, height=220,
                           xaxis_title="Predicted RUL (cycles)",
                           yaxis_title="Engine Count",
                           margin=dict(l=40, r=10, t=10, b=40))
        st.plotly_chart(hist, use_container_width=True)

    with col_table:
        st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Engine Status Table</p>", unsafe_allow_html=True)

        sort_order = {"CRITICAL": 0, "WARNING": 1, "HEALTHY": 2}

        def risk_label(rul):
            if rul < 30:   return "CRITICAL"
            elif rul < 80: return "WARNING"
            else:          return "HEALTHY"

        display = fleet.copy()
        display["Risk"]         = display["pred_RUL"].apply(risk_label)
        display["sort_key"]     = display["Risk"].map(sort_order)
        display = display.sort_values(["sort_key", "pred_RUL"]).drop(columns="sort_key")
        display["Health Score"] = display["health_score"].apply(lambda x: f"{x:.1f}%")
        display["Pred RUL"]     = display["pred_RUL"].apply(lambda x: f"{x:.1f}")
        display["Actual RUL"]   = display["RUL"].apply(lambda x: f"{x:.0f}")

        display_final = display[["unit", "cycle", "Pred RUL", "Actual RUL", "Health Score", "Risk"]].rename(
            columns={"unit": "Engine", "cycle": "Last Cycle"}
        )

        def color_risk(val):
            if val == "CRITICAL": return f"color: {CRITICAL_CLR}; font-weight: 600"
            elif val == "WARNING": return f"color: {WARNING_CLR}; font-weight: 600"
            else: return f"color: {HEALTHY_CLR}; font-weight: 600"

        styled = display_final.style\
            .applymap(color_risk, subset=["Risk"])\
            .set_properties(**{
                "background-color": BG_CARD,
                "color": TEXT_PRIMARY,
                "font-family": "IBM Plex Mono, monospace",
                "font-size": "12px",
                "border-color": BORDER,
            })\
            .set_table_styles([{
                "selector": "th",
                "props": [
                    ("background-color", BG_CARD2),
                    ("color", TEXT_MUTED),
                    ("font-size", "11px"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "0.06em"),
                    ("border-color", BORDER),
                ]
            }])

        st.dataframe(styled, use_container_width=True, height=530)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Engine Deep Dive
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Engine Deep Dive":
    section_header("ENGINE DEEP DIVE", "Per-unit sensor trends, degradation trajectory, and health score")

    engine_ids = sorted(test_df["unit"].unique())
    selected   = st.selectbox("Select Engine Unit", engine_ids,
                              format_func=lambda x: f"Engine #{x:03d}")

    engine_test  = test_df[test_df["unit"] == selected].sort_values("cycle")
    engine_train = train_df[train_df["unit"] == (selected % train_df["unit"].max() + 1)].sort_values("cycle")

    feat_cols = get_feature_columns(engine_test)
    X_engine  = scaler.transform(engine_test[feat_cols].values)
    preds     = np.clip(model.predict(X_engine), 0, 125)

    current_rul    = preds[-1]
    actual_rul     = engine_test["RUL"].iloc[-1]
    health_score   = min(current_rul / 125 * 100, 100)
    last_cycle     = engine_test["cycle"].max()

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Engine Unit",       f"#{selected:03d}")
    m2.metric("Predicted RUL",     f"{current_rul:.1f} cycles")
    m3.metric("Health Score",      f"{health_score:.1f}%")
    m4.metric("Last Observed Cycle", last_cycle)

    risk_html = risk_badge(current_rul)
    card(f"""
    <div style="display:flex;align-items:center;gap:1rem">
        <span style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em">Risk Status</span>
        {risk_html}
        <span style="color:{TEXT_MUTED};font-size:0.72rem;margin-left:auto">
            Actual RUL at last cycle: <span style="color:{TEXT_PRIMARY}">{actual_rul:.0f}</span>
        </span>
    </div>
    """, border_left=CRITICAL_CLR if current_rul < 30 else (WARNING_CLR if current_rul < 80 else HEALTHY_CLR))

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # Health gauge
    col_gauge, col_rul = st.columns([1, 2], gap="large")

    with col_gauge:
        st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem'>Health Score Gauge</p>", unsafe_allow_html=True)
        gauge_color = CRITICAL_CLR if health_score < 24 else (WARNING_CLR if health_score < 64 else HEALTHY_CLR)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number=dict(suffix="%", font=dict(family="IBM Plex Mono", size=26, color=gauge_color)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color=TEXT_MUTED, size=9),
                          tickwidth=1, tickcolor=BORDER),
                bar=dict(color=gauge_color, thickness=0.22),
                bgcolor=BG_CARD2,
                borderwidth=0,
                steps=[
                    dict(range=[0, 24],   color="rgba(239,68,68,0.09)"),
                    dict(range=[24, 64],  color="rgba(234,179,8,0.09)"),
                    dict(range=[64, 100], color="rgba(34,197,94,0.09)"),
                ],
                threshold=dict(line=dict(color=gauge_color, width=2), thickness=0.7, value=health_score)
            ),
        ))
        gauge.update_layout(**PLOTLY_LAYOUT, height=240, margin=dict(l=30, r=30, t=20, b=10))
        st.plotly_chart(gauge, use_container_width=True)

    with col_rul:
        st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem'>Predicted vs Actual RUL Over Cycles</p>", unsafe_allow_html=True)
        rul_fig = go.Figure()
        rul_fig.add_trace(go.Scatter(
            x=engine_test["cycle"], y=engine_test["RUL"],
            name="Actual RUL", mode="lines",
            line=dict(color=TEXT_MUTED, width=1.5, dash="dot"),
        ))
        rul_fig.add_trace(go.Scatter(
            x=engine_test["cycle"], y=preds,
            name="Predicted RUL", mode="lines",
            line=dict(color=ACCENT, width=2),
            fill="tonexty", fillcolor="rgba(249,115,22,0.09)",
        ))
        rul_fig.add_hrect(y0=0, y1=30, fillcolor="rgba(239,68,68,0.08)",
                          line_width=0, annotation_text="Critical Zone",
                          annotation_font=dict(color=CRITICAL_CLR, size=9))
        rul_fig.update_layout(**PLOTLY_LAYOUT, height=240,
                              xaxis_title="Cycle", yaxis_title="RUL (cycles)",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                         font=dict(size=10, color=TEXT_MUTED)))
        st.plotly_chart(rul_fig, use_container_width=True)

    # Sensor trends
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Sensor Trends (Raw Readings)</p>", unsafe_allow_html=True)

    base_sensors = [c for c in engine_test.columns
                    if c.startswith("s") and not c.endswith(("_roll_mean", "_roll_std"))
                    and c not in NON_FEATURE_COLS and c != "RUL"]

    if base_sensors:
        rows   = (len(base_sensors) + 2) // 3
        sensor_fig = make_subplots(rows=rows, cols=3,
                                   subplot_titles=base_sensors,
                                   vertical_spacing=0.12,
                                   horizontal_spacing=0.06)
        for i, sensor in enumerate(base_sensors):
            r, c = divmod(i, 3)
            sensor_fig.add_trace(
                go.Scatter(x=engine_test["cycle"], y=engine_test[sensor],
                           mode="lines", line=dict(color=ACCENT, width=1.2),
                           name=sensor, showlegend=False,
                           hovertemplate=f"{sensor}: %{{y:.3f}}<extra></extra>"),
                row=r+1, col=c+1
            )
            roll_col = f"{sensor}_roll_mean"
            if roll_col in engine_test.columns:
                sensor_fig.add_trace(
                    go.Scatter(x=engine_test["cycle"], y=engine_test[roll_col],
                               mode="lines", line=dict(color="#60a5fa", width=1, dash="dot"),
                               name=f"{sensor} mean", showlegend=False,
                               hovertemplate=f"roll mean: %{{y:.3f}}<extra></extra>"),
                    row=r+1, col=c+1
                )

        sensor_fig.update_annotations(font=dict(color=TEXT_MUTED, size=9, family="IBM Plex Mono"))
        sensor_fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_MUTED, size=8))
        sensor_fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_MUTED, size=8))
        sensor_fig.update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                                  font=dict(family="IBM Plex Mono", color=TEXT_PRIMARY),
                                  height=120 * rows + 60,
                                  margin=dict(l=40, r=20, t=40, b=20))
        st.plotly_chart(sensor_fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    section_header("MODEL PERFORMANCE", "XGBoost RUL regression — evaluation on NASA CMAPSS FD001 test set")

    rmse = metrics["RMSE"]
    mae  = metrics["MAE"]
    r2   = metrics["R2"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE",         f"{rmse:.4f}", delta="Target < 20 ✓" if rmse < 20 else f"Target < 20")
    m2.metric("MAE",          f"{mae:.4f}")
    m3.metric("R² Score",     f"{r2:.4f}",  delta="78.5% variance explained")
    m4.metric("Eval Protocol","Last Cycle",  delta="per engine · CMAPSS standard")

    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

    col_scatter, col_fi = st.columns(2, gap="large")

    # Predicted vs Actual scatter
    with col_scatter:
        st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem'>Predicted vs Actual RUL</p>", unsafe_allow_html=True)

        last_test = test_df.loc[test_df.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
        feat_cols = get_feature_columns(last_test)
        X_last    = scaler.transform(last_test[feat_cols].values)
        y_pred    = np.clip(model.predict(X_last), 0, 125)
        y_actual  = last_test["RUL"].values

        scatter = go.Figure()
        scatter.add_trace(go.Scatter(
            x=y_actual, y=y_pred,
            mode="markers",
            marker=dict(color=ACCENT, size=7, opacity=0.75,
                        line=dict(color=BG_CARD, width=0.5)),
            hovertemplate="Actual: %{x:.0f}<br>Predicted: %{y:.1f}<extra></extra>",
            name="Engines"
        ))
        lim = max(y_actual.max(), y_pred.max()) + 5
        scatter.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim],
            mode="lines", line=dict(color=TEXT_MUTED, dash="dash", width=1.2),
            name="Perfect fit", hoverinfo="skip"
        ))
        scatter.update_layout(**PLOTLY_LAYOUT, height=360,
                              xaxis_title="Actual RUL (cycles)",
                              yaxis_title="Predicted RUL (cycles)",
                              legend=dict(font=dict(size=10, color=TEXT_MUTED)))
        st.plotly_chart(scatter, use_container_width=True)

    # Feature importance
    with col_fi:
        st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem'>Top 20 Feature Importances</p>", unsafe_allow_html=True)

        fi_vals  = model.feature_importances_
        fi_names = get_feature_columns(test_df)
        fi_df    = pd.DataFrame({"feature": fi_names, "importance": fi_vals})
        fi_df    = fi_df.sort_values("importance", ascending=True).tail(20)

        fi_fig = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker=dict(
                color=fi_df["importance"],
                colorscale=[[0, ACCENT_DIM], [1, ACCENT]],
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fi_fig.update_layout(**PLOTLY_LAYOUT, height=360,
                             xaxis_title="Importance Score",
                             margin=dict(l=160, r=20, t=10, b=40))
        st.plotly_chart(fi_fig, use_container_width=True)

    # Residuals
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem'>Residuals Distribution</p>", unsafe_allow_html=True)

    residuals = y_pred - y_actual
    res_fig = go.Figure()
    res_fig.add_trace(go.Histogram(
        x=residuals, nbinsx=25,
        marker=dict(color=ACCENT, opacity=0.8, line=dict(color=BG_CARD, width=0.5)),
        name="Residuals",
        hovertemplate="Error %{x:.1f}: %{y} engines<extra></extra>",
    ))
    res_fig.add_vline(x=0, line_dash="dash", line_color=TEXT_MUTED, line_width=1.2)
    res_fig.update_layout(**PLOTLY_LAYOUT, height=240,
                          xaxis_title="Prediction Error (cycles)",
                          yaxis_title="Engine Count",
                          margin=dict(l=50, r=20, t=10, b=40))
    st.plotly_chart(res_fig, use_container_width=True)

    # Model config summary
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    card(f"""
    <p style="color:{TEXT_MUTED};font-size:0.68rem;text-transform:uppercase;
              letter-spacing:0.1em;margin:0 0 0.7rem 0">Model Configuration</p>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem">
        <div><span style="color:{TEXT_MUTED};font-size:0.68rem">Algorithm</span><br>
             <span style="color:{TEXT_PRIMARY};font-size:0.82rem">XGBoost Regressor</span></div>
        <div><span style="color:{TEXT_MUTED};font-size:0.68rem">Estimators</span><br>
             <span style="color:{TEXT_PRIMARY};font-size:0.82rem">800</span></div>
        <div><span style="color:{TEXT_MUTED};font-size:0.68rem">Max Depth</span><br>
             <span style="color:{TEXT_PRIMARY};font-size:0.82rem">6</span></div>
        <div><span style="color:{TEXT_MUTED};font-size:0.68rem">Learning Rate</span><br>
             <span style="color:{TEXT_PRIMARY};font-size:0.82rem">0.05</span></div>
        <div><span style="color:{TEXT_MUTED};font-size:0.68rem">RUL Clip</span><br>
             <span style="color:{TEXT_PRIMARY};font-size:0.82rem">125 cycles</span></div>
        <div><span style="color:{TEXT_MUTED};font-size:0.68rem">Rolling Window</span><br>
             <span style="color:{TEXT_PRIMARY};font-size:0.82rem">5 cycles</span></div>
        <div><span style="color:{TEXT_MUTED};font-size:0.68rem">Dropped Sensors</span><br>
             <span style="color:{TEXT_PRIMARY};font-size:0.82rem">7 (low variance)</span></div>
        <div><span style="color:{TEXT_MUTED};font-size:0.68rem">Eval Protocol</span><br>
             <span style="color:{TEXT_PRIMARY};font-size:0.82rem">Last cycle / engine</span></div>
    </div>
    """)