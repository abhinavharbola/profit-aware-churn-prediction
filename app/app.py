import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COST_OF_OFFER, INTERVENTION_SUCCESS_RATE, MONTHS_REVENUE_SAVED, MODELS_DIR, PROCESSED_DIR
from src.evaluation.explainability import get_tree_explainer, compute_shap_explanation
from src.evaluation.profit_optimizer import compute_expected_profit, compute_avg_monthly_spend

st.set_page_config(page_title="Churn Ledger", layout="wide", initial_sidebar_state="expanded")

BG = "#FBF7EE"
PANEL = "#FFFDF8"
PANEL_RAISED = "#F3ECDC"
INK = "#26241D"
INK_SOFT = "#75705F"
LINE = "#E2D9C3"
PROFIT = "#1B7A4D"
PROFIT_SOFT = "#E3F0E7"
LOSS = "#A8432C"
LOSS_SOFT = "#F9E9E1"

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {INK};
}}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {BG};
}}

#MainMenu, footer, header {{ visibility: hidden; }}
div[data-testid="stToolbar"] {{ visibility: hidden; }}

.block-container {{
    padding-top: 2rem;
    max-width: 1180px;
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    color: {INK};
}}

p, span, div, label, li {{
    color: {INK};
}}

.header-block {{
    text-align: center;
    max-width: 720px;
    margin: 0 auto 1rem auto;
}}

.ledger-title {{
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 2.6rem;
    margin-bottom: 0.3rem;
    letter-spacing: -0.01em;
    color: {INK};
}}

.ledger-subtitle {{
    color: {INK_SOFT};
    font-size: 1.02rem;
    line-height: 1.55;
    margin-bottom: 0.4rem;
}}

.ledger-rule {{
    border: none;
    border-top: 1px solid {LINE};
    margin: 1.1rem 0 1.6rem 0;
}}

.section-label {{
    font-size: 0.82rem;
    color: {INK_SOFT} !important;
    border-bottom: 1px solid {LINE};
    padding-bottom: 0.35rem;
    margin-bottom: 0.9rem;
}}

.stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background-color: {LINE};
    border: 1px solid {LINE};
    margin-bottom: 1.4rem;
}}

.stat-card {{
    background-color: {PANEL};
    padding: 1rem 1.1rem;
}}

.stat-card .label {{
    font-size: 0.78rem;
    color: {INK_SOFT} !important;
    margin-bottom: 0.3rem;
}}

.stat-card .value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: {INK};
}}

.stat-card.profit .value {{ color: {PROFIT} !important; }}
.stat-card.loss .value {{ color: {LOSS} !important; }}

.decision-badge {{
    display: inline-block;
    padding: 0.35rem 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    border: 1px solid;
}}

.decision-badge.intervene {{
    color: {PROFIT} !important;
    background-color: {PROFIT_SOFT};
    border-color: {PROFIT};
}}

.decision-badge.hold {{
    color: {LOSS} !important;
    background-color: {LOSS_SOFT};
    border-color: {LOSS};
}}

.ledger-panel {{
    background-color: {PANEL};
    border: 1px solid {LINE};
    padding: 1.2rem 1.3rem;
    margin-bottom: 1.2rem;
    color: {INK};
}}

.breakdown-row {{
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px dashed {LINE};
    font-size: 0.92rem;
    color: {INK};
}}

.breakdown-row:last-child {{
    border-bottom: none;
    padding-top: 0.6rem;
    font-weight: 600;
}}

.breakdown-row .amount {{
    font-family: 'IBM Plex Mono', monospace;
}}

.formula-panel {{
    background-color: {PANEL};
    border: 1px solid {LINE};
    border-left: 3px solid {PROFIT};
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: {INK};
}}

.insight-line {{
    border-left: 3px solid {PROFIT};
    background-color: {PROFIT_SOFT};
    padding: 0.7rem 1rem;
    font-size: 0.95rem;
    margin: 0.6rem 0 1.2rem 0;
    color: {INK};
}}

div[data-baseweb="tab-list"] {{
    display: flex;
    width: 100%;
    gap: 0;
    border-bottom: 1px solid {LINE};
}}

button[data-baseweb="tab"] {{
    flex: 1 1 0;
    display: flex;
    justify-content: center;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.02rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: {INK_SOFT} !important;
    padding: 0.5rem 0 0.8rem 0;
    background-color: transparent;
}}

button[data-baseweb="tab"] p {{
    color: {INK_SOFT} !important;
    font-size: 1.02rem;
    font-weight: 500;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PROFIT} !important;
    font-weight: 600;
}}

button[data-baseweb="tab"][aria-selected="true"] p {{
    color: {PROFIT} !important;
    font-weight: 600;
}}

div[data-baseweb="tab-highlight"] {{
    background-color: {PROFIT} !important;
    height: 2px !important;
}}

.stButton>button, .stDownloadButton>button {{
    background-color: {PROFIT};
    color: #FFFFFF !important;
    border-radius: 0;
    border: none;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}}

.stButton>button *, .stDownloadButton>button * {{ color: #FFFFFF !important; }}

.stButton>button:hover, .stDownloadButton>button:hover {{
    background-color: {INK};
    color: #FFFFFF !important;
}}

section[data-testid="stSidebar"] {{
    background-color: {PANEL_RAISED};
    border-right: 1px solid {LINE};
    color: {INK};
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem;
}}

.sidebar-heading {{
    font-family: 'Source Serif 4', serif;
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.9rem;
    color: {PROFIT} !important;
}}

.sidebar-row {{
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    padding: 0.35rem 0;
    border-bottom: 1px dashed {LINE};
    color: {INK} !important;
}}

.sidebar-row .val {{
    font-family: 'IBM Plex Mono', monospace;
    color: {INK} !important;
}}

.legend-swatch {{
    display: inline-block;
    width: 0.7rem;
    height: 0.7rem;
    margin-right: 0.4rem;
    vertical-align: middle;
}}

code {{
    background-color: {PROFIT_SOFT} !important;
    color: {PROFIT} !important;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {LINE};
}}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
CALIBRATOR_PATH = os.path.join(MODELS_DIR, "calibrator.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
CALIBRATION_METHOD_PATH = os.path.join(MODELS_DIR, "calibration_method.pkl")
FEATURE_MATRIX_PATH = os.path.join(PROCESSED_DIR, "feature_matrix.pkl")

FEATURE_LABELS = {
    "recency": "Recency (days)",
    "frequency": "Frequency",
    "monetary_total": "Monetary total",
    "monetary_avg": "Monetary avg",
    "unique_products": "Unique products",
    "spend_30d": "Spend, 30d",
    "spend_90d": "Spend, 90d",
    "interpurchase_mean": "Interpurchase mean",
    "interpurchase_std": "Interpurchase std",
    "spend_trend": "Spend trend",
    "product_diversity": "Product diversity",
    "seasonal_dropoff": "Recent drop-off (90d)",
}


@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, CALIBRATOR_PATH, FEATURE_NAMES_PATH, CALIBRATION_METHOD_PATH]):
        st.error("Model artifacts not found. Run 'python scripts/run_pipeline.py' first.")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator_obj = pickle.load(f)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    with open(CALIBRATION_METHOD_PATH, "rb") as f:
        calibration_method = pickle.load(f)

    def calibrate(scores):
        if calibration_method == "isotonic":
            return calibrator_obj.transform(scores)
        return calibrator_obj.predict_proba(np.array(scores).reshape(-1, 1))[:, 1]

    return model, calibrate, feature_names


@st.cache_data
def load_feature_matrix():
    if not os.path.exists(FEATURE_MATRIX_PATH):
        return None
    return pd.read_pickle(FEATURE_MATRIX_PATH)


def stat_card(label, value, kind=""):
    css_class = f"stat-card {kind}".strip()
    return f'<div class="{css_class}"><div class="label">{label}</div><div class="value">{value}</div></div>'


def decision_badge(is_intervene):
    if is_intervene:
        return '<span class="decision-badge intervene">INTERVENE</span>'
    return '<span class="decision-badge hold">DO NOT INTERVENE</span>'


def render_threshold_chart(threshold_df, optimal_threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=threshold_df["threshold"],
        y=threshold_df["net_profit"],
        mode="lines",
        line=dict(color=PROFIT, width=2),
        fill="tozeroy",
        fillcolor="rgba(211,169,75,0.15)",
        name="Net profit"
    ))
    optimal_row = threshold_df[threshold_df["threshold"] == optimal_threshold]
    if not optimal_row.empty:
        fig.add_trace(go.Scatter(
            x=optimal_row["threshold"],
            y=optimal_row["net_profit"],
            mode="markers",
            marker=dict(color=PROFIT, size=10, line=dict(color=INK, width=1)),
            name="Optimal threshold"
        ))
    fig.update_layout(
        plot_bgcolor=PANEL,
        paper_bgcolor=PANEL,
        font=dict(family="IBM Plex Sans", color=INK, size=13),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(title="Threshold", gridcolor=LINE, zeroline=False, color=INK),
        yaxis=dict(title="Net profit (£)", gridcolor=LINE, zeroline=True, zerolinecolor=LINE, color=INK),
        height=340
    )
    return fig


def style_shap_figure(fig):
    fig.patch.set_facecolor(PANEL)
    for ax in fig.axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        for spine in ax.spines.values():
            spine.set_color(LINE)
        for text_obj in ax.texts:
            text_obj.set_color(INK)
    return fig


st.markdown(
    '<div class="header-block">'
    '<div class="ledger-title">Customer Churn, Profit Ledger</div>'
    '<div class="ledger-subtitle">Calibrated churn probabilities, priced against intervention cost and '
    'retention value, to decide who is worth a retention offer and who is not.</div>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-heading">Assumptions</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-row"><span>Intervention cost</span><span class="val">£{COST_OF_OFFER:.2f}</span></div>
        <div class="sidebar-row"><span>Success rate</span><span class="val">{INTERVENTION_SUCCESS_RATE:.0%}</span></div>
        <div class="sidebar-row"><span>Revenue horizon</span><span class="val">{MONTHS_REVENUE_SAVED} mo</span></div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-heading" style="margin-top:1.6rem;">Reading the decision</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:0.85rem; line-height:1.6; color:{INK_SOFT};">
        <span class="legend-swatch" style="background-color:{PROFIT};"></span>Intervene — expected profit is positive.<br>
        <span class="legend-swatch" style="background-color:{LOSS};"></span>Do not intervene — expected profit is negative.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-heading" style="margin-top:1.6rem;">Change these</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.85rem; color:{INK_SOFT}; line-height:1.5;">'
        f'Edit <code>config.py</code> and re-run the pipeline to reflect different business assumptions.</div>',
        unsafe_allow_html=True
    )

model, calibrate, feature_names = load_artifacts()

tab1, tab2, tab3, tab4 = st.tabs(["Single prediction", "Batch analysis", "Model info", "Batch export"])

with tab1:
    input_mode = st.radio("Input mode", ["Manual feature entry", "Customer ID lookup"], horizontal=True)

    if input_mode == "Manual feature entry":
        st.markdown('<div class="section-label">RFM features</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.slider("Recency (days)", 0, 365, 30)
            frequency = st.slider("Frequency", 1, 100, 5)
            monetary_total = st.number_input("Monetary total (£)", 0.0, 50000.0, 500.0, step=50.0)
        with col2:
            monetary_avg = st.number_input("Avg revenue per line item (£)", 0.0, 5000.0, 100.0, step=10.0)
            unique_products = st.slider("Unique products", 1, 200, 10)
            spend_30d = st.number_input("Spend, 30 days (£)", 0.0, 20000.0, 200.0, step=50.0)
        with col3:
            spend_90d = st.number_input("Spend, 90 days (£)", 0.0, 30000.0, 600.0, step=50.0)
            interpurchase_mean = st.number_input("Avg days between purchases", 0.0, 365.0, 30.0)
            interpurchase_std = st.number_input("Std days between purchases", 0.0, 200.0, 15.0)

        col4, col5 = st.columns(2)
        with col4:
            spend_trend = st.number_input("Spend trend (slope)", -500.0, 500.0, 0.0, step=10.0)
        with col5:
            product_diversity = st.slider("Product diversity", 0.0, 1.0, 0.5)
            seasonal_dropoff = st.selectbox("Recent drop-off (91-180d ago, quiet since)", [0, 1])

        manual_inputs = {
            "recency": recency,
            "frequency": frequency,
            "monetary_total": monetary_total,
            "monetary_avg": monetary_avg,
            "unique_products": unique_products,
            "spend_30d": spend_30d,
            "spend_90d": spend_90d,
            "interpurchase_mean": interpurchase_mean,
            "interpurchase_std": interpurchase_std,
            "spend_trend": spend_trend,
            "product_diversity": product_diversity,
            "seasonal_dropoff": seasonal_dropoff,
        }
        missing_features = [f for f in feature_names if f not in manual_inputs]
        if missing_features:
            st.error(f"Manual entry form is missing input for: {', '.join(missing_features)}")
            st.stop()
        input_values = np.array([[manual_inputs[f] for f in feature_names]])

        avg_monthly_spend_for_profit = compute_avg_monthly_spend(monetary_total)

    else:
        st.markdown('<div class="section-label">Customer ID lookup</div>', unsafe_allow_html=True)

        feature_df = load_feature_matrix()
        if feature_df is None:
            st.error("Feature matrix not found. Run 'python scripts/run_pipeline.py' first.")
            st.stop()

        available_ids = sorted(feature_df["customer_id"].unique())
        customer_id_input = st.selectbox("Select customer ID", available_ids)

        customer_data = feature_df[feature_df["customer_id"] == customer_id_input]
        if customer_data.empty:
            st.error("Customer ID not found in feature matrix.")
            st.stop()

        latest_window = customer_data.sort_values("obs_end").iloc[-1]
        st.caption(f"Most recent observation window ending {latest_window['obs_end'].strftime('%Y-%m-%d')}")

        input_values = np.array([[latest_window[col] for col in feature_names]])

        monetary_total_val = latest_window["monetary_total"]
        avg_monthly_spend_for_profit = compute_avg_monthly_spend(monetary_total_val)

    if st.button("Predict churn probability", type="primary"):
        input_df = pd.DataFrame(input_values, columns=feature_names)

        raw_prob = model.predict_proba(input_df)[:, 1][0]
        calibrated_prob = calibrate(np.array([raw_prob]))[0]
        if isinstance(calibrated_prob, np.ndarray):
            calibrated_prob = float(calibrated_prob)

        expected_profit = compute_expected_profit(calibrated_prob, avg_monthly_spend_for_profit)
        revenue_3m = avg_monthly_spend_for_profit * MONTHS_REVENUE_SAVED
        is_intervene = expected_profit > 0

        st.markdown(
            f"""
            <div class="stat-grid">
                {stat_card("Churn probability", f"{calibrated_prob:.1%}")}
                {stat_card("Expected profit", f"£{expected_profit:,.2f}", "profit" if is_intervene else "loss")}
                <div class="stat-card"><div class="label">Decision</div><div class="value">{decision_badge(is_intervene)}</div></div>
                {stat_card("Revenue at stake", f"£{revenue_3m:,.2f}")}
            </div>
            """,
            unsafe_allow_html=True
        )

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown('<div class="section-label">Financial breakdown</div>', unsafe_allow_html=True)
            breakdown_rows = [
                ("Avg monthly spend", f"£{avg_monthly_spend_for_profit:,.2f}"),
                ("Revenue at stake, 3 months", f"£{revenue_3m:,.2f}"),
                ("Intervention success rate", f"{INTERVENTION_SUCCESS_RATE:.0%}"),
                ("Expected revenue saved", f"£{calibrated_prob * INTERVENTION_SUCCESS_RATE * revenue_3m:,.2f}"),
                ("Intervention cost", f"£{COST_OF_OFFER:,.2f}"),
                ("Expected profit", f"£{expected_profit:,.2f}"),
            ]
            rows_html = "".join(
                f'<div class="breakdown-row"><span>{label}</span><span class="amount">{value}</span></div>'
                for label, value in breakdown_rows
            )
            st.markdown(f'<div class="ledger-panel">{rows_html}</div>', unsafe_allow_html=True)

        with detail_col2:
            st.markdown('<div class="section-label">SHAP explanation</div>', unsafe_allow_html=True)
            try:
                explainer = get_tree_explainer(model)
                display_names = [FEATURE_LABELS.get(f, f) for f in feature_names]
                explanation = compute_shap_explanation(explainer, input_df, display_names)

                mpl.rcParams["font.family"] = "sans-serif"
                mpl.rcParams["text.color"] = INK
                mpl.rcParams["axes.labelcolor"] = INK
                mpl.rcParams["axes.edgecolor"] = LINE
                mpl.rcParams["xtick.color"] = INK
                mpl.rcParams["ytick.color"] = INK
                fig, ax = plt.subplots(figsize=(5, 3.2))
                import shap
                shap.waterfall_plot(explanation, show=False)
                fig = plt.gcf()
                style_shap_figure(fig)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

with tab2:
    COMPARISON_PATH = os.path.join(PROCESSED_DIR, "profit_comparison.csv")
    THRESHOLD_PATH = os.path.join(PROCESSED_DIR, "threshold_analysis.csv")

    if os.path.exists(COMPARISON_PATH):
        comparison_df = pd.read_csv(COMPARISON_PATH)

        st.markdown('<div class="section-label">Baseline comparison</div>', unsafe_allow_html=True)
        st.dataframe(comparison_df, hide_index=True, width='stretch')

        default_profit = float(
            comparison_df[comparison_df["Strategy"] == "Default Threshold (0.5)"]["Net Campaign Profit"]
            .str.replace("£", "", regex=False).str.replace(",", "", regex=False).values[0]
        )
        optimal_profit = float(
            comparison_df[comparison_df["Strategy"] == "Profit-Optimized"]["Net Campaign Profit"]
            .str.replace("£", "", regex=False).str.replace(",", "", regex=False).values[0]
        )
        profit_delta = optimal_profit - default_profit

        if default_profit > 0:
            lift_pct = (profit_delta / default_profit) * 100
            insight_text = (
                f"The profit-optimized threshold lifts net profit by {lift_pct:.1f}% "
                f"over the default 0.5 cutoff."
            )
        else:
            insight_text = (
                f"The profit-optimized threshold changes net profit by £{profit_delta:,.0f} "
                f"versus the default 0.5 cutoff (percentage lift isn't meaningful when the "
                f"baseline is zero or negative)."
            )

        st.markdown(
            f'<div class="insight-line">{insight_text}</div>',
            unsafe_allow_html=True
        )

        if os.path.exists(THRESHOLD_PATH):
            threshold_df = pd.read_csv(THRESHOLD_PATH)
            optimal_threshold = threshold_df.loc[threshold_df["net_profit"].idxmax(), "threshold"]

            st.markdown('<div class="section-label">Threshold sweep</div>', unsafe_allow_html=True)
            st.plotly_chart(render_threshold_chart(threshold_df, optimal_threshold), width='stretch')
            st.caption("Net profit at each candidate threshold. The marker sits at the argmax.")
    else:
        st.info("Profit comparison data not found. Run 'python scripts/run_pipeline.py' first.")

with tab3:
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown('<div class="section-label">Feature descriptions</div>', unsafe_allow_html=True)
        feature_descriptions = {
            "recency": "Days since last purchase",
            "frequency": "Unique invoices in window",
            "monetary_total": "Total revenue in window",
            "monetary_avg": "Avg revenue per line item (not per order)",
            "unique_products": "Distinct products purchased",
            "spend_30d": "Spend in last 30 days",
            "spend_90d": "Spend in last 90 days",
            "interpurchase_mean": "Avg days between purchases",
            "interpurchase_std": "Std days between purchases",
            "spend_trend": "Slope of monthly spend",
            "product_diversity": "Unique products / total orders",
            "seasonal_dropoff": "Active 91-180d ago, inactive last 90d",
        }
        rows_html = "".join(
            f'<div class="breakdown-row"><span>{feat}</span><span style="color:{INK_SOFT};">{desc}</span></div>'
            for feat, desc in feature_descriptions.items()
        )
        st.markdown(f'<div class="ledger-panel">{rows_html}</div>', unsafe_allow_html=True)

    with info_col2:
        st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="ledger-panel" style="font-size:0.92rem; line-height:1.7;">
            Model: XGBoost, natural class imbalance<br>
            Split: customer-grouped, no window leakage<br>
            Calibration: isotonic regression<br>
            Features: 12 RFM-based<br>
            Window: 12-month observation, 90-day prediction<br>
            Tuning: 50 Optuna trials on PR-AUC
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">Configurable parameters</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="ledger-panel" style="font-size:0.92rem; line-height:1.7;">
            Intervention cost: £{COST_OF_OFFER}<br>
            Success rate: {INTERVENTION_SUCCESS_RATE:.0%}<br>
            Revenue horizon: {MONTHS_REVENUE_SAVED} months
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="section-label" style="margin-top:0.6rem;">Profit formula</div>', unsafe_allow_html=True)
        st.markdown(
            f'''
            <div class="formula-panel">
                <span style="font-family:'IBM Plex Mono', monospace; font-size:1.05rem; color:{INK};">
                    E[&Delta;Profit] = p<sub>i</sub> &middot; (&gamma; &middot; V<sub>i</sub>) &minus; C
                </span>
            </div>
            ''',
            unsafe_allow_html=True
        )
        st.caption("p: calibrated churn probability · γ: success rate · V: avg monthly spend × 3 · C: cost")

with tab4:
    feature_df = load_feature_matrix()
    if feature_df is None:
        st.error("Feature matrix not found. Run 'python scripts/run_pipeline.py' first.")
        st.stop()

    THRESHOLD_PATH = os.path.join(PROCESSED_DIR, "threshold_analysis.csv")
    if not os.path.exists(THRESHOLD_PATH):
        st.error("Threshold analysis not found. Run 'python scripts/run_pipeline.py' first.")
        st.stop()

    st.markdown(
        '<div class="insight-line">Each customer is scored on their own expected profit, not a single '
        'population-level threshold. INTERVENE means the expected gain for that specific customer exceeds '
        'the intervention cost.</div>',
        unsafe_allow_html=True
    )

    if st.button("Score all customers", type="primary"):
        with st.spinner("Scoring customers..."):
            latest_windows = feature_df.sort_values("obs_end").groupby("customer_id").last().reset_index()

            X_export = latest_windows[feature_names].copy()
            raw_probs = model.predict_proba(X_export)[:, 1]
            calibrated_probs = calibrate(raw_probs)
            if isinstance(calibrated_probs, np.ndarray):
                calibrated_probs = calibrated_probs.flatten()

            latest_windows["calibrated_churn_prob"] = calibrated_probs
            latest_windows["avg_monthly_spend"] = compute_avg_monthly_spend(latest_windows["monetary_total"])
            latest_windows["expected_profit"] = latest_windows.apply(
                lambda row: compute_expected_profit(row["calibrated_churn_prob"], row["avg_monthly_spend"]),
                axis=1
            )
            latest_windows["intervention_decision"] = latest_windows["expected_profit"].apply(
                lambda x: "INTERVENE" if x > 0 else "DO NOT INTERVENE"
            )
            latest_windows["revenue_at_stake"] = latest_windows["avg_monthly_spend"] * MONTHS_REVENUE_SAVED

            export_columns = [
                "customer_id", "calibrated_churn_prob", "avg_monthly_spend",
                "revenue_at_stake", "expected_profit", "intervention_decision",
                "recency", "frequency", "monetary_total", "obs_end"
            ]
            available_export_cols = [c for c in export_columns if c in latest_windows.columns]
            results_df = latest_windows[available_export_cols].copy()
            results_df = results_df.sort_values("expected_profit", ascending=False)

            intervene_df = results_df[results_df["intervention_decision"] == "INTERVENE"]
            do_not_df = results_df[results_df["intervention_decision"] == "DO NOT INTERVENE"]

            st.markdown(
                f"""
                <div class="stat-grid">
                    {stat_card("Customers scored", f"{len(results_df):,}")}
                    {stat_card("Intervene", f"{len(intervene_df):,}", "profit")}
                    {stat_card("Do not intervene", f"{len(do_not_df):,}", "loss")}
                    {stat_card("Total expected profit", f"£{intervene_df['expected_profit'].sum():,.0f}", "profit")}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-label">1 · Intervention list, top 50 by expected profit</div>', unsafe_allow_html=True)
            st.dataframe(
                intervene_df.head(50).style.format({
                    "calibrated_churn_prob": "{:.3f}",
                    "avg_monthly_spend": "£{:,.2f}",
                    "revenue_at_stake": "£{:,.2f}",
                    "expected_profit": "£{:,.2f}",
                    "monetary_total": "£{:,.2f}"
                }),
                width='stretch'
            )

            st.markdown('<div class="section-label">2 · Export</div>', unsafe_allow_html=True)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="Download intervention list (CSV)",
                    data=intervene_df.to_csv(index=False),
                    file_name="intervention_list.csv",
                    mime="text/csv"
                )
            with dl_col2:
                st.download_button(
                    label="Download full results (CSV)",
                    data=results_df.to_csv(index=False),
                    file_name="all_customers_scored.csv",
                    mime="text/csv"
                )
