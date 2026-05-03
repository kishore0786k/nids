from __future__ import annotations

import json
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import streamlit as st

from backend import nids_engine as engine


API_BASE = "http://127.0.0.1:5000"


def _api_get(path: str, **params):
    query = f"?{urlencode(params)}" if params else ""
    with urlopen(f"{API_BASE}{path}{query}", timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def _backend_get(path: str, **params):
    if path == "/api/overview":
        return engine.overview_data()
    if path == "/api/research":
        return engine.analyse_window(**params)
    if path == "/api/charts":
        return engine.chart_data(**params)
    if path == "/api/single-flow":
        return engine.predict_row(
            params.get("flow_index", params.get("idx", 0)),
            alpha=params.get("alpha", engine.DEFAULT_ALPHA),
            beta=params.get("beta"),
            fusion_mode=params.get("fusion_mode", engine.SYMBOLIC_FUSION_MODE),
            seed=params.get("seed", engine.DEFAULT_SEED),
        )
    if path == "/api/comparison":
        data = engine.analyse_window(
            window_size=params.get("window_size", params.get("n", 200)),
            flow_index=params.get("flow_index", 0),
            alpha=params.get("alpha", engine.DEFAULT_ALPHA),
            beta=params.get("beta"),
            fusion_mode=params.get("fusion_mode", engine.SYMBOLIC_FUSION_MODE),
            seed=params.get("seed", engine.DEFAULT_SEED),
        )
        return {
            "n_samples": data["limit"],
            "base_accuracy": data["window_metrics"]["baseline_mlp"][0],
            "ns_accuracy": data["window_metrics"]["neuro_symbolic"][0],
            "table": [
                {
                    "idx": row["idx"],
                    "true": row["true"],
                    "baseline": row["baseline"],
                    "neuro_symbolic": row["proposed"],
                }
                for row in data["rows"][:50]
            ],
        }
    raise ValueError(f"Unsupported local backend path: {path}")


@st.cache_data(ttl=15, show_spinner=False)
def get_backend_json(path: str, **params):
    try:
        return _api_get(path, **params)
    except (URLError, TimeoutError, OSError):
        return _backend_get(path, **params)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


st.set_page_config(page_title="Neuro-Symbolic IoT NIDS", layout="wide")
st.title("Neuro-Symbolic IoT NIDS")
st.caption("Live backend evidence from model predictions, symbolic rules, and NF-ToN-IoT-V2 test data.")

overview = get_backend_json("/api/overview")
max_index = int(overview["max_index"])
window = st.sidebar.slider("Evaluation window", 100, 2000, 750, step=50)
alpha = st.sidebar.slider("Alpha neural weight", 0.05, 0.95, 0.65, step=0.05)
beta = st.sidebar.number_input("Beta rule weight", 0.0, 0.95, value=round(1.0 - alpha, 2), step=0.05)
fusion_mode = st.sidebar.radio("Fusion mode", ["soft", "hard"], horizontal=True)
seed = st.sidebar.number_input("Seed", 0, 2_147_483_647, value=engine.DEFAULT_SEED, step=1)
page = st.sidebar.selectbox("Mode", ["Overview", "Single-Flow Analysis", "Model Comparison"])
params = {
    "window_size": window,
    "alpha": alpha,
    "beta": beta,
    "fusion_mode": fusion_mode,
    "seed": int(seed),
}

research = get_backend_json("/api/research", **params)
charts = get_backend_json("/api/charts", **params)

if page == "Overview":
    baseline = research["window_metrics"]["baseline_mlp"]
    neuro_symbolic = research["window_metrics"]["neuro_symbolic"]
    lift = float(neuro_symbolic[0]) - float(baseline[0])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Neuro-symbolic F1", pct(neuro_symbolic[3]))
    col2.metric("Neuro-symbolic accuracy", pct(neuro_symbolic[0]), delta=f"{lift * 100:.2f} pts")
    col3.metric("Changed predictions", research["rule_analytics"]["changed_predictions"])
    col4.metric("Rule triggers", research["rule_analytics"]["rule_trigger_count"])

    metric_df = pd.DataFrame(
        {
            "Metric": research["window_metrics"]["labels"],
            "Baseline MLP": baseline,
            "Neuro-symbolic": neuro_symbolic,
        }
    )
    st.dataframe(metric_df, use_container_width=True)

    counts = pd.DataFrame(
        {
            "Class": research["class_distribution"]["labels"],
            "Samples": research["class_distribution"]["values"],
        }
    )
    st.plotly_chart(px.bar(counts, x="Class", y="Samples"), use_container_width=True)
    if "baseline_values" in research["class_distribution"]:
        dist = pd.DataFrame(
            {
                "Class": research["class_distribution"]["labels"],
                "Baseline": research["class_distribution"]["baseline_values"],
                "Proposed": research["class_distribution"]["proposed_values"],
            }
        ).melt(id_vars="Class", var_name="Model", value_name="Samples")
        st.plotly_chart(px.bar(dist, x="Class", y="Samples", color="Model", barmode="group"), use_container_width=True)

    curve = pd.DataFrame(
        {
            "Window": charts["improvement_curve"]["labels"],
            "Baseline MLP": charts["improvement_curve"]["existing_accuracy"],
            "Neuro-symbolic": charts["improvement_curve"]["proposed_accuracy"],
        }
    ).melt(id_vars="Window", var_name="Model", value_name="Accuracy")
    st.plotly_chart(px.line(curve, x="Window", y="Accuracy", color="Model", markers=True), use_container_width=True)

elif page == "Single-Flow Analysis":
    idx = st.number_input("Test index", 0, max_index, 0, step=1)
    flow = get_backend_json("/api/single-flow", **{**params, "flow_index": idx, "idx": idx})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**True label:** `{flow['true_label']}`")
        st.markdown(f"**Baseline MLP prediction:** `{flow['base_pred']}`")
        st.markdown(f"**Neuro-symbolic final label:** `{flow['ns_label']}`")
        st.markdown(f"**Changed prediction:** `{flow['changed_prediction']}`")
        st.markdown(f"**Rule strength:** `{flow['rule_strength']}`")
        st.markdown(f"**Confidence:** `{pct(flow['confidence'])}`")
        st.markdown(f"**Defense action:** {flow['defense']['action']}")
        st.dataframe(pd.DataFrame(flow["fired_rules"]), use_container_width=True)

    with col2:
        prob_df = pd.DataFrame({"Class": flow["probabilities"]["labels"], "Probability": flow["probabilities"]["values"]})
        st.plotly_chart(px.bar(prob_df, x="Class", y="Probability"), use_container_width=True)

    with st.expander("Feature values"):
        st.dataframe(pd.DataFrame(flow["features"].items(), columns=["Feature", "Value"]), use_container_width=True)

else:
    n_samples = st.slider("Number of test samples", 50, min(2000, max_index + 1), 200, step=50)
    comparison = get_backend_json("/api/comparison", **{**params, "window_size": n_samples, "n": n_samples})
    st.metric("Baseline MLP accuracy", pct(comparison["base_accuracy"]))
    st.metric("Neuro-symbolic accuracy", pct(comparison["ns_accuracy"]))
    st.dataframe(pd.DataFrame(comparison["table"]), use_container_width=True)

    acc_df = pd.DataFrame(
        {"Model": ["Baseline MLP", "Neuro-symbolic"], "Accuracy": [comparison["base_accuracy"], comparison["ns_accuracy"]]}
    )
    st.plotly_chart(px.bar(acc_df, x="Model", y="Accuracy", text="Accuracy"), use_container_width=True)
