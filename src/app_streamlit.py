import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# allow importing neuro_symbolic from project root
sys.path.append('..')

from neuro_symbolic import apply_symbolic_rules  # function in neuro_symbolic.py

MODEL_PATH = '../models/ns_nids_model.pkl'
ROBUST_PATH = '../models/robust_nsnids.pkl'
TEST_PATH = '../data/test_processed.csv'


@st.cache_resource
def load_models():
    base_model = joblib.load(MODEL_PATH)
    robust_model = joblib.load(ROBUST_PATH) if os.path.exists(ROBUST_PATH) else None
    return base_model, robust_model


@st.cache_data
def load_test_data():
    df = pd.read_csv(TEST_PATH)
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y


st.set_page_config(page_title="Neuro‑Symbolic IoT NIDS", layout="wide")
st.title("🛡️ Neuro‑Symbolic IoT NIDS – NF‑ToN‑IoT‑V2")
st.caption("Explainable Neuro‑Symbolic Intrusion Detection on NF‑ToN‑IoT‑V2 NetFlow features.")

base_model, robust_model = load_models()
X, y = load_test_data()
classes = base_model.classes_

page = st.sidebar.selectbox(
    "Mode",
    ["📊 Overview", "🔍 Single‑Flow Analysis", "⚡ Model Comparison"],
)

# ================= OVERVIEW ================= #

if page == "📊 Overview":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Macro F1 (NS)", "98.1%")
    with col2:
        st.metric("Accuracy", "94.2%")
    with col3:
        st.metric("Attack Classes", str(y.nunique()))
    with col4:
        st.metric("Dataset", "NF‑ToN‑IoT‑V2")

    st.markdown("### Baseline vs Neuro‑Symbolic (Macro Scores)")
    data = {
        "Model": ["Baseline MLP", "Neuro‑Symbolic"],
        "Accuracy": [0.900, 0.942],
        "Macro‑F1": [0.875, 0.981],
    }
    df_cmp = pd.DataFrame(data)
    st.dataframe(df_cmp.style.format({"Accuracy": "{:.3f}", "Macro‑F1": "{:.3f}"}))

    st.markdown("### Class distribution on test set")
    class_counts = y.value_counts().sort_index()
    df_counts = pd.DataFrame({"Class": class_counts.index, "Samples": class_counts.values})
    fig_cls = px.bar(
        df_counts,
        x="Class",
        y="Samples",
    )
    fig_cls.update_traces(width=0.6)
    fig_cls.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_cls, use_container_width=True)

    st.markdown("### Accuracy comparison (bar chart)")
    acc_df_over = pd.DataFrame(
        {"Model": ["Baseline MLP", "Neuro‑Symbolic"], "Accuracy": [0.900, 0.942]}
    )
    fig_acc_over = px.bar(
        acc_df_over,
        x="Model",
        y="Accuracy",
        text=acc_df_over["Accuracy"].map(lambda v: f"{v:.3f}"),
    )
    fig_acc_over.update_traces(width=0.3)
    fig_acc_over.update_yaxes(range=[0, 1.0])
    fig_acc_over.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_acc_over, use_container_width=True)

    st.markdown("### Notes")
    st.markdown(
        "- Baseline: plain MLP on NF‑ToN‑IoT‑V2 flows.\n"
        "- Neuro‑Symbolic: MLP + symbolic rules on rate/duration features.\n"
        "- Metrics shown are from your held‑out test split."
    )

# ================= SINGLE FLOW ================= #

elif page == "🔍 Single‑Flow Analysis":
    st.subheader("Inspect one NetFlow record")
    idx = st.number_input("Test index", 0, len(X) - 1, 0, step=1)

    sample = X.iloc[idx]
    true_label = y.iloc[idx]
    sample_array = sample.values.reshape(1, -1)

    # baseline neural prediction
    base_probs = base_model.predict_proba(sample_array)[0]
    base_pred = classes[np.argmax(base_probs)]

    # neuro‑symbolic refinement
    ns_label, fired_rules = apply_symbolic_rules(sample, base_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**True label:** `{true_label}`")
        st.markdown(f"**Baseline MLP prediction:** `{base_pred}`")
        st.markdown(f"**Neuro‑Symbolic final label:** `🛡 {ns_label}`")
        if robust_model is not None:
            robust_pred = robust_model.predict(sample_array)[0]
            st.markdown(f"**Robust MLP prediction:** `{robust_pred}`")

        st.markdown("#### Symbolic explanation")
        if fired_rules and fired_rules[0].get("ruleid", "NONE") != "NONE":
            for r in fired_rules:
                rid = r.get("ruleid", "R?")
                reason = r.get("reason", "")
                st.markdown(f"- **{rid}** – {reason}")
        else:
            st.write("No rule fired; final label equals neural prediction.")

    with col2:
        prob_df = pd.DataFrame({"Class": classes, "Baseline probability": base_probs})
        fig_prob = px.bar(
            prob_df,
            x="Class",
            y="Baseline probability",
        )
        fig_prob.update_traces(width=0.6)
        fig_prob.update_yaxes(range=[0, 1.0])
        fig_prob.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)

    with st.expander("View feature values"):
        st.dataframe(sample.to_frame(name="value"))

# ================= MODEL COMPARISON ================= #

elif page == "⚡ Model Comparison":
    st.subheader("Compare baseline vs neuro‑symbolic on multiple samples")

    n_samples = st.slider(
        "Number of test samples",
        min_value=50,
        max_value=min(500, len(X)),
        value=200,
        step=50,
    )

    subset_X = X.head(n_samples)
    subset_y = y.head(n_samples)

    # baseline predictions
    base_preds = base_model.predict(subset_X)

    # neuro‑symbolic predictions
    ns_preds = []
    for i in range(len(subset_X)):
        p = base_preds[i]
        ns_label, _ = apply_symbolic_rules(subset_X.iloc[i], p)
        ns_preds.append(ns_label)

    df_comp = pd.DataFrame(
        {
            "True": subset_y.values,
            "Baseline": base_preds,
            "Neuro‑Symbolic": ns_preds,
        }
    )

    st.markdown("### Sample‑wise prediction table (first 50 rows)")
    st.dataframe(df_comp.head(50))

    st.markdown("### Accuracy comparison on selected subset")
    base_acc = (df_comp["Baseline"] == df_comp["True"]).mean()
    ns_acc = (df_comp["Neuro‑Symbolic"] == df_comp["True"]).mean()
    acc_df = pd.DataFrame(
        {"Model": ["Baseline", "Neuro‑Symbolic"], "Accuracy": [base_acc, ns_acc]}
    )

    fig_acc = px.bar(
        acc_df,
        x="Model",
        y="Accuracy",
        text=acc_df["Accuracy"].map(lambda v: f"{v:.3f}"),
    )
    fig_acc.update_traces(width=0.3)      # thinner, less wide bars
    fig_acc.update_yaxes(range=[0, 1.0])
    fig_acc.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_acc, use_container_width=True)
