"""
Generate IEEE-style publication artifacts from backend evaluation results.

Outputs:
- results/publication_package/publication_package.json
- results/publication_package/publication_summary.md
- results/publication_package/submission_readiness.md
- paper/generated/*.tex
- paper/figures/generated/*.png and *.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend import nids_engine as engine
from src.project_paths import PAPER_DIR, RESULTS_DIR as ROOT_RESULTS_DIR


RESULTS_DIR = ROOT_RESULTS_DIR / "publication_package"
PAPER_GEN = PAPER_DIR / "generated"
FIG_DIR = PAPER_DIR / "figures" / "generated"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(name: str, fig) -> dict:
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.tight_layout()
    for path in (png, pdf):
        if path.exists():
            path.unlink()
    fig.savefig(png, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def generate_figures(charts: dict) -> list[dict]:
    figures = []
    baseline_color = "#687989"
    proposed_color = "#087f8c"
    accent_color = "#c94f3d"
    grid_color = "#d7e0e6"
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    x = [int(v) for v in charts["improvement_curve"]["labels"]]
    ax.plot(x, charts["improvement_curve"]["existing_accuracy"], marker="o", linewidth=1.8, color=baseline_color, label="Baseline MLP")
    ax.plot(x, charts["improvement_curve"]["proposed_accuracy"], marker="s", linewidth=2.4, color=proposed_color, label="Neuro-symbolic")
    ax.set_xlabel("Evaluation window size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy Improvement Curve")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7)
    ax.legend(frameon=False)
    figures.append({"id": "figure_01_accuracy_improvement", **save_plot("figure_01_accuracy_improvement", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    labels = charts["per_class"]["labels"]
    idx = np.arange(len(labels))
    ax.plot(idx, charts["per_class"]["existing_f1"], marker="o", linewidth=1.8, color=baseline_color, label="Baseline MLP")
    ax.plot(idx, charts["per_class"]["proposed_f1"], marker="s", linewidth=2.4, color=proposed_color, label="Neuro-symbolic")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.set_title("Per-Class F1 Comparison")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7)
    ax.legend(frameon=False)
    figures.append({"id": "figure_02_per_class_f1", **save_plot("figure_02_per_class_f1", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(charts["confidence_histogram"]["labels"], charts["confidence_histogram"]["values"], marker="o", linewidth=2.2, color=proposed_color)
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Flow count")
    ax.set_title("Prediction Confidence Distribution")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7)
    ax.tick_params(axis="x", rotation=30)
    figures.append({"id": "figure_03_confidence_distribution", **save_plot("figure_03_confidence_distribution", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    labels = charts["class_error_rate"]["labels"]
    idx = np.arange(len(labels))
    width = 0.36
    ax.bar(idx - width / 2, charts["class_error_rate"].get("baseline_values", charts["class_error_rate"]["values"]), width, color=baseline_color, label="Baseline")
    ax.bar(idx + width / 2, charts["class_error_rate"].get("proposed_values", charts["class_error_rate"]["values"]), width, color=proposed_color, label="Proposed")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 1)
    ax.set_title("Class Error Rate")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7)
    ax.legend(frameon=False)
    figures.append({"id": "figure_04_class_error_rate", **save_plot("figure_04_class_error_rate", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    roc = charts["roc_curve"]
    ax.plot([p["x"] for p in roc["baseline"]["points"]], [p["y"] for p in roc["baseline"]["points"]], "--", linewidth=1.8, color=baseline_color, label=f"Baseline AUC={roc['baseline']['auc']}")
    ax.plot([p["x"] for p in roc["proposed"]["points"]], [p["y"] for p in roc["proposed"]["points"]], linewidth=2.5, color=proposed_color, label=f"Proposed AUC={roc['proposed']['auc']}")
    ax.plot([0, 1], [0, 1], ":", color="#9aa7b1", label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Micro-Average ROC Curve")
    ax.grid(True, color=grid_color, linewidth=0.7)
    ax.legend(frameon=False)
    figures.append({"id": "figure_05_roc_curve", **save_plot("figure_05_roc_curve", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    pr = charts["pr_curve"]
    ax.plot([p["x"] for p in pr["baseline"]["points"]], [p["y"] for p in pr["baseline"]["points"]], "--", linewidth=1.8, color=baseline_color, label=f"Baseline AP={pr['baseline']['average_precision']}")
    ax.plot([p["x"] for p in pr["proposed"]["points"]], [p["y"] for p in pr["proposed"]["points"]], linewidth=2.5, color=proposed_color, label=f"Proposed AP={pr['proposed']['average_precision']}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Micro-Average Precision-Recall Curve")
    ax.set_ylim(0, 1)
    ax.grid(True, color=grid_color, linewidth=0.7)
    ax.legend(frameon=False)
    figures.append({"id": "figure_06_precision_recall_curve", **save_plot("figure_06_precision_recall_curve", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.bar(charts["detection_counts"]["labels"], charts["detection_counts"]["values"], color=[baseline_color, baseline_color, proposed_color, "#228b5b", accent_color])
    ax.set_ylabel("Flow count")
    ax.set_title("Attack Detection and Containment Coverage")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7)
    ax.tick_params(axis="x", rotation=15)
    figures.append({"id": "figure_07_detection_coverage", **save_plot("figure_07_detection_coverage", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    gain = charts["attack_recall_gain"]
    idx = np.arange(len(gain["labels"]))
    width = 0.36
    ax.bar(idx - width / 2, gain["baseline"], width, color=baseline_color, label="Baseline")
    ax.bar(idx + width / 2, gain["proposed"], width, color=proposed_color, label="Proposed")
    ax.set_xticks(idx)
    ax.set_xticklabels(gain["labels"], rotation=25, ha="right")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.set_title("Attack-Wise Recall Gain")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7)
    ax.legend(frameon=False)
    figures.append({"id": "figure_08_attack_recall_gain", **save_plot("figure_08_attack_recall_gain", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    unknown = charts["unknown_attack_detection"]
    ax.bar(unknown["labels"], unknown["values"], color=[baseline_color, accent_color])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Attack review rate")
    ax.set_title("UNKNOWN Attack Detection")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7)
    figures.append({"id": "figure_09_unknown_attack_detection", **save_plot("figure_09_unknown_attack_detection", fig)})

    fig, ax1 = plt.subplots(figsize=(6.3, 4.0))
    latency = charts["latency_comparison"]
    throughput = charts["throughput_comparison"]
    x = np.arange(len(latency["labels"]))
    ax1.bar(x - 0.18, latency["values"], 0.36, color=accent_color, label="Latency (ms)")
    ax1.set_ylabel("Latency (ms)")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, throughput["values"], 0.36, color=proposed_color, label="Throughput (flows/s)")
    ax2.set_ylabel("Throughput (flows/s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(latency["labels"])
    ax1.set_title("Detection Latency and Throughput")
    ax1.grid(True, axis="y", color=grid_color, linewidth=0.7)
    figures.append({"id": "figure_10_latency_throughput", **save_plot("figure_10_latency_throughput", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    rules = charts["rule_trigger_analysis"]
    idx = np.arange(len(rules["labels"]))
    width = 0.36
    ax.bar(idx - width / 2, rules["triggered"], width, color="#c98211", label="Triggered")
    ax.bar(idx + width / 2, rules["applied"], width, color=proposed_color, label="Applied")
    ax.set_xticks(idx)
    ax.set_xticklabels(rules["labels"], rotation=25, ha="right")
    ax.set_ylabel("Flow count")
    ax.set_title("Symbolic Rule Trigger Analysis")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7)
    ax.legend(frameon=False)
    figures.append({"id": "figure_11_rule_trigger_analysis", **save_plot("figure_11_rule_trigger_analysis", fig)})

    return figures


def write_tex_files(overview: dict, charts: dict, backend: dict, ablation: dict, novelty: dict) -> None:
    existing_acc = charts["metric_comparison"]["existing"][0]
    proposed_acc = charts["metric_comparison"]["proposed"][0]
    f1 = charts["metric_comparison"]["proposed"][3]
    lift = proposed_acc - existing_acc

    (PAPER_GEN / "abstract_text.tex").write_text(
        (
            "This work presents a neuro-symbolic network intrusion detection system "
            "for NF-ToN-IoT-V2 traffic. The proposed backend combines neural "
            "classification with symbolic rule traces and an adaptive defence "
            f"workflow, achieving {proposed_acc:.4f} accuracy and {f1:.4f} macro-F1 "
            f"against a live baseline-MLP accuracy of {existing_acc:.4f}."
        ),
        encoding="utf-8",
    )

    (PAPER_GEN / "setup_table.tex").write_text(
        "\\begin{tabular}{ll}\n"
        "\\hline\n"
        "Item & Value \\\\\n"
        "\\hline\n"
        f"Dataset & NF-ToN-IoT-V2 \\\\\n"
        f"Test rows & {backend['test_rows']} \\\\\n"
        f"Features & {backend['feature_count']} \\\\\n"
        f"Classes & {len(backend['classes'])} \\\\\n"
        "Classifier & MLP + symbolic rules \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )

    (PAPER_GEN / "comparison_table.tex").write_text(
        "\\begin{tabular}{lrrrr}\n"
        "\\hline\n"
        "System & Accuracy & Precision & Recall & Macro-F1 \\\\\n"
        "\\hline\n"
        f"Baseline MLP & {charts['metric_comparison']['existing'][0]:.4f} & {charts['metric_comparison']['existing'][1]:.4f} & {charts['metric_comparison']['existing'][2]:.4f} & {charts['metric_comparison']['existing'][3]:.4f} \\\\\n"
        f"Neuro-symbolic & {charts['metric_comparison']['proposed'][0]:.4f} & {charts['metric_comparison']['proposed'][1]:.4f} & {charts['metric_comparison']['proposed'][2]:.4f} & {charts['metric_comparison']['proposed'][3]:.4f} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )

    baseline = ablation["systems"][0]["metrics"]
    neuro_symbolic = ablation["systems"][-1]["metrics"]
    (PAPER_GEN / "ablation_table.tex").write_text(
        "\\begin{tabular}{lrrrr}\n"
        "\\hline\n"
        "Variant & Accuracy & Precision & Recall & Macro-F1 \\\\\n"
        "\\hline\n"
        f"Baseline MLP & {baseline[0]:.4f} & {baseline[1]:.4f} & {baseline[2]:.4f} & {baseline[3]:.4f} \\\\\n"
        f"MLP + Symbolic Rules & {neuro_symbolic[0]:.4f} & {neuro_symbolic[1]:.4f} & {neuro_symbolic[2]:.4f} & {neuro_symbolic[3]:.4f} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )

    (PAPER_GEN / "reliability_table.tex").write_text(
        "\\begin{tabular}{lr}\n"
        "\\hline\n"
        "Reliability signal & Value \\\\\n"
        "\\hline\n"
        f"Expected calibration error & {novelty['calibration']['ece']:.4f} \\\\\n"
        f"Conformal target coverage & {novelty['conformal']['target_coverage']:.4f} \\\\\n"
        f"Empirical conformal coverage & {novelty['conformal']['empirical_coverage']:.4f} \\\\\n"
        f"Average prediction-set size & {novelty['conformal']['average_set_size']:.4f} \\\\\n"
        f"OOD rate & {novelty['ood_drift']['ood_rate']:.4f} \\\\\n"
        f"High-uncertainty flows & {novelty['uncertainty']['high_uncertainty_count']} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )

    (PAPER_GEN / "results_paragraph.tex").write_text(
        (
            f"The proposed NS-NIDS improves accuracy by {lift * 100:.2f} percentage "
            "points over the existing system while preserving class-wise inspection "
            "through confusion-matrix, ROC, confidence, and rule-trace evidence. "
            f"The reliability layer reports ECE={novelty['calibration']['ece']:.4f}, "
            f"conformal coverage={novelty['conformal']['empirical_coverage']:.4f}, "
            f"and OOD rate={novelty['ood_drift']['ood_rate']:.4f}. "
            "The generated artifacts are reproducible from backend model and test data."
        ),
        encoding="utf-8",
    )

    (PAPER_GEN / "algorithm_box.tex").write_text(
        "\\begin{enumerate}\n"
        "\\item Load normalized NF-ToN-IoT-V2 flow features.\n"
        "\\item Predict class probabilities with the neural classifier.\n"
        "\\item Apply symbolic security rules to refine the label.\n"
        "\\item Estimate uncertainty, conformal set size, and OOD score.\n"
        "\\item Generate an incident and containment recommendation for attack labels.\n"
        "\\item Export metrics, figures, and evidence tables for publication.\n"
        "\\end{enumerate}\n",
        encoding="utf-8",
    )


def build_package(
    limit: int | None = None,
    alpha: float = engine.DEFAULT_ALPHA,
    beta: float | None = None,
    fusion_mode: str = engine.SYMBOLIC_FUSION_MODE,
    seed: int = engine.DEFAULT_SEED,
    flow_index: int = 0,
) -> dict:
    ensure_dirs()
    backend = engine.backend_status()
    evaluation_limit = backend["test_rows"] if limit is None else limit
    overview = engine.overview_data(
        window_size=evaluation_limit,
        flow_index=flow_index,
        alpha=alpha,
        beta=beta,
        fusion_mode=fusion_mode,
        seed=seed,
    )
    params = {
        "window_size": evaluation_limit,
        "flow_index": flow_index,
        "alpha": alpha,
        "beta": beta,
        "fusion_mode": fusion_mode,
        "seed": seed,
    }
    charts = engine.chart_data(**params)
    ablation = engine.ablation_data(**params)
    novelty = engine.novelty_data(
        evaluation_limit,
        min(0.40, max(0.01, alpha)),
        flow_index=flow_index,
        seed=seed,
        beta=beta,
        fusion_mode=fusion_mode,
    )
    figures = generate_figures(charts)
    write_tex_files(overview, charts, backend, ablation, novelty)

    package = {
        "project": "Neuro-Symbolic NIDS",
        "dataset": "NF-ToN-IoT-V2",
        "protocol": charts.get("parameters", params),
        "backend": backend,
        "overview": overview,
        "charts": charts,
        "ablation": ablation,
        "novelty": novelty,
        "rule_analytics": charts.get("rule_analytics", {}),
        "evidence_policy": {
            "live_evaluation": "figures and tables are generated from backend recomputation on model predictions and test labels",
            "paper_summary": "saved-paper-summary values are retained only under charts.paper_summary for traceability",
        },
        "figures": figures,
        "readiness": {
            "software_package": True,
            "backend_separated": True,
            "frontend_separated": True,
            "publication_figures_generated": True,
            "needs_manual_review": [
                "Confirm baseline definitions against cited literature.",
                "Add final bibliography with verified IEEE references.",
                "Document train/test split and hyperparameters in manuscript.",
            ],
        },
    }

    (RESULTS_DIR / "publication_package.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    (RESULTS_DIR / "publication_summary.md").write_text(
        "# Publication Package Summary\n\n"
        f"- Backend: {backend['backend']}\n"
        f"- Test rows: {backend['test_rows']}\n"
        f"- Feature count: {backend['feature_count']}\n"
        f"- Baseline MLP accuracy: {charts['metric_comparison']['existing'][0]:.4f}\n"
        f"- Neuro-symbolic accuracy: {charts['metric_comparison']['proposed'][0]:.4f}\n"
        f"- Changed predictions: {charts.get('rule_analytics', {}).get('changed_predictions', 0)}\n"
        f"- Calibration ECE: {novelty['calibration']['ece']:.4f}\n"
        f"- Conformal coverage: {novelty['conformal']['empirical_coverage']:.4f}\n"
        f"- OOD rate: {novelty['ood_drift']['ood_rate']:.4f}\n"
        f"- Figures generated: {len(figures)}\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "submission_readiness.md").write_text(
        "# Submission Readiness\n\n"
        "This package is now structured like a research software artifact. "
        "Before IEEE submission, manually verify baselines, references, and manuscript claims.\n",
        encoding="utf-8",
    )
    return package


if __name__ == "__main__":
    pkg = build_package()
    print(f"Generated publication package with {len(pkg['figures'])} figures.")
    print(RESULTS_DIR)
