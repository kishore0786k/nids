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
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def generate_figures(charts: dict) -> list[dict]:
    figures = []

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    x = [int(v) for v in charts["improvement_curve"]["labels"]]
    ax.plot(x, charts["improvement_curve"]["existing_accuracy"], marker="o", label="Baseline MLP")
    ax.plot(x, charts["improvement_curve"]["proposed_accuracy"], marker="s", label="Neuro-symbolic")
    ax.set_xlabel("Evaluation window size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Improvement Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    figures.append({"id": "figure_01_accuracy_improvement", **save_plot("figure_01_accuracy_improvement", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    labels = charts["per_class"]["labels"]
    idx = np.arange(len(labels))
    ax.plot(idx, charts["per_class"]["existing_f1"], marker="o", label="Baseline MLP")
    ax.plot(idx, charts["per_class"]["proposed_f1"], marker="s", label="Neuro-symbolic")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("Per-Class F1 Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    figures.append({"id": "figure_02_per_class_f1", **save_plot("figure_02_per_class_f1", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(charts["confidence_histogram"]["labels"], charts["confidence_histogram"]["values"], marker="o")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Flow count")
    ax.set_title("Prediction Confidence Distribution")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    figures.append({"id": "figure_03_confidence_distribution", **save_plot("figure_03_confidence_distribution", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.bar(charts["class_error_rate"]["labels"], charts["class_error_rate"]["values"])
    ax.set_ylabel("Error rate")
    ax.set_title("Class Error Rate")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    figures.append({"id": "figure_04_class_error_rate", **save_plot("figure_04_class_error_rate", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    roc = charts["roc_curve"]
    ax.plot([p["x"] for p in roc["points"]], [p["y"] for p in roc["points"]], label=f"AUC={roc['auc']}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Micro-Average ROC Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    figures.append({"id": "figure_05_roc_curve", **save_plot("figure_05_roc_curve", fig)})

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.bar(charts["detection_counts"]["labels"], charts["detection_counts"]["values"])
    ax.set_ylabel("Flow count")
    ax.set_title("Attack Detection and Containment Coverage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    figures.append({"id": "figure_06_detection_coverage", **save_plot("figure_06_detection_coverage", fig)})

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
    neuro_symbolic = ablation["systems"][1]["metrics"]
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
    overview = engine.overview_data()
    backend = engine.backend_status()
    evaluation_limit = backend["test_rows"] if limit is None else limit
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
    novelty = engine.novelty_data(evaluation_limit, min(0.40, max(0.01, alpha)), flow_index=flow_index, seed=seed)
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
