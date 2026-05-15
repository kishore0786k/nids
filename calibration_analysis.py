from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.publication_protocol import (
    abstention_metrics,
    apply_rules_closed_set,
    apply_temperature,
    build_context_from_split,
    class_predictions,
    expected_calibration_error,
    fit_temperature,
    json_ready_dataclass,
    model_input,
    tune_rule_fusion,
    tune_tau,
    validation_split,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ns_nids_model.pkl"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_processed.csv"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train_processed.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "calibration_curve.png"
DEFAULT_JSON_PATH = PROJECT_ROOT / "results" / "calibration_results.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ECE and reliability diagrams for DNN-only and neuro-symbolic NIDS outputs.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--json_path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--validation_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max_rows", type=int, default=None)
    return parser.parse_args()


def load_split(path: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, nrows=max_rows)
    label_col = "label" if "label" in frame.columns else "Label"
    if label_col not in frame.columns:
        raise ValueError(f"{path} must contain label or Label.")
    return frame.drop(columns=[label_col]), frame[label_col].astype(str)


def curve_points(correct: np.ndarray, confidence: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    prob_true, prob_pred = calibration_curve(correct.astype(int), confidence, n_bins=bins, strategy="uniform")
    return prob_pred, prob_true


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model_path)
    X, y = load_split(args.data_path, args.max_rows)
    X_train, y_train = load_split(args.train_path)
    class_labels = [str(label) for label in getattr(model, "classes_", sorted(y.unique()))]

    X_ref, y_ref, X_val, y_val = validation_split(
        X_train,
        y_train,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    context = build_context_from_split(model, X_ref, y_ref, class_labels)
    val_probabilities_raw = model.predict_proba(model_input(model, X_val))
    temperature, validation_nll = fit_temperature(val_probabilities_raw, y_val, class_labels)
    val_probabilities = apply_temperature(val_probabilities_raw, temperature)
    val_dnn_predictions = class_predictions(val_probabilities, class_labels)
    rule_params = tune_rule_fusion(X_val, y_val, val_dnn_predictions, val_probabilities, class_labels, context)
    val_proposed = apply_rules_closed_set(X_val, val_dnn_predictions, val_probabilities, class_labels, context, rule_params)
    tau_selection = (
        tune_tau(y_val, val_proposed, val_probabilities, class_labels)
        if args.tau is None
        else None
    )
    tau = float(args.tau if args.tau is not None else (tau_selection.tau if tau_selection else 0.20))

    probabilities_raw = model.predict_proba(model_input(model, X))
    probabilities = apply_temperature(probabilities_raw, temperature)
    confidence = np.max(probabilities, axis=1)
    dnn_predictions = class_predictions(probabilities, class_labels)
    proposed = apply_rules_closed_set(X, dnn_predictions, probabilities, class_labels, context, rule_params)

    dnn_correct = (dnn_predictions == y.to_numpy()).astype(int)
    proposed_correct = (proposed == y.to_numpy()).astype(int)
    dnn_pred_conf, dnn_true_acc = curve_points(dnn_correct, confidence, args.bins)
    proposed_pred_conf, proposed_true_acc = curve_points(proposed_correct, confidence, args.bins)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(dnn_pred_conf, dnn_true_acc, marker="o", label="DNN only")
    ax.plot(proposed_pred_conf, proposed_true_acc, marker="s", label="Proposed")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    results = {
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "train_path": str(args.train_path),
        "tau": tau,
        "temperature": temperature,
        "validation_nll": validation_nll,
        "tau_selection": (
            json_ready_dataclass(tau_selection)
            if tau_selection is not None
            else {"tau": tau, "source": "command_line"}
        ),
        "rule_fusion": json_ready_dataclass(rule_params),
        "bins": args.bins,
        "dnn_only": {
            "ece": expected_calibration_error(confidence, dnn_correct, args.bins),
            "mean_confidence": float(np.mean(confidence)),
            "accuracy": float(np.mean(dnn_correct)),
        },
        "proposed": {
            "ece": expected_calibration_error(confidence, proposed_correct, args.bins),
            "mean_confidence": float(np.mean(confidence)),
            "accuracy": float(np.mean(proposed_correct)),
            "abstention": abstention_metrics(y, proposed, probabilities, tau),
        },
        "fairness_note": "Proposed reliability is computed from closed-set labels; UNKNOWN remains a separate abstention metric.",
        "plot": str(args.output_path),
    }
    args.json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
