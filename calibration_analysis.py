from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.neuro_symbolic import apply_symbolic_rules_batch, build_symbolic_context


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ns_nids_model.pkl"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_processed.csv"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train_processed.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "calibration_curve.png"
DEFAULT_JSON_PATH = PROJECT_ROOT / "results" / "calibration_results.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
UNKNOWN_LABEL = "UNKNOWN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ECE and reliability diagrams for DNN-only and neuro-symbolic NIDS outputs.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--json_path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max_rows", type=int, default=None)
    return parser.parse_args()


def read_tau(config_path: Path, default: float = 0.70) -> float:
    if not config_path.exists():
        return default
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {"unknown_confidence_threshold", "tau"}:
            try:
                return float(value.strip().strip("'\""))
            except ValueError:
                return default
    return default


def model_input(model: Any, frame: pd.DataFrame) -> pd.DataFrame | np.ndarray:
    return frame if getattr(model, "feature_names_in_", None) is not None else frame.to_numpy()


def load_split(path: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, nrows=max_rows)
    label_col = "label" if "label" in frame.columns else "Label"
    if label_col not in frame.columns:
        raise ValueError(f"{path} must contain label or Label.")
    return frame.drop(columns=[label_col]), frame[label_col].astype(str)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = max(1, len(confidence))
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        ece += float(mask.sum() / n) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def build_context(model: Any, train_path: Path, class_labels: list[str]) -> dict[str, Any]:
    X_train, y_train = load_split(train_path)
    probs = model.predict_proba(model_input(model, X_train))
    base_predictions = [class_labels[int(np.argmax(row))] for row in probs]
    return build_symbolic_context(
        X_train,
        reference_y=y_train.tolist(),
        class_labels=class_labels,
        predicted_probs=probs,
        base_predictions=base_predictions,
    )


def proposed_predictions(
    X: pd.DataFrame,
    dnn_predictions: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[str],
    context: dict[str, Any],
    tau: float,
) -> np.ndarray:
    confidence = np.max(probabilities, axis=1)
    accepted = confidence >= tau
    final = dnn_predictions.astype(object).copy()
    final[~accepted] = UNKNOWN_LABEL
    if accepted.any():
        rules_pred, _, _ = apply_symbolic_rules_batch(
            X.loc[accepted].reset_index(drop=True),
            dnn_predictions[accepted],
            probabilities[accepted],
            class_labels=class_labels,
            rule_context=context,
            fusion_mode="hard",
            alpha=0.65,
            beta=0.35,
            confidence_threshold=0.55 + 0.30 * 0.65,
            strong_rule_threshold=0.72 + 0.20 * 0.65,
        )
        final[accepted] = rules_pred
    return final.astype(str)


def curve_points(correct: np.ndarray, confidence: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    prob_true, prob_pred = calibration_curve(correct.astype(int), confidence, n_bins=bins, strategy="uniform")
    return prob_pred, prob_true


def main() -> None:
    args = parse_args()
    tau = args.tau if args.tau is not None else read_tau(args.config_path)
    model = joblib.load(args.model_path)
    X, y = load_split(args.data_path, args.max_rows)
    class_labels = [str(label) for label in getattr(model, "classes_", sorted(y.unique()))]

    probabilities = model.predict_proba(model_input(model, X))
    confidence = np.max(probabilities, axis=1)
    dnn_predictions = np.asarray([class_labels[int(np.argmax(row))] for row in probabilities])
    context = build_context(model, args.train_path, class_labels)
    proposed = proposed_predictions(X, dnn_predictions, probabilities, class_labels, context, tau)

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
        "tau": tau,
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
        },
        "plot": str(args.output_path),
    }
    args.json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
