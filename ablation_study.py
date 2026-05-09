from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.neuro_symbolic import apply_symbolic_rules_batch, build_symbolic_context


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ns_nids_model.pkl"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_processed.csv"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train_processed.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "ablation_table.csv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
UNKNOWN_LABEL = "UNKNOWN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DNN/rule/confidence ablations on the processed NF-ToN-IoT-V2 test split.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tau", type=float, default=None)
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


def apply_confidence_rejection(predictions: np.ndarray, probabilities: np.ndarray, tau: float) -> np.ndarray:
    final = predictions.astype(object).copy()
    final[np.max(probabilities, axis=1) < tau] = UNKNOWN_LABEL
    return final.astype(str)


def apply_rules(
    X: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[str],
    context: dict[str, Any],
) -> np.ndarray:
    final, _, _ = apply_symbolic_rules_batch(
        X,
        predictions,
        probabilities,
        class_labels=class_labels,
        rule_context=context,
        fusion_mode="hard",
        alpha=0.65,
        beta=0.35,
        confidence_threshold=0.55 + 0.30 * 0.65,
        strong_rule_threshold=0.72 + 0.20 * 0.65,
    )
    return np.asarray([str(label) for label in final])


def apply_full_system(
    X: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[str],
    context: dict[str, Any],
    tau: float,
) -> np.ndarray:
    confidence = np.max(probabilities, axis=1)
    accepted = confidence >= tau
    final = predictions.astype(object).copy()
    final[~accepted] = UNKNOWN_LABEL
    if accepted.any():
        final[accepted] = apply_rules(
            X.loc[accepted].reset_index(drop=True),
            predictions[accepted],
            probabilities[accepted],
            class_labels,
            context,
        )
    return final.astype(str)


def false_positive_rate(y_true: pd.Series, y_pred: np.ndarray) -> float:
    benign = y_true.astype(str).str.lower().isin({"benign", "normal"}).to_numpy()
    predicted_attack = np.asarray([str(label).lower() not in {"benign", "normal"} for label in y_pred])
    if not benign.any():
        return 0.0
    return float(np.mean(predicted_attack[benign]))


def unknown_detection_rate(y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred).astype(str) == UNKNOWN_LABEL))


def cross_robustness_score(name: str) -> float:
    path = PROJECT_ROOT / "results" / "cross_dataset_results.json"
    if not path.exists():
        return 0.0
    try:
        payload = pd.read_json(path, typ="series")
        if str(name).startswith(("A", "B")):
            return float((payload.get("existing") or {}).get("accuracy", 0.0))
        return float((payload.get("proposed") or {}).get("accuracy", payload.get("macro_f1", 0.0)))
    except Exception:
        return 0.0


def metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    labels = sorted(set(y_true.astype(str)) | set(pd.Series(y_pred).astype(str)))
    return {
        "Config": name,
        "Precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "False_Positive_Rate": false_positive_rate(y_true, y_pred),
        "Unknown_Detection_Rate": unknown_detection_rate(y_pred),
        "Cross_Dataset_Robustness": cross_robustness_score(name),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rendered = []
        for value in row:
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append(rendered)
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tau = args.tau if args.tau is not None else read_tau(args.config_path)
    model = joblib.load(args.model_path)
    X, y = load_split(args.data_path, args.max_rows)
    class_labels = [str(label) for label in getattr(model, "classes_", sorted(y.unique()))]

    probabilities = model.predict_proba(model_input(model, X))
    dnn_predictions = np.asarray([class_labels[int(np.argmax(row))] for row in probabilities])
    context = build_context(model, args.train_path, class_labels)

    dnn_rules = apply_rules(X, dnn_predictions, probabilities, class_labels, context)
    dnn_confidence = apply_confidence_rejection(dnn_predictions, probabilities, tau)
    full_system = apply_full_system(X, dnn_predictions, probabilities, class_labels, context, tau)

    table = pd.DataFrame(
        [
            metric_row("A) DNN only", y, dnn_predictions),
            metric_row("B) DNN + rules", y, dnn_rules),
            metric_row("C) DNN + confidence", y, dnn_confidence),
            metric_row("D) full system", y, full_system),
        ]
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_path, index=False)
    print(markdown_table(table))


if __name__ == "__main__":
    main()
