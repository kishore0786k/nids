from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.publication_protocol import (
    abstention_metrics,
    apply_rules_closed_set,
    apply_temperature,
    build_context_from_split,
    class_predictions,
    closed_set_metrics,
    expected_calibration_error,
    false_positive_rate,
    fit_temperature,
    json_ready_dataclass,
    model_input,
    tune_rule_fusion,
    tune_tau,
    validation_split,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ns_nids_model.pkl"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_processed.csv"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train_processed.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "ablation_table.csv"
DEFAULT_OPEN_SET_OUTPUT_PATH = PROJECT_ROOT / "results" / "open_set_abstention_table.csv"
DEFAULT_CROSS_DATASET_OUTPUT_PATH = PROJECT_ROOT / "results" / "cross_dataset_table.csv"
DEFAULT_PROTOCOL_OUTPUT_PATH = PROJECT_ROOT / "results" / "publication_reporting_protocol.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DNN/rule/confidence ablations on the processed NF-ToN-IoT-V2 test split.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--open_set_output_path", type=Path, default=DEFAULT_OPEN_SET_OUTPUT_PATH)
    parser.add_argument("--cross_dataset_output_path", type=Path, default=DEFAULT_CROSS_DATASET_OUTPUT_PATH)
    parser.add_argument("--protocol_output_path", type=Path, default=DEFAULT_PROTOCOL_OUTPUT_PATH)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--validation_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=None)
    return parser.parse_args()


def load_split(path: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, nrows=max_rows)
    label_col = "label" if "label" in frame.columns else "Label"
    if label_col not in frame.columns:
        raise ValueError(f"{path} must contain label or Label.")
    return frame.drop(columns=[label_col]), frame[label_col].astype(str)


def metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    metrics = closed_set_metrics(y_true, y_pred)
    return {
        "System": name,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Macro_F1": metrics["macro_f1"],
        "False_Positive_Rate": false_positive_rate(y_true, y_pred),
    }


def cross_dataset_table() -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "cross_dataset_results.json"
    if not path.exists():
        return pd.DataFrame(
            [{
                "Dataset": "NF-UNSW-NB15",
                "Role": "robustness_test",
                "Status": "not_run",
                "Note": "Run evaluate_cross_dataset.py to populate this robustness table.",
            }]
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    existing = payload.get("existing", {})
    proposed = payload.get("proposed", {})
    return pd.DataFrame(
        [{
            "Dataset": Path(str(payload.get("data_path", "NF-UNSW-NB15"))).name,
            "Role": str(payload.get("role", "robustness_test_not_main_accuracy_headline")),
            "Existing_Macro_F1": float(existing.get("macro_f1", 0.0)),
            "Proposed_Macro_F1": float(proposed.get("macro_f1", 0.0)),
            "Open_Set_Macro_F1_With_UNKNOWN": float(proposed.get("open_set_macro_f1_with_unknown", payload.get("open_set_macro_f1_with_unknown", 0.0))),
            "Proposed_Rejection_Rate": float(proposed.get("rejection_rate", 0.0)),
            "Unknown_Attack_Detection_Rate": float(proposed.get("unknown_attack_detection_rate", 0.0)),
        }]
    )


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

    raw_probabilities = model.predict_proba(model_input(model, X))
    probabilities = apply_temperature(raw_probabilities, temperature)
    dnn_predictions = class_predictions(probabilities, class_labels)
    dnn_rules = apply_rules_closed_set(X, dnn_predictions, probabilities, class_labels, context, rule_params)
    proposed_closed = dnn_rules.copy()

    table = pd.DataFrame(
        [
            metric_row("DNN-only", y, dnn_predictions),
            metric_row("DNN+rules (validation-tuned soft fusion)", y, dnn_rules),
            metric_row("Proposed closed-set (calibrated + soft abstention flag)", y, proposed_closed),
        ]
    )

    confidence = np.max(probabilities, axis=1)
    dnn_correct = (dnn_predictions == y.to_numpy()).astype(int)
    proposed_correct = (proposed_closed == y.to_numpy()).astype(int)
    open_set_table = pd.DataFrame(
        [
            {
                "System": "DNN-only",
                "Tau": np.nan,
                "Temperature": temperature,
                "Unknown_Rejection_Rate": 0.0,
                "Benign_Unknown_FPR": 0.0,
                "Accepted_Coverage": 1.0,
                "Accepted_Macro_F1": closed_set_metrics(y, dnn_predictions)["macro_f1"],
                "ECE": expected_calibration_error(confidence, dnn_correct, bins=10),
            },
            {
                "System": "Proposed abstention flag",
                "Tau": tau,
                "Temperature": temperature,
                **{
                    "Unknown_Rejection_Rate": abstention_metrics(y, proposed_closed, probabilities, tau)["unknown_rejection_rate"],
                    "Benign_Unknown_FPR": abstention_metrics(y, proposed_closed, probabilities, tau)["benign_unknown_false_positive_rate"],
                    "Accepted_Coverage": abstention_metrics(y, proposed_closed, probabilities, tau)["accepted_coverage"],
                    "Accepted_Macro_F1": abstention_metrics(y, proposed_closed, probabilities, tau)["accepted_macro_f1"],
                },
                "ECE": expected_calibration_error(confidence, proposed_correct, bins=10),
            },
        ]
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_path, index=False)
    open_set_table.to_csv(args.open_set_output_path, index=False)
    table_c = cross_dataset_table()
    table_c.to_csv(args.cross_dataset_output_path, index=False)

    protocol = {
        "validation": {
            "train_path": str(args.train_path),
            "reference_rows_for_rules": int(len(X_ref)),
            "validation_rows_for_tuning": int(len(X_val)),
            "validation_size": args.validation_size,
            "random_state": args.random_state,
        },
        "temperature_scaling": {
            "temperature": temperature,
            "validation_nll": validation_nll,
        },
        "tau_selection": (
            json_ready_dataclass(tau_selection)
            if tau_selection is not None
            else {"tau": tau, "source": "command_line"}
        ),
        "rule_fusion": json_ready_dataclass(rule_params),
        "outputs": {
            "table_a_closed_set": str(args.output_path),
            "table_b_open_set_abstention": str(args.open_set_output_path),
            "table_c_cross_dataset": str(args.cross_dataset_output_path),
        },
        "fairness_note": (
            "Table A keeps all systems in the same closed-set label space. UNKNOWN is reported only as "
            "an abstention/review metric in Table B."
        ),
    }
    args.protocol_output_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\nTable A: closed-set comparison")
    print(markdown_table(table))
    print("\nTable B: UNKNOWN abstention and calibration")
    print(markdown_table(open_set_table))
    print("\nTable C: cross-dataset robustness")
    print(markdown_table(table_c))


if __name__ == "__main__":
    main()
