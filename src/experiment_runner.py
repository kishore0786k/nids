from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.baseline_models import get_models
from src.neuro_symbolic import apply_symbolic_rules, build_symbolic_context
from src.project_paths import PUBLICATION_EXPERIMENT_PATH, TEST_PATH, TRAIN_PATH


LABEL_COLUMN = "label"
EVALUATED_BASELINES = ("RandomForest", "MLP")
RESULTS_PATH = PUBLICATION_EXPERIMENT_PATH


def _set_reproducible_params(model: Any, seed: int = 42) -> Any:
    params = model.get_params() if hasattr(model, "get_params") else {}
    updates: dict[str, Any] = {}
    if "random_state" in params:
        updates["random_state"] = seed
    if "n_jobs" in params:
        updates["n_jobs"] = -1
    if updates and hasattr(model, "set_params"):
        model.set_params(**updates)
    return model


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing required '{LABEL_COLUMN}' column.")
    return df.drop(LABEL_COLUMN, axis=1), df[LABEL_COLUMN].astype(str)


def load_dataset(quick_limit: int | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Use the fixed processed train/test holdout; fallback preserves old behavior."""
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        X_train, y_train = _split_xy(pd.read_csv(TRAIN_PATH))
        X_test, y_test = _split_xy(pd.read_csv(TEST_PATH))
    else:
        df = pd.read_csv(TEST_PATH)
        X, y = _split_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if quick_limit:
        train_n = min(len(X_train), max(200, int(quick_limit) * 4))
        test_n = min(len(X_test), max(100, int(quick_limit)))
        X_train, y_train = X_train.head(train_n), y_train.head(train_n)
        X_test, y_test = X_test.head(test_n), y_test.head(test_n)
    return X_train, y_train, X_test, y_test


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    return seeds or [42]


def metric_summary(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def is_attack_label(label: str) -> bool:
    return str(label).lower() not in {"benign", "normal"}


def binary_attack_recall(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    true_attack = np.asarray([is_attack_label(label) for label in y_true], dtype=bool)
    pred_attack = np.asarray([is_attack_label(label) for label in y_pred], dtype=bool)
    return float(np.sum(true_attack & pred_attack) / max(1, true_attack.sum()))


def evaluate_baseline(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = np.asarray([str(label) for label in model.predict(X_test)])
    return {
        "predictions": predictions,
        "metrics": metric_summary(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=list(model.classes_) if hasattr(model, "classes_") else None,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_neuro_symbolic(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fusion_mode: str = "soft",
    alpha: float = 0.65,
    rules_enabled: bool = True,
) -> dict[str, Any]:
    class_labels = [str(label) for label in model.classes_]

    train_probs = model.predict_proba(X_train)
    train_base_preds = [class_labels[int(idx)] for idx in np.argmax(train_probs, axis=1)]
    rule_context = build_symbolic_context(
        X_train,
        reference_y=y_train.tolist(),
        class_labels=class_labels,
        predicted_probs=train_probs,
        base_predictions=train_base_preds,
    )

    probabilities = model.predict_proba(X_test)
    base_predictions = np.asarray([class_labels[int(idx)] for idx in np.argmax(probabilities, axis=1)])
    ns_predictions: list[str] = []
    rule_traces: list[list[dict[str, Any]]] = []
    strengths: list[float] = []

    for row_idx, base_label in enumerate(base_predictions):
        if rules_enabled:
            final_label, fired_rules, strength = apply_symbolic_rules(
                X_test.iloc[row_idx],
                base_label,
                predicted_probs=probabilities[row_idx],
                class_labels=class_labels,
                rule_context=rule_context,
                fusion_mode=fusion_mode,
                alpha=alpha,
            )
        else:
            final_label, fired_rules, strength = base_label, [], 0.0
        ns_predictions.append(str(final_label))
        rule_traces.append(fired_rules)
        strengths.append(float(strength))

    ns_predictions_arr = np.asarray(ns_predictions)
    diagnostics = rule_diagnostics(
        y_test=y_test,
        base_predictions=base_predictions,
        ns_predictions=ns_predictions_arr,
        rule_traces=rule_traces,
        strengths=strengths,
    )

    return {
        "predictions": ns_predictions_arr,
        "base_predictions": base_predictions,
        "probabilities": probabilities,
        "rule_context": rule_context,
        "rule_traces": rule_traces,
        "diagnostics": diagnostics,
        "metrics": metric_summary(y_test, ns_predictions_arr),
        "classification_report": classification_report(
            y_test,
            ns_predictions_arr,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def rule_diagnostics(
    y_test: Sequence[str],
    base_predictions: Sequence[str],
    ns_predictions: Sequence[str],
    rule_traces: Sequence[Sequence[dict[str, Any]]],
    strengths: Sequence[float],
) -> dict[str, Any]:
    n_samples = len(base_predictions)
    fired_samples = 0
    rule_counts: Counter[str] = Counter()
    applied_counts: Counter[str] = Counter()

    for rules in rule_traces:
        active_rules = [rule for rule in rules if rule.get("rule_id") != "NONE"]
        if active_rules:
            fired_samples += 1
        for rule in active_rules:
            rule_id = str(rule.get("rule_id", "UNKNOWN"))
            rule_counts[rule_id] += 1
            if bool(rule.get("applied")):
                applied_counts[rule_id] += 1

    base_predictions_arr = np.asarray(base_predictions)
    ns_predictions_arr = np.asarray(ns_predictions)
    changed_mask = base_predictions_arr != ns_predictions_arr
    base_acc = accuracy_score(y_test, base_predictions_arr)
    ns_acc = accuracy_score(y_test, ns_predictions_arr)
    base_f1 = f1_score(y_test, base_predictions_arr, average="macro", zero_division=0)
    ns_f1 = f1_score(y_test, ns_predictions_arr, average="macro", zero_division=0)
    base_attack_recall = binary_attack_recall(y_test, base_predictions_arr)
    ns_attack_recall = binary_attack_recall(y_test, ns_predictions_arr)

    return {
        "samples": int(n_samples),
        "rules_fired": int(fired_samples),
        "rules_fired_pct": 100.0 * fired_samples / max(1, n_samples),
        "predictions_changed": int(changed_mask.sum()),
        "prediction_change_count": int(changed_mask.sum()),
        "predictions_changed_pct": 100.0 * float(changed_mask.mean()),
        "prediction_change_pct": 100.0 * float(changed_mask.mean()),
        "accuracy_before_rules": float(base_acc),
        "accuracy_after_rules": float(ns_acc),
        "accuracy_delta": float(ns_acc - base_acc),
        "macro_f1_before_rules": float(base_f1),
        "macro_f1_after_rules": float(ns_f1),
        "macro_f1_delta": float(ns_f1 - base_f1),
        "binary_attack_recall_before_rules": float(base_attack_recall),
        "binary_attack_recall_after_rules": float(ns_attack_recall),
        "binary_attack_recall_delta": float(ns_attack_recall - base_attack_recall),
        "mean_rule_strength": float(np.mean(strengths)) if strengths else 0.0,
        "rule_counts": dict(rule_counts),
        "applied_rule_counts": dict(applied_counts),
        "false_negatives_before": false_negative_count(y_test, base_predictions_arr),
        "false_negatives_after": false_negative_count(y_test, ns_predictions_arr),
    }


def false_negative_count(y_true: Sequence[str], y_pred: Sequence[str]) -> int:
    return int(sum(is_attack_label(str(t)) and not is_attack_label(str(p)) for t, p in zip(y_true, y_pred)))


def novelty_examples(
    y_true: Sequence[str],
    base_predictions: Sequence[str],
    ns_predictions: Sequence[str],
    rule_traces: Sequence[Sequence[dict[str, Any]]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for idx, (truth, base_label, ns_label) in enumerate(zip(y_true, base_predictions, ns_predictions)):
        if str(base_label) == str(ns_label):
            continue
        if not (is_attack_label(str(truth)) and not is_attack_label(str(base_label)) and is_attack_label(str(ns_label))):
            continue
        applied = [rule for rule in rule_traces[idx] if rule.get("applied")]
        rule = applied[0] if applied else next((item for item in rule_traces[idx] if item.get("rule_id") != "NONE"), {})
        examples.append(
            {
                "sample": int(idx),
                "true_label": str(truth),
                "mlp_label": str(base_label),
                "neuro_symbolic_label": str(ns_label),
                "exact_correction": bool(str(truth) == str(ns_label)),
                "rule_id": str(rule.get("rule_id", "")),
                "rule_strength": float(rule.get("strength", 0.0) or 0.0),
                "explanation": str(rule.get("reason", "")),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def mcnemar_approx(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> dict[str, float]:
    """Approximate paired McNemar test for classifier disagreement."""
    a_correct = np.asarray(pred_a) == np.asarray(y_true)
    b_correct = np.asarray(pred_b) == np.asarray(y_true)
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    denom = b01 + b10
    chi2 = 0.0 if denom == 0 else (abs(b01 - b10) - 1.0) ** 2 / denom
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {"mlp_correct_ns_wrong": b01, "mlp_wrong_ns_correct": b10, "chi2": chi2, "p_value": p_value}


def class_delta_rows(labels: Sequence[str], base_report: Mapping[str, Any], ns_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        base_row = base_report.get(label, {})
        ns_row = ns_report.get(label, {})
        rows.append({
            "class": label,
            "base_recall": float(base_row.get("recall", 0.0)),
            "ns_recall": float(ns_row.get("recall", 0.0)),
            "recall_delta": float(ns_row.get("recall", 0.0)) - float(base_row.get("recall", 0.0)),
            "base_f1": float(base_row.get("f1-score", 0.0)),
            "ns_f1": float(ns_row.get("f1-score", 0.0)),
            "f1_delta": float(ns_row.get("f1-score", 0.0)) - float(base_row.get("f1-score", 0.0)),
        })
    return rows


def save_publication_artifacts(
    output: Mapping[str, Any],
    labels: Sequence[str],
    y_test: Sequence[str],
    results: Mapping[str, Any],
) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    comparison_rows = []
    for name, result in results.items():
        if "metrics" not in result:
            continue
        diag = result.get("diagnostics", {})
        comparison_rows.append({
            "model": name,
            **result["metrics"],
            "rules_fired": diag.get("rules_fired", 0),
            "predictions_changed": diag.get("predictions_changed", 0),
            "attack_false_negatives": diag.get("false_negatives_after", false_negative_count(y_test, result["predictions"])),
        })
    pd.DataFrame(comparison_rows).to_csv(RESULTS_PATH.parent / "model_comparison.csv", index=False)
    pd.DataFrame(output["attack_class_deltas"]).to_csv(RESULTS_PATH.parent / "attack_class_deltas.csv", index=False)
    pd.DataFrame([output["mcnemar_mlp_vs_neurosymbolic"]]).to_csv(RESULTS_PATH.parent / "mcnemar_mlp_vs_neurosymbolic.csv", index=False)

    for name in ("RandomForest", "MLP", "NeuroSymbolic"):
        matrix = confusion_matrix(y_test, results[name]["predictions"], labels=labels)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(RESULTS_PATH.parent / f"confusion_{name.lower()}.csv")

    rule_counts = output["rule_diagnostics"].get("rule_counts", {})
    applied_counts = output["rule_diagnostics"].get("applied_rule_counts", {})
    rule_rows = [
        {"rule_id": rule_id, "trigger_count": count, "applied_count": applied_counts.get(rule_id, 0)}
        for rule_id, count in rule_counts.items()
    ]
    pd.DataFrame(rule_rows).to_csv(RESULTS_PATH.parent / "rule_diagnostics.csv", index=False)


def print_model_report(name: str, y_true: pd.Series, predictions: Sequence[str], labels: Sequence[str]) -> None:
    metrics = metric_summary(y_true, predictions)
    print(f"\n=== {name} ===")
    print(
        "Accuracy={accuracy:.4f} | Precision={precision:.4f} | "
        "Recall={recall:.4f} | F1={f1:.4f}".format(**metrics)
    )
    print("\nClassification report:")
    print(classification_report(y_true, predictions, labels=labels, zero_division=0))
    print("Confusion matrix:")
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())


def print_rule_diagnostics(diagnostics: Mapping[str, Any]) -> None:
    print("\n=== Neuro-Symbolic Rule Diagnostics ===")
    print(
        f"Rules fired: {diagnostics['rules_fired']}/{diagnostics['samples']} "
        f"({diagnostics['rules_fired_pct']:.2f}%)"
    )
    print(
        f"Predictions changed: {diagnostics['predictions_changed']}/{diagnostics['samples']} "
        f"({diagnostics['predictions_changed_pct']:.2f}%)"
    )
    print(
        "Accuracy before vs after rules: "
        f"{diagnostics['accuracy_before_rules']:.4f} -> {diagnostics['accuracy_after_rules']:.4f} "
        f"(delta {diagnostics['accuracy_delta']:+.4f})"
    )
    print(
        "Macro F1 before vs after rules: "
        f"{diagnostics['macro_f1_before_rules']:.4f} -> {diagnostics['macro_f1_after_rules']:.4f} "
        f"(delta {diagnostics['macro_f1_delta']:+.4f})"
    )
    print(
        "Binary attack recall before vs after rules: "
        f"{diagnostics['binary_attack_recall_before_rules']:.4f} -> "
        f"{diagnostics['binary_attack_recall_after_rules']:.4f} "
        f"(delta {diagnostics['binary_attack_recall_delta']:+.4f})"
    )
    print(f"Mean rule strength: {diagnostics['mean_rule_strength']:.4f}")
    print(f"Rule trigger counts: {diagnostics['rule_counts']}")
    print(f"Applied override counts: {diagnostics['applied_rule_counts']}")
    print(
        "Attack false negatives before vs after rules: "
        f"{diagnostics['false_negatives_before']} -> {diagnostics['false_negatives_after']}"
    )


def print_attack_improvements(
    labels: Sequence[str],
    mlp_report: Mapping[str, Any],
    ns_report: Mapping[str, Any],
) -> None:
    print("\n=== Attack-Class Improvement vs MLP ===")
    print("Class | MLP Recall | NS Recall | Recall Delta | Attack Class")
    rows = []
    any_improved = False
    for label in labels:
        if label.lower() in {"benign", "normal"}:
            continue
        mlp_row = mlp_report.get(label, {})
        ns_row = ns_report.get(label, {})
        recall_delta = float(ns_row.get("recall", 0.0)) - float(mlp_row.get("recall", 0.0))
        f1_delta = float(ns_row.get("f1-score", 0.0)) - float(mlp_row.get("f1-score", 0.0))
        any_improved = any_improved or recall_delta > 0 or f1_delta > 0
        rows.append(
            {
                "class": label,
                "mlp_recall": float(mlp_row.get("recall", 0.0)),
                "ns_recall": float(ns_row.get("recall", 0.0)),
                "recall_delta": recall_delta,
                "mlp_f1": float(mlp_row.get("f1-score", 0.0)),
                "ns_f1": float(ns_row.get("f1-score", 0.0)),
                "f1_delta": f1_delta,
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Attack classes improved: {'YES' if any_improved else 'NO'}")


def print_novelty_examples(examples: Sequence[Mapping[str, Any]]) -> None:
    print("\n=== Example Neuro-Symbolic Corrections ===")
    if not examples:
        print("No benign-to-attack correction examples were found in this evaluation window.")
        return
    for item in examples:
        mark = "correct" if item.get("exact_correction") else "attack-rescue"
        print(
            f"Sample {item['sample']}: MLP -> {item['mlp_label']} | "
            f"NS -> {item['neuro_symbolic_label']} | true={item['true_label']} ({mark})"
        )
        print(f"Rule -> {item['rule_id']} ({item['rule_strength']:.3f}): {item['explanation']}")


def run_experiment(
    fusion_mode: str = "soft",
    alpha: float = 0.65,
    quick_limit: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    print("Loading processed dataset...")
    X_train, y_train, X_test, y_test = load_dataset(quick_limit=quick_limit)
    print(f"Train rows={len(X_train)} | Test rows={len(X_test)} | Features={X_train.shape[1]}")
    print(f"Seed={seed}")

    models = get_models()
    trained_models: dict[str, Any] = {}
    results: dict[str, Any] = {}

    print("\n=== Training Baseline Models ===")
    for name in EVALUATED_BASELINES:
        if name not in models:
            raise KeyError(f"Required baseline '{name}' was not returned by get_models().")
        print(f"Training {name}...")
        model = _set_reproducible_params(models[name], seed=seed)
        model.fit(X_train, y_train)
        trained_models[name] = model
        results[name] = evaluate_baseline(model, X_test, y_test)

    print("\nEvaluating Neuro-Symbolic ablations...")
    no_rules = evaluate_neuro_symbolic(
        trained_models["MLP"],
        X_train,
        y_train,
        X_test,
        y_test,
        fusion_mode="hard",
        alpha=1.0,
        rules_enabled=False,
    )
    hard_results = evaluate_neuro_symbolic(
        trained_models["MLP"],
        X_train,
        y_train,
        X_test,
        y_test,
        fusion_mode="hard",
        alpha=alpha,
    )
    ns_results = evaluate_neuro_symbolic(
        trained_models["MLP"],
        X_train,
        y_train,
        X_test,
        y_test,
        fusion_mode=fusion_mode,
        alpha=alpha,
    )
    results["MLP_NoRules"] = no_rules
    results["NeuroSymbolic_Hard"] = hard_results
    results["NeuroSymbolic"] = ns_results

    labels = [str(label) for label in trained_models["MLP"].classes_]
    print("\n=== Final Model Comparison ===")
    for name in (*EVALUATED_BASELINES, "NeuroSymbolic"):
        print_model_report(name, y_test, results[name]["predictions"], labels)

    print_rule_diagnostics(ns_results["diagnostics"])
    print_attack_improvements(labels, results["MLP"]["classification_report"], ns_results["classification_report"])
    examples = novelty_examples(
        y_test,
        results["MLP"]["predictions"],
        ns_results["predictions"],
        ns_results["rule_traces"],
    )
    print_novelty_examples(examples)

    print("\n=== Ablation Summary ===")
    for name in ("MLP_NoRules", "NeuroSymbolic_Hard", "NeuroSymbolic"):
        m = results[name]["metrics"]
        d = results[name]["diagnostics"]
        print(f"{name}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} Changed={d['predictions_changed']} FN={d['false_negatives_after']}")

    mlp_metrics = results["MLP"]["metrics"]
    ns_metrics = ns_results["metrics"]
    print("\n=== Improvement Over MLP ===")
    print(f"Accuracy delta: {ns_metrics['accuracy'] - mlp_metrics['accuracy']:+.4f}")
    print(f"Macro precision delta: {ns_metrics['precision'] - mlp_metrics['precision']:+.4f}")
    print(f"Macro recall delta: {ns_metrics['recall'] - mlp_metrics['recall']:+.4f}")
    print(f"Macro F1 delta: {ns_metrics['f1'] - mlp_metrics['f1']:+.4f}")
    print(f"Binary attack recall delta: {ns_results['diagnostics']['binary_attack_recall_delta']:+.4f}")

    mcnemar = mcnemar_approx(y_test, results["MLP"]["predictions"], ns_results["predictions"])
    print(
        "McNemar MLP vs NeuroSymbolic: "
        f"MLP-correct/NS-wrong={mcnemar['mlp_correct_ns_wrong']}, "
        f"MLP-wrong/NS-correct={mcnemar['mlp_wrong_ns_correct']}, "
        f"chi2={mcnemar['chi2']:.4f}, p={mcnemar['p_value']:.4f}"
    )
    novelty_proof = {
        "ns_beats_mlp_accuracy": bool(ns_metrics["accuracy"] > mlp_metrics["accuracy"]),
        "ns_beats_mlp_macro_f1": bool(ns_metrics["f1"] > mlp_metrics["f1"]),
        "binary_attack_recall_delta": ns_results["diagnostics"]["binary_attack_recall_delta"],
        "attack_recall_improved": bool(ns_results["diagnostics"]["binary_attack_recall_delta"] > 0),
        "verdict": (
            "proven"
            if ns_metrics["accuracy"] > mlp_metrics["accuracy"]
            or ns_metrics["f1"] > mlp_metrics["f1"]
            or ns_results["diagnostics"]["binary_attack_recall_delta"] > 0
            else "not_proven"
        ),
    }
    print(f"Novelty proof verdict: {novelty_proof['verdict']}")

    output = {
        "protocol": {
            "train_path": str(TRAIN_PATH),
            "test_path": str(TEST_PATH),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "fusion_mode": fusion_mode,
            "alpha": alpha,
            "quick_limit": quick_limit,
            "seed": seed,
        },
        "metrics": {name: value["metrics"] for name, value in results.items() if "metrics" in value},
        "classification_reports": {
            name: value["classification_report"]
            for name, value in results.items()
            if "classification_report" in value
        },
        "confusion_matrices": {
            name: confusion_matrix(y_test, value["predictions"], labels=labels).tolist()
            for name, value in results.items()
            if name in ("RandomForest", "MLP", "NeuroSymbolic")
        },
        "rule_diagnostics": ns_results["diagnostics"],
        "attack_class_deltas": class_delta_rows(labels, results["MLP"]["classification_report"], ns_results["classification_report"]),
        "mcnemar_mlp_vs_neurosymbolic": mcnemar,
        "novelty_examples": examples,
        "novelty_proof": novelty_proof,
        "novelty_claim": (
            "A confidence-aware neuro-symbolic NIDS that calibrates symbolic rules from training percentiles "
            "and fuses them with neural probabilities to reduce attack false negatives with auditable rule traces."
        ),
    }
    save_publication_artifacts(output, labels, y_test, results)
    print(f"\nSaved publication experiment artifact: {RESULTS_PATH}")

    return results


def run_multi_seed(seeds: Sequence[int], fusion_mode: str, alpha: float, quick_limit: int | None) -> None:
    rows = []
    for seed in seeds:
        results = run_experiment(fusion_mode=fusion_mode, alpha=alpha, quick_limit=quick_limit, seed=seed)
        for model_name in ("RandomForest", "MLP", "NeuroSymbolic"):
            row = {"seed": seed, "model": model_name}
            row.update(results[model_name]["metrics"])
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_PATH.parent / "multiseed_metrics.csv", index=False)
    aggregate = summary.groupby("model")[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"]).round(6)
    aggregate.to_csv(RESULTS_PATH.parent / "multiseed_summary.csv")
    print("\n=== Multi-Seed Summary ===")
    print(aggregate.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline and neuro-symbolic NIDS experiments.")
    parser.add_argument("--fusion-mode", choices=("hard", "soft"), default="soft")
    parser.add_argument("--alpha", type=float, default=0.65, help="Neural probability weight for soft fusion.")
    parser.add_argument("--quick-limit", type=int, default=None, help="Optional small test window for fast smoke runs.")
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds, e.g. 42,43,44.")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if len(seeds) == 1:
        run_experiment(fusion_mode=args.fusion_mode, alpha=args.alpha, quick_limit=args.quick_limit, seed=seeds[0])
    else:
        run_multi_seed(seeds=seeds, fusion_mode=args.fusion_mode, alpha=args.alpha, quick_limit=args.quick_limit)
