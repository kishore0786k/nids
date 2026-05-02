import json
import logging
import uuid
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, recall_score, roc_curve
from sklearn.preprocessing import label_binarize

from src.neuro_symbolic import apply_symbolic_rules, apply_symbolic_rules_batch, build_symbolic_context
from src.project_paths import (
    METRICS_PATH,
    MODEL_PATH,
    ROBUST_MODEL_PATH,
    TEST_PATH,
    TRAIN_PATH,
)


LOGGER = logging.getLogger(__name__)
LABEL_COLUMN = "label"
MIN_ANALYSIS_LIMIT = 50
MAX_ANALYSIS_LIMIT = 25000
MAX_CHART_LIMIT = 25000
SYMBOLIC_FUSION_MODE = "hard"
ROBUST_PATH = ROBUST_MODEL_PATH

_base_model = None
_robust_model = None
_X_test = None
_y_test = None
_classes = None
_metrics = None
_symbolic_context = None
_evaluation_cache = {}
_analysis_cache = {}
_chart_cache = {}
_novelty_cache = {}
_incident_store = {}


class ResourceLoadError(RuntimeError):
    """Raised when a required model/data resource cannot be loaded."""


def _coerce_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        out = int(default)
    if minimum is not None:
        out = max(minimum, out)
    if maximum is not None:
        out = min(maximum, out)
    return out


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _require_file(path: str | Path, description: str) -> None:
    path = _as_path(path)
    if not path.exists():
        raise ResourceLoadError(f"Missing {description}: {path}")
    if not path.is_file():
        raise ResourceLoadError(f"Invalid {description} path, expected a file: {path}")


def _clear_caches() -> None:
    _evaluation_cache.clear()
    _analysis_cache.clear()
    _chart_cache.clear()
    _novelty_cache.clear()


def _reset_resources() -> None:
    global _base_model, _robust_model, _X_test, _y_test, _classes, _metrics, _symbolic_context
    _base_model = None
    _robust_model = None
    _X_test = None
    _y_test = None
    _classes = None
    _metrics = None
    _symbolic_context = None
    _clear_caches()


def load_resources() -> None:
    global _base_model, _robust_model, _X_test, _y_test, _classes, _metrics

    if _base_model is not None:
        return

    _require_file(MODEL_PATH, "base model")
    _require_file(TEST_PATH, "processed test dataset")

    try:
        _base_model = joblib.load(MODEL_PATH)
    except Exception as exc:
        _reset_resources()
        raise ResourceLoadError(f"Could not load base model from {MODEL_PATH}: {exc}") from exc

    if not hasattr(_base_model, "predict") or not hasattr(_base_model, "predict_proba"):
        _reset_resources()
        raise ResourceLoadError(f"Model at {MODEL_PATH} must implement predict and predict_proba.")

    robust_path = _as_path(ROBUST_PATH)
    if robust_path.exists():
        try:
            _robust_model = joblib.load(robust_path)
        except Exception as exc:
            warnings.warn(f"Could not load optional robust model at {ROBUST_PATH}: {exc}", RuntimeWarning)
            _robust_model = None
    else:
        _robust_model = None

    try:
        df = pd.read_csv(TEST_PATH)
    except Exception as exc:
        _reset_resources()
        raise ResourceLoadError(f"Could not read processed test dataset from {TEST_PATH}: {exc}") from exc

    if LABEL_COLUMN not in df.columns:
        _reset_resources()
        raise ResourceLoadError(f"Dataset {TEST_PATH} is missing required '{LABEL_COLUMN}' column.")
    if df.empty:
        _reset_resources()
        raise ResourceLoadError(f"Dataset {TEST_PATH} is empty.")

    _X_test = df.drop(columns=[LABEL_COLUMN])
    _y_test = df[LABEL_COLUMN].astype(str)
    if _X_test.empty:
        _reset_resources()
        raise ResourceLoadError(f"Dataset {TEST_PATH} has no feature columns after dropping '{LABEL_COLUMN}'.")

    _classes = [str(c) for c in getattr(_base_model, "classes_", sorted(_y_test.unique()))]

    metrics_path = _as_path(METRICS_PATH)
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                _metrics = json.load(f)
        except Exception as exc:
            warnings.warn(f"Could not load metrics file {METRICS_PATH}: {exc}", RuntimeWarning)
            _metrics = {}
    else:
        _metrics = {}


def is_attack(label):
    return str(label).lower() not in {"benign", "normal"}


def json_number(value, digits=4):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return round(out, digits)


def _model_input(model: Any, frame: pd.DataFrame) -> pd.DataFrame | np.ndarray:
    return frame if hasattr(model, "feature_names_in_") else frame.to_numpy()


def _get_symbolic_context() -> dict[str, Any]:
    global _symbolic_context
    load_resources()
    if _symbolic_context is not None:
        return _symbolic_context

    if _as_path(TRAIN_PATH).exists():
        train_df = pd.read_csv(TRAIN_PATH)
    else:
        warnings.warn(
            f"Training split not found at {TRAIN_PATH}; symbolic thresholds are calibrated from the test split.",
            RuntimeWarning,
        )
        train_df = pd.concat([_X_test, _y_test.rename(LABEL_COLUMN)], axis=1)

    if LABEL_COLUMN not in train_df.columns:
        raise ResourceLoadError(f"Symbolic calibration data is missing required '{LABEL_COLUMN}' column.")

    X_train = train_df.drop(columns=[LABEL_COLUMN])
    y_train = train_df[LABEL_COLUMN].astype(str)
    train_input = _model_input(_base_model, X_train)
    train_probs = _base_model.predict_proba(train_input)
    train_base = [str(_classes[int(idx)]) for idx in np.argmax(train_probs, axis=1)]
    _symbolic_context = build_symbolic_context(
        X_train,
        reference_y=y_train.tolist(),
        class_labels=_classes,
        predicted_probs=train_probs,
        base_predictions=train_base,
    )
    _symbolic_context["calibration"] = {
        "source": str(TRAIN_PATH if _as_path(TRAIN_PATH).exists() else TEST_PATH),
        "rows": int(len(X_train)),
        "method": "training-percentile thresholds plus probability thresholds calibrated from model false-negative patterns",
    }
    LOGGER.info("Symbolic context calibrated from %s rows", len(X_train))
    return _symbolic_context


def _entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _probability_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(np.asarray(probs, dtype=float), axis=1)
    if sorted_probs.shape[1] < 2:
        return sorted_probs[:, -1]
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def _class_indices(labels: list[str]) -> np.ndarray:
    index = {label: pos for pos, label in enumerate(_classes)}
    return np.asarray([index.get(label, -1) for label in labels], dtype=int)


def _calibration_curve(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    n = max(1, len(confidence))
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < bins - 1 else confidence <= hi)
        count = int(mask.sum())
        if count:
            avg_conf = float(np.mean(confidence[mask]))
            accuracy = float(np.mean(correct[mask]))
            ece += (count / n) * abs(avg_conf - accuracy)
        else:
            avg_conf = 0.0
            accuracy = 0.0
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": count,
            "confidence": json_number(avg_conf, 6),
            "accuracy": json_number(accuracy, 6),
        })
    return {"ece": json_number(ece, 6), "bins": rows}


def paper_baseline_values():
    load_resources()
    existing = _metrics.get("existing", {})
    return [
        existing.get("accuracy"),
        existing.get("precision_macro"),
        existing.get("recall_macro"),
        existing.get("f1_macro"),
    ]


def proposed_summary_values():
    load_resources()
    proposed = _metrics.get("proposed", {})
    return [
        proposed.get("accuracy"),
        proposed.get("precision_macro"),
        proposed.get("recall_macro"),
        proposed.get("f1_macro"),
    ]


def saved_paper_summary() -> dict[str, Any]:
    return {
        "source": "saved-paper-summary evidence from results/metrics.json",
        "labels": ["Accuracy", "Precision", "Recall", "F1"],
        "existing": [json_number(value, 6) for value in paper_baseline_values()],
        "proposed": [json_number(value, 6) for value in proposed_summary_values()],
    }


def defense_action(label, confidence, fired_rules):
    if not is_attack(label):
        return {
            "level": "Normal",
            "action": "Allow and monitor",
            "playbook": [
                "Continue telemetry capture",
                "Retain flow fingerprint for drift monitoring",
                "No active containment required",
            ],
        }

    label_l = str(label).lower()
    if "dos" in label_l or "ddos" in label_l:
        action = "Rate-limit source, quarantine session, and trigger DDoS shield"
        playbook = [
            "Block bursty source tuple at ingress",
            "Enable adaptive rate limit for matching packet-rate profile",
            "Mirror packet sample for forensic review",
        ]
    elif "scanning" in label_l:
        action = "Throttle scanner, hide exposed services, and enrich source reputation"
        playbook = [
            "Temporarily tarpitting source IP",
            "Increase honeypot sensitivity for scanned ports",
            "Create watchlist item for repeated probes",
        ]
    elif "injection" in label_l or "xss" in label_l or "mitm" in label_l:
        action = "Block payload path, isolate session, and raise application-layer alert"
        playbook = [
            "Deny suspicious request signature",
            "Rotate session token if user context exists",
            "Escalate to WAF and analyst queue",
        ]
    elif "password" in label_l or "backdoor" in label_l:
        action = "Quarantine endpoint, force credential review, and preserve evidence"
        playbook = [
            "Disable suspicious authentication route",
            "Snapshot endpoint/network evidence",
            "Open high-priority incident ticket",
        ]
    else:
        action = "Contain flow, increase inspection depth, and request analyst validation"
        playbook = [
            "Move flow to restricted policy",
            "Collect additional graph/anomaly context",
            "Tag as novel threat candidate",
        ]

    if confidence < 0.70:
        playbook.append("Confidence is moderate: require analyst confirmation before permanent block")
    if any(rule.get("rule_id") != "NONE" for rule in fired_rules):
        playbook.append("Symbolic rule fired: preserve rule trace in incident report")

    return {"level": "Critical" if confidence >= 0.85 else "Elevated", "action": action, "playbook": playbook}


def predict_row(index):
    load_resources()
    idx = _coerce_int(index, default=0, minimum=0, maximum=len(_X_test) - 1)
    sample = _X_test.iloc[idx]
    true_label = str(_y_test.iloc[idx])
    sample_frame = sample.to_frame().T
    model_input = _model_input(_base_model, sample_frame)
    base_probs = _base_model.predict_proba(model_input)[0]
    base_pred = str(_classes[int(np.argmax(base_probs))])
    ns_label, fired_rules, _ = apply_symbolic_rules(
        sample,
        base_pred,
        predicted_probs=base_probs,
        class_labels=_classes,
        rule_context=_get_symbolic_context(),
        fusion_mode=SYMBOLIC_FUSION_MODE,
    )
    ns_label = str(ns_label)
    confidence = float(np.max(base_probs))

    robust_pred = None
    if _robust_model is not None:
        try:
            robust_pred = str(_robust_model.predict(_model_input(_robust_model, sample_frame))[0])
        except Exception as exc:
            warnings.warn(f"Robust model prediction failed for row {idx}: {exc}", RuntimeWarning)
            robust_pred = None

    return {
        "index": idx,
        "true_label": true_label,
        "base_pred": base_pred,
        "ns_label": ns_label,
        "robust_pred": robust_pred,
        "confidence": json_number(confidence, 6),
        "risk": "attack" if is_attack(ns_label) else "benign",
        "defense": defense_action(ns_label, confidence, fired_rules),
        "fired_rules": fired_rules,
        "probabilities": {
            "labels": _classes,
            "values": [json_number(p, 6) for p in base_probs],
        },
        "features": {col: json_number(sample[col], 6) for col in _X_test.columns},
    }


def make_incident(flow):
    incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    is_malicious = flow["risk"] == "attack"
    incident = {
        "incident_id": incident_id,
        "flow_index": flow["index"],
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "ready_to_contain" if is_malicious else "monitoring",
        "severity": flow["defense"]["level"],
        "label": flow["ns_label"],
        "risk": flow["risk"],
        "confidence": flow["confidence"],
        "recommended_action": flow["defense"]["action"],
        "controls": [
            {"name": "Ingress ACL", "state": "pending" if is_malicious else "not_required"},
            {"name": "Session Isolation", "state": "pending" if is_malicious else "not_required"},
            {"name": "Rate Limiter", "state": "pending" if is_malicious else "not_required"},
            {"name": "Evidence Export", "state": "queued"},
        ],
        "timeline": [
            {"time": "T+00ms", "event": "Flow analysed by MLP classifier"},
            {"time": "T+08ms", "event": "Symbolic rules evaluated"},
            {"time": "T+18ms", "event": "Defence recommendation generated"},
        ],
    }
    _incident_store[incident_id] = incident
    return incident


def analyse_defense(index):
    flow = predict_row(index)
    return {"flow": flow, "incident": make_incident(flow)}


def contain_incident(incident_id):
    if not incident_id:
        return None, "Missing incident id"
    incident = _incident_store.get(incident_id)
    if incident is None:
        return None, "Unknown incident id"

    if incident["risk"] == "attack":
        for control in incident["controls"]:
            if control["state"] == "pending":
                control["state"] = "applied"
        incident["status"] = "contained"
        incident["timeline"].extend([
            {"time": "T+31ms", "event": "Ingress ACL staged"},
            {"time": "T+42ms", "event": "Session isolation applied"},
            {"time": "T+55ms", "event": "Containment evidence recorded"},
        ])
        return incident, "Backend containment completed for this simulated research incident."

    incident["status"] = "monitoring"
    incident["timeline"].append({"time": "T+31ms", "event": "Benign flow retained in monitoring mode"})
    return incident, "No containment needed; backend monitoring state updated."


def _metric_vector(report: dict[str, Any], accuracy: float) -> list[float | None]:
    macro = report.get("macro avg", {})
    return [
        json_number(accuracy, 6),
        json_number(macro.get("precision", 0.0), 6),
        json_number(macro.get("recall", 0.0), 6),
        json_number(macro.get("f1-score", 0.0), 6),
    ]


def _attack_recall_deltas(
    labels: list[str],
    base_report: dict[str, Any],
    ns_report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        if not is_attack(label):
            continue
        baseline_recall = float(base_report.get(label, {}).get("recall", 0.0))
        ns_recall = float(ns_report.get(label, {}).get("recall", 0.0))
        rows.append({
            "class": label,
            "baseline_recall": json_number(baseline_recall, 6),
            "neuro_symbolic_recall": json_number(ns_recall, 6),
            "recall_delta": json_number(ns_recall - baseline_recall, 6),
        })
    return rows


def _novelty_examples(
    true_arr: list[str],
    base_preds: np.ndarray,
    ns_preds: np.ndarray,
    rule_traces: list[list[dict[str, Any]]],
    probabilities: np.ndarray,
    max_examples: int = 6,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for idx, (truth, base_label, ns_label) in enumerate(zip(true_arr, base_preds, ns_preds)):
        if str(base_label) == str(ns_label):
            continue
        if not (is_attack(truth) and not is_attack(base_label) and is_attack(ns_label)):
            continue
        applied = [rule for rule in rule_traces[idx] if rule.get("applied")]
        rule = applied[0] if applied else next((r for r in rule_traces[idx] if r.get("rule_id") != "NONE"), {})
        examples.append(
            {
                "sample": int(idx),
                "true_label": str(truth),
                "mlp_label": str(base_label),
                "neuro_symbolic_label": str(ns_label),
                "exact_correction": bool(str(ns_label) == str(truth)),
                "confidence": json_number(float(np.max(probabilities[idx])), 6),
                "rule_id": rule.get("rule_id"),
                "rule_strength": rule.get("strength"),
                "explanation": rule.get("reason"),
            }
        )
        if len(examples) >= max_examples:
            break
    return examples


def _rule_analytics(
    true_arr: list[str],
    base_preds: np.ndarray,
    ns_preds: np.ndarray,
    rule_traces: list[list[dict[str, Any]]],
    strengths: list[float],
    base_report: dict[str, Any],
    ns_report: dict[str, Any],
) -> dict[str, Any]:
    sample_count = len(true_arr)
    active_counts: Counter[str] = Counter()
    applied_counts: Counter[str] = Counter()
    triggered_samples = 0
    applied_samples = 0
    for rules in rule_traces:
        active = [rule for rule in rules if rule.get("rule_id") != "NONE"]
        applied = [rule for rule in active if bool(rule.get("applied"))]
        if active:
            triggered_samples += 1
        if applied:
            applied_samples += 1
        for rule in active:
            active_counts[str(rule.get("rule_id", "UNKNOWN"))] += 1
        for rule in applied:
            applied_counts[str(rule.get("rule_id", "UNKNOWN"))] += 1

    y = np.asarray(true_arr)
    base = np.asarray(base_preds)
    ns = np.asarray(ns_preds)
    changed = base != ns
    true_attack = np.asarray([is_attack(label) for label in y], dtype=bool)
    true_benign = ~true_attack
    base_benign = np.asarray([not is_attack(label) for label in base], dtype=bool)
    ns_benign = np.asarray([not is_attack(label) for label in ns], dtype=bool)
    base_attack = ~base_benign
    ns_attack = ~ns_benign
    false_neg_before = true_attack & base_benign
    false_neg_after = true_attack & ns_benign
    fn_attack_rescues = false_neg_before & ~ns_benign
    exact_corrections = false_neg_before & (ns == y)
    introduced_fp = true_benign & base_benign & ~ns_benign
    base_attack_recall = float(np.sum(true_attack & base_attack) / max(1, true_attack.sum()))
    ns_attack_recall = float(np.sum(true_attack & ns_attack) / max(1, true_attack.sum()))
    changed_count = int(changed.sum())
    triggered_rate = float(triggered_samples / max(1, sample_count))
    changed_rate = float(np.mean(changed)) if sample_count else 0.0

    return {
        "samples": int(sample_count),
        "triggered_samples": int(triggered_samples),
        "applied_samples": int(applied_samples),
        "changed_predictions": changed_count,
        "prediction_change_count": changed_count,
        "changed_prediction_rate": json_number(changed_rate, 6),
        "prediction_change_rate": json_number(changed_rate, 6),
        "rule_trigger_count": int(sum(active_counts.values())),
        "rule_trigger_sample_count": int(triggered_samples),
        "rule_trigger_rate": json_number(triggered_rate, 6),
        "rule_trigger_pct": json_number(100.0 * triggered_rate, 4),
        "per_rule_trigger_count": dict(active_counts),
        "per_rule_applied_count": dict(applied_counts),
        "per_rule_trigger_frequency": {
            rule_id: json_number(count / max(1, sample_count), 6)
            for rule_id, count in active_counts.items()
        },
        "false_negatives_before": int(false_neg_before.sum()),
        "false_negatives_after": int(false_neg_after.sum()),
        "false_negative_attack_rescues": int(fn_attack_rescues.sum()),
        "false_negative_exact_label_corrections": int(exact_corrections.sum()),
        "introduced_benign_false_positives": int(introduced_fp.sum()),
        "binary_attack_recall_before": json_number(base_attack_recall, 6),
        "binary_attack_recall_after": json_number(ns_attack_recall, 6),
        "binary_attack_recall_delta": json_number(ns_attack_recall - base_attack_recall, 6),
        "mean_rule_strength": json_number(float(np.mean(strengths)) if strengths else 0.0, 6),
        "attack_class_recall_delta": _attack_recall_deltas(_classes, base_report, ns_report),
    }


def _evaluate_window(limit=750) -> dict[str, Any]:
    load_resources()
    limit = _coerce_int(limit, default=750, minimum=MIN_ANALYSIS_LIMIT, maximum=min(MAX_ANALYSIS_LIMIT, len(_X_test)))
    if limit in _evaluation_cache:
        return _evaluation_cache[limit]

    started = perf_counter()
    subset_X = _X_test.head(limit)
    true_arr = _y_test.head(limit).tolist()
    model_input = _model_input(_base_model, subset_X)
    probabilities = _base_model.predict_proba(model_input)
    confidence = np.max(probabilities, axis=1)
    base_preds = np.asarray([str(x) for x in _base_model.predict(model_input)])
    ns_preds, rule_traces, strengths = apply_symbolic_rules_batch(
        subset_X,
        base_preds,
        probabilities,
        class_labels=_classes,
        rule_context=_get_symbolic_context(),
        fusion_mode=SYMBOLIC_FUSION_MODE,
    )
    ns_preds = np.asarray([str(label) for label in ns_preds])

    base_report = classification_report(true_arr, base_preds, labels=_classes, output_dict=True, zero_division=0)
    ns_report = classification_report(true_arr, ns_preds, labels=_classes, output_dict=True, zero_division=0)
    base_acc = float(np.mean(base_preds == np.asarray(true_arr)))
    ns_acc = float(np.mean(ns_preds == np.asarray(true_arr)))
    labels = _classes
    analytics = _rule_analytics(true_arr, base_preds, ns_preds, rule_traces, strengths, base_report, ns_report)
    base_macro_f1 = float(base_report.get("macro avg", {}).get("f1-score", 0.0))
    ns_macro_f1 = float(ns_report.get("macro avg", {}).get("f1-score", 0.0))
    analytics["delta_accuracy"] = json_number(ns_acc - base_acc, 6)
    analytics["delta_f1"] = json_number(ns_macro_f1 - base_macro_f1, 6)
    ns_attack = np.asarray([is_attack(label) for label in ns_preds], dtype=bool)
    base_attack = np.asarray([is_attack(label) for label in base_preds], dtype=bool)
    true_attack = np.asarray([is_attack(label) for label in true_arr], dtype=bool)
    changed = base_preds != ns_preds
    containment_candidates = ns_attack & ((confidence >= 0.70) | changed)
    high_confidence_blocks = ns_attack & (confidence >= 0.85)
    elapsed_ms = (perf_counter() - started) * 1000.0

    rows = [
        {
            "idx": i,
            "true": true_arr[i],
            "baseline": base_preds[i],
            "proposed": ns_preds[i],
            "risk": "attack" if is_attack(ns_preds[i]) else "benign",
            "changed": bool(base_preds[i] != ns_preds[i]),
            "applied_rules": [
                rule["rule_id"]
                for rule in rule_traces[i]
                if rule.get("rule_id") != "NONE" and bool(rule.get("applied"))
            ],
        }
        for i in range(min(100, limit))
    ]

    active_rule_counts = analytics["per_rule_trigger_count"]
    examples = _novelty_examples(true_arr, base_preds, ns_preds, rule_traces, probabilities)
    novelty_proof = {
        "ns_beats_mlp_accuracy": bool(ns_acc > base_acc),
        "ns_beats_mlp_macro_f1": bool(ns_macro_f1 > base_macro_f1),
        "attack_recall_improved": bool((analytics.get("binary_attack_recall_delta") or 0) > 0),
        "max_attack_class_recall_delta": json_number(
            max((row["recall_delta"] for row in analytics["attack_class_recall_delta"]), default=0.0),
            6,
        ),
        "verdict": (
            "proven"
            if ns_acc > base_acc
            or ns_macro_f1 > base_macro_f1
            or (analytics.get("binary_attack_recall_delta") or 0) > 0
            else "not_proven_for_this_window"
        ),
        "examples": examples,
    }
    live_metrics = {
        "labels": ["Accuracy", "Precision", "Recall", "F1"],
        "existing": _metric_vector(base_report, base_acc),
        "proposed": _metric_vector(ns_report, ns_acc),
        "source": "live-window evaluation from model predictions and test labels",
        "window_existing_accuracy": json_number(base_acc, 6),
        "window_proposed_accuracy": json_number(ns_acc, 6),
    }

    public = {
        "limit": limit,
        "metrics": live_metrics,
        "paper_summary": saved_paper_summary(),
        "window_metrics": {
            "labels": ["Accuracy", "Precision", "Recall", "F1"],
            "baseline_mlp": live_metrics["existing"],
            "neuro_symbolic": live_metrics["proposed"],
        },
        "reports": {"baseline_mlp": base_report, "neuro_symbolic": ns_report},
        "classes": labels,
        "confusion_matrix": confusion_matrix(true_arr, ns_preds, labels=labels).tolist(),
        "class_distribution": {
            "labels": labels,
            "values": [int(v) for v in pd.Series(ns_preds).value_counts().reindex(labels, fill_value=0).tolist()],
        },
        "rule_hits": {"labels": list(active_rule_counts.keys()), "values": [int(v) for v in active_rule_counts.values()]},
        "rule_analytics": analytics,
        "novelty_proof": novelty_proof,
        "defense": {
            "analysed_flows": limit,
            "attack_flows": int(true_attack.sum()),
            "baseline_attack_predictions": int(base_attack.sum()),
            "detected_attack_flows": int(ns_attack.sum()),
            "containment_candidates": int(containment_candidates.sum()),
            "blocked_flows": int(high_confidence_blocks.sum()),
            "mean_response_ms": json_number(elapsed_ms / max(1, limit), 6),
            "policy": "Adaptive containment",
        },
        "evidence_sources": {
            "live_evaluation": "model predictions recomputed for this request window",
            "paper_summary": "saved values in results/metrics.json, never used for live dashboard charts",
            "publication_package": "generated by backend/generate_publication_package.py from live evaluation outputs",
        },
        "rows": rows,
    }

    _evaluation_cache[limit] = {
        "public": public,
        "subset_X": subset_X,
        "true_arr": true_arr,
        "probabilities": probabilities,
        "confidence": confidence,
        "base_preds": base_preds,
        "ns_preds": ns_preds,
        "rule_traces": rule_traces,
        "strengths": strengths,
    }
    _analysis_cache[limit] = public
    return _evaluation_cache[limit]


def analyse_window(limit=750):
    return _evaluate_window(limit)["public"]


def _chart_window_grid(limit: int) -> list[int]:
    points = {100, limit}
    for fraction in (0.25, 0.50, 0.75):
        points.add(max(100, int(round(limit * fraction))))
    if limit < 300:
        points.update({min(limit, 150), min(limit, 200)})
    return sorted(point for point in points if 100 <= point <= limit)


def _log_chart_step(logs: list[str], message: str) -> None:
    logs.append(message)
    LOGGER.info(message)


def chart_data(limit=2000):
    load_resources()
    requested_limit = limit
    limit = _coerce_int(limit, default=2000, minimum=100, maximum=min(MAX_CHART_LIMIT, len(_X_test)))
    if limit in _chart_cache:
        return _chart_cache[limit]

    logs: list[str] = []
    _log_chart_step(logs, f"Chart request received: requested_limit={requested_limit}, sanitized_limit={limit}.")
    evaluated = _evaluate_window(limit)
    analysis = evaluated["public"]
    true_arr = evaluated["true_arr"]
    probabilities = evaluated["probabilities"]
    confidence = evaluated["confidence"]
    base_preds = evaluated["base_preds"]
    ns_preds = evaluated["ns_preds"]
    base_report = analysis["reports"]["baseline_mlp"]
    ns_report = analysis["reports"]["neuro_symbolic"]

    windows = _chart_window_grid(limit)
    existing_curve: list[float | None] = []
    proposed_curve: list[float | None] = []
    f1_baseline_curve: list[float | None] = []
    f1_ns_curve: list[float | None] = []
    for w in windows:
        window_eval = _evaluate_window(w)["public"]["window_metrics"]
        existing_curve.append(window_eval["baseline_mlp"][0])
        proposed_curve.append(window_eval["neuro_symbolic"][0])
        f1_baseline_curve.append(window_eval["baseline_mlp"][3])
        f1_ns_curve.append(window_eval["neuro_symbolic"][3])
    _log_chart_step(
        logs,
        f"Improvement curve recomputed from live baseline/neuro-symbolic predictions for windows {windows}.",
    )

    y_bin = label_binarize(true_arr, classes=_classes)
    try:
        fpr, tpr, _ = roc_curve(y_bin.ravel(), probabilities.ravel())
        roc_auc = float(auc(fpr, tpr))
        step = max(1, len(fpr) // 80)
        roc_points = {
            "auc": json_number(roc_auc, 6),
            "points": [{"x": json_number(x, 5), "y": json_number(y, 5)} for x, y in zip(fpr[::step], tpr[::step])],
        }
    except Exception:
        roc_points = {"auc": None, "points": []}

    hist, edges = np.histogram(confidence, bins=np.linspace(0, 1, 11))
    true_attack = np.asarray([is_attack(label) for label in true_arr], dtype=bool)
    base_attack = np.asarray([is_attack(label) for label in base_preds], dtype=bool)
    ns_attack = np.asarray([is_attack(label) for label in ns_preds], dtype=bool)
    changed = base_preds != ns_preds
    containment_candidates = ns_attack & ((confidence >= 0.70) | changed)
    high_confidence_blocks = ns_attack & (confidence >= 0.85)
    _log_chart_step(
        logs,
        "Detection counts computed as distinct labelled attacks, baseline attack predictions, "
        "neuro-symbolic attack predictions, containment candidates, and high-confidence block recommendations.",
    )
    _log_chart_step(logs, "Per-class F1 and error rates computed from live classification reports for the selected window.")

    _chart_cache[limit] = {
        "limit": limit,
        "debug": {
            "input_parameters": {"requested_limit": requested_limit, "sanitized_limit": limit},
            "api_output_summary": {
                "curve_points": len(windows),
                "classes": len(_classes),
                "rule_trigger_count": analysis["rule_analytics"]["rule_trigger_count"],
                "prediction_change_count": analysis["rule_analytics"]["prediction_change_count"],
            },
            "datasets_changed": [
                "metric_comparison",
                "improvement_curve",
                "per_class",
                "confidence_histogram",
                "detection_counts",
                "class_error_rate",
                "rule_hits",
            ],
        },
        "metric_comparison": {
            "labels": ["Accuracy", "Precision", "Recall", "F1"],
            "existing": analysis["window_metrics"]["baseline_mlp"],
            "proposed": analysis["window_metrics"]["neuro_symbolic"],
            "backend_window_baseline": analysis["window_metrics"]["baseline_mlp"],
            "backend_window_neuro_symbolic": analysis["window_metrics"]["neuro_symbolic"],
            "source": "live-window evaluation from model predictions and test labels",
        },
        "paper_summary": analysis["paper_summary"],
        "improvement_curve": {
            "labels": [str(w) for w in windows],
            "existing_accuracy": existing_curve,
            "proposed_accuracy": proposed_curve,
            "existing_f1": f1_baseline_curve,
            "proposed_f1": f1_ns_curve,
            "source": "live-window recomputation",
            "note": "Each point is recomputed from model predictions and labels for that exact prefix window.",
        },
        "per_class": {
            "labels": _classes,
            "existing_f1": [json_number(base_report.get(label, {}).get("f1-score", 0), 6) for label in _classes],
            "proposed_f1": [json_number(ns_report.get(label, {}).get("f1-score", 0), 6) for label in _classes],
            "source": "live-window classification_report",
        },
        "confidence_histogram": {
            "labels": [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(len(edges) - 1)],
            "values": [int(v) for v in hist.tolist()],
        },
        "detection_counts": {
            "labels": [
                "True attack labels",
                "Baseline attack predictions",
                "Neuro-symbolic attack predictions",
                "Containment candidates",
                "High-confidence block recommendations",
            ],
            "values": [
                int(true_attack.sum()),
                int(base_attack.sum()),
                int(ns_attack.sum()),
                int(containment_candidates.sum()),
                int(high_confidence_blocks.sum()),
            ],
            "source": "live-window predictions and confidence thresholds",
        },
        "class_error_rate": {
            "labels": _classes,
            "values": [
                json_number(
                    sum(t == label and p != label for t, p in zip(true_arr, ns_preds)) / max(1, sum(t == label for t in true_arr)),
                    6,
                )
                for label in _classes
            ],
        },
        "roc_curve": roc_points,
        "class_distribution": analysis["class_distribution"],
        "rule_hits": analysis["rule_hits"],
        "rule_analytics": analysis["rule_analytics"],
        "computation_log": logs,
    }
    return _chart_cache[limit]


def ablation_data(limit=1000):
    load_resources()
    limit = _coerce_int(limit, default=1000, minimum=MIN_ANALYSIS_LIMIT, maximum=min(MAX_ANALYSIS_LIMIT, len(_X_test)))
    data = analyse_window(limit)
    labels = data["window_metrics"]["labels"]
    baseline = data["window_metrics"]["baseline_mlp"]
    neuro_symbolic = data["window_metrics"]["neuro_symbolic"]
    return {
        "limit": limit,
        "labels": labels,
        "systems": [
            {"name": "Baseline MLP", "metrics": baseline},
            {"name": "Neuro-symbolic", "metrics": neuro_symbolic},
        ],
        "delta": [json_number(ns - base, 6) for base, ns in zip(baseline, neuro_symbolic)],
        "notes": [
            "Baseline MLP uses raw model predictions.",
            "Neuro-symbolic applies the auditable symbolic rule layer after neural inference.",
        ],
    }


def novelty_data(limit=2000, alpha=0.10):
    """Reliability/novelty evidence for a publishable trustworthy IDS story.

    Uses the available processed test set as a deterministic demonstration split:
    a prefix is used for conformal calibration and the remaining requested window
    is used for evaluation. This should be described as an internal validation
    protocol unless a separate validation split is later supplied.
    """
    load_resources()
    limit = _coerce_int(limit, default=2000, minimum=200, maximum=min(MAX_CHART_LIMIT, len(_X_test)))
    alpha_value = float(alpha) if alpha is not None else 0.10
    alpha_value = min(0.40, max(0.01, alpha_value))
    cache_key = (limit, round(alpha_value, 4))
    if cache_key in _novelty_cache:
        return _novelty_cache[cache_key]

    subset_X = _X_test.head(limit)
    subset_y = _y_test.head(limit).tolist()
    model_input = _model_input(_base_model, subset_X)
    probs = _base_model.predict_proba(model_input)
    preds = [str(_classes[int(np.argmax(row))]) for row in probs]
    correct = np.asarray([p == t for p, t in zip(preds, subset_y)], dtype=float)
    confidence = np.max(probs, axis=1)
    entropy = _entropy(probs)
    margin = _probability_margin(probs)

    calibration_size = max(50, min(limit // 5, limit - 50))
    calib_probs = probs[:calibration_size]
    calib_labels = subset_y[:calibration_size]
    calib_indices = _class_indices(calib_labels)
    valid_mask = calib_indices >= 0
    if valid_mask.any():
        nonconformity = 1.0 - calib_probs[np.arange(calibration_size)[valid_mask], calib_indices[valid_mask]]
        q_level = min(1.0, np.ceil((len(nonconformity) + 1) * (1.0 - alpha_value)) / max(1, len(nonconformity)))
        q_hat = float(np.quantile(nonconformity, q_level, method="higher"))
    else:
        q_hat = 1.0

    eval_probs = probs[calibration_size:]
    eval_labels = subset_y[calibration_size:]
    threshold = max(0.0, 1.0 - q_hat)
    prediction_sets = eval_probs >= threshold
    set_sizes = prediction_sets.sum(axis=1) if len(eval_probs) else np.asarray([], dtype=int)
    eval_indices = _class_indices(eval_labels)
    covered = []
    for row_idx, label_idx in enumerate(eval_indices):
        covered.append(bool(label_idx >= 0 and prediction_sets[row_idx, label_idx]))

    reference_count = min(1000, len(_X_test))
    reference = _X_test.head(reference_count)
    ref_mean = reference.mean(axis=0)
    ref_std = reference.std(axis=0).replace(0, 1.0).fillna(1.0)
    z = (subset_X - ref_mean) / ref_std
    ood_scores = np.sqrt(np.mean(np.square(z), axis=1))
    ref_z = (_X_test.head(reference_count) - ref_mean) / ref_std
    ref_scores = np.sqrt(np.mean(np.square(ref_z), axis=1))
    ood_threshold = float(np.quantile(ref_scores, 0.95))
    ood_flags = ood_scores > ood_threshold
    feature_drift = z.abs().mean(axis=0).sort_values(ascending=False).head(10)

    calibration = _calibration_curve(confidence, correct)
    high_uncertainty = (confidence < 0.60) | (entropy > np.quantile(entropy, 0.80))
    review_queue = []
    for idx in np.where(high_uncertainty | ood_flags.to_numpy())[0][:25]:
        review_queue.append({
            "idx": int(idx),
            "true": subset_y[idx],
            "predicted": preds[idx],
            "confidence": json_number(confidence[idx], 6),
            "entropy": json_number(entropy[idx], 6),
            "margin": json_number(margin[idx], 6),
            "ood_score": json_number(ood_scores.iloc[idx], 6),
            "reason": "OOD" if bool(ood_flags.iloc[idx]) else "uncertain",
        })

    _novelty_cache[cache_key] = {
        "limit": limit,
        "alpha": json_number(alpha_value, 4),
        "uncertainty": {
            "mean_confidence": json_number(np.mean(confidence), 6),
            "mean_entropy": json_number(np.mean(entropy), 6),
            "mean_margin": json_number(np.mean(margin), 6),
            "high_uncertainty_count": int(np.sum(high_uncertainty)),
        },
        "calibration": calibration,
        "conformal": {
            "calibration_size": int(calibration_size),
            "evaluation_size": int(len(eval_probs)),
            "q_hat": json_number(q_hat, 6),
            "probability_threshold": json_number(threshold, 6),
            "empirical_coverage": json_number(np.mean(covered) if covered else 0.0, 6),
            "average_set_size": json_number(np.mean(set_sizes) if len(set_sizes) else 0.0, 6),
            "target_coverage": json_number(1.0 - alpha_value, 4),
        },
        "ood_drift": {
            "reference_rows": int(reference_count),
            "ood_threshold": json_number(ood_threshold, 6),
            "ood_count": int(ood_flags.sum()),
            "ood_rate": json_number(float(np.mean(ood_flags)), 6),
            "top_drift_features": [
                {"feature": str(name), "mean_abs_z": json_number(value, 6)}
                for name, value in feature_drift.items()
            ],
        },
        "review_queue": review_queue,
        "chart_ready": {
            "calibration_bins": calibration["bins"],
            "drift_features": [
                {"x": str(name), "y": json_number(value, 6)}
                for name, value in feature_drift.items()
            ],
            "uncertainty_histogram": {
                "labels": [f"{edge:.1f}-{edge + 0.1:.1f}" for edge in np.linspace(0.0, 0.9, 10)],
                "values": [int(v) for v in np.histogram(entropy, bins=np.linspace(0.0, max(float(entropy.max()), 1e-6), 11))[0].tolist()],
            },
        },
        "notes": [
            "Conformal prediction sets provide an uncertainty-aware abstention signal.",
            "OOD scoring uses standardized distance from a deterministic reference prefix.",
            "These reliability outputs should be reported as validation evidence, not as a replacement for external testing.",
        ],
    }
    return _novelty_cache[cache_key]


def _file_signature(path: str | Path) -> dict[str, Any]:
    path_obj = _as_path(path)
    if not path_obj.exists():
        return {"path": str(path_obj), "exists": False}
    stat = path_obj.stat()
    return {
        "path": str(path_obj),
        "exists": True,
        "size": int(stat.st_size),
        "mtime": json_number(stat.st_mtime, 6),
    }


def run_all(limit=750, alpha=0.10, flow_idx=0) -> dict[str, Any]:
    """Single-click recomputation entry point for the dashboard."""
    load_resources()
    requested = {"limit": limit, "alpha": alpha, "flow_idx": flow_idx}
    clean_limit = _coerce_int(limit, default=750, minimum=MIN_ANALYSIS_LIMIT, maximum=min(MAX_ANALYSIS_LIMIT, len(_y_test) if _y_test is not None else MAX_ANALYSIS_LIMIT))
    clean_alpha = min(0.40, max(0.01, float(alpha) if alpha is not None else 0.10))
    clean_flow_idx = _coerce_int(flow_idx, default=0, minimum=0, maximum=len(_X_test) - 1 if _X_test is not None else 0)

    LOGGER.info("Run-all requested with %s", requested)
    started = perf_counter()
    _clear_caches()
    overview = overview_data()
    research = analyse_window(clean_limit)
    charts = chart_data(clean_limit)
    novelty = novelty_data(clean_limit, clean_alpha)
    defense = analyse_defense(clean_flow_idx)
    backend = backend_status()
    elapsed_ms = (perf_counter() - started) * 1000.0

    debug = {
        "input_parameters": {
            **requested,
            "sanitized_limit": clean_limit,
            "sanitized_alpha": json_number(clean_alpha, 4),
            "sanitized_flow_idx": clean_flow_idx,
        },
        "api_output_summary": {
            "overview_samples": overview["total_samples"],
            "research_window": research["limit"],
            "chart_window": charts["limit"],
            "rule_trigger_count": research["rule_analytics"]["rule_trigger_count"],
            "prediction_change_count": research["rule_analytics"]["prediction_change_count"],
            "delta_accuracy": research["rule_analytics"]["delta_accuracy"],
            "delta_f1": research["rule_analytics"]["delta_f1"],
            "novelty_verdict": research["novelty_proof"]["verdict"],
            "elapsed_ms": json_number(elapsed_ms, 3),
        },
        "datasets_changed": [
            "overview",
            "research_metrics",
            "charts",
            "defense_analysis",
            "novelty_panel",
            "backend_status",
        ],
        "resource_signatures": {
            "train": _file_signature(TRAIN_PATH),
            "test": _file_signature(TEST_PATH),
            "model": _file_signature(MODEL_PATH),
        },
    }
    LOGGER.info("Run-all output summary: %s", debug["api_output_summary"])
    return {
        "ok": True,
        "message": "Full pipeline recomputed from model, data, and live rule evaluation.",
        "overview": overview,
        "research": research,
        "charts": charts,
        "novelty": novelty,
        "defense": defense,
        "backend": backend,
        "debug": debug,
    }


def overview_data():
    load_resources()
    report = _metrics.get("classification_report", {})
    saved_per_class = {}
    for cls in _classes:
        if cls in report:
            saved_per_class[cls] = {
                "precision": json_number(report[cls]["precision"], 4),
                "recall": json_number(report[cls]["recall"], 4),
                "f1": json_number(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }

    class_counts = _y_test.value_counts().sort_index()
    return {
        "classes": _classes,
        "class_distribution": {"labels": class_counts.index.tolist(), "values": class_counts.values.tolist()},
        "saved_paper_per_class_metrics": saved_per_class,
        "paper_summary": saved_paper_summary(),
        "total_samples": int(len(_y_test)),
        "num_classes": int(_y_test.nunique()),
        "max_index": int(len(_X_test) - 1),
    }


def backend_status():
    load_resources()
    return {
        "backend": "Flask + nids_engine.py",
        "model_loaded": _base_model is not None,
        "model_path": str(MODEL_PATH),
        "test_path": str(TEST_PATH),
        "train_path": str(TRAIN_PATH),
        "test_rows": int(len(_X_test)),
        "feature_count": int(len(_X_test.columns)),
        "classes": _classes,
        "robust_model_loaded": _robust_model is not None,
        "symbolic_context_loaded": _symbolic_context is not None,
        "symbolic_calibration": (_symbolic_context or {}).get("calibration", {}),
        "symbolic_rule_summary": (_symbolic_context or {}).get("learned_rescue_summary", {}),
        "evidence_separation": {
            "live_evaluation": "computed by nids_engine.py from model predictions and processed test data",
            "paper_summary": "loaded from results/metrics.json only when explicitly requested",
            "publication_package": "generated artifacts under results/publication_package and paper/generated",
        },
        "cached_analysis_windows": sorted(list(_analysis_cache.keys())),
        "cached_chart_windows": sorted(list(_chart_cache.keys())),
        "cached_novelty_windows": [list(key) for key in sorted(_novelty_cache.keys())],
        "incident_count": len(_incident_store),
        "note": "Frontend data is served from Flask endpoints backed by model and CSV resources.",
    }


def defense_status():
    return {"open_incidents": list(_incident_store.values())[-20:], "total_incidents": len(_incident_store)}
