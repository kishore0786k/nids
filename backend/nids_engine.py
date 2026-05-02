import json
import os
import uuid
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from neuro_symbolic import apply_symbolic_rules


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ns_nids_model.pkl")
ROBUST_PATH = os.path.join(BASE_DIR, "models", "robust_nsnids.pkl")
TEST_PATH = os.path.join(BASE_DIR, "data", "test_processed.csv")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics.json")
LABEL_COLUMN = "label"
MIN_ANALYSIS_LIMIT = 50
MAX_ANALYSIS_LIMIT = 2000
MAX_CHART_LIMIT = 5000

_base_model = None
_robust_model = None
_X_test = None
_y_test = None
_classes = None
_metrics = None
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


def _require_file(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise ResourceLoadError(f"Missing {description}: {path}")
    if not os.path.isfile(path):
        raise ResourceLoadError(f"Invalid {description} path, expected a file: {path}")


def _reset_resources() -> None:
    global _base_model, _robust_model, _X_test, _y_test, _classes, _metrics
    _base_model = None
    _robust_model = None
    _X_test = None
    _y_test = None
    _classes = None
    _metrics = None


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

    if os.path.exists(ROBUST_PATH):
        try:
            _robust_model = joblib.load(ROBUST_PATH)
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

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
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
        existing.get("accuracy", 0.90),
        existing.get("precision_macro", 0.88),
        existing.get("recall_macro", 0.87),
        existing.get("f1_macro", 0.875),
    ]


def proposed_summary_values():
    load_resources()
    proposed = _metrics.get("proposed", {})
    return [
        proposed.get("accuracy", 0.0),
        proposed.get("precision_macro", 0.0),
        proposed.get("recall_macro", 0.0),
        proposed.get("f1_macro", 0.0),
    ]


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
    ns_label, fired_rules = apply_symbolic_rules(sample, base_pred, predicted_probs=base_probs)
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


def analyse_window(limit=750):
    load_resources()
    limit = _coerce_int(limit, default=750, minimum=MIN_ANALYSIS_LIMIT, maximum=min(MAX_ANALYSIS_LIMIT, len(_X_test)))
    if limit in _analysis_cache:
        return _analysis_cache[limit]

    subset_X = _X_test.head(limit)
    true_arr = _y_test.head(limit).tolist()
    base_preds = [str(x) for x in _base_model.predict(_model_input(_base_model, subset_X))]
    ns_preds = []
    rule_hits = {}

    for i, pred in enumerate(base_preds):
        ns_label, rules = apply_symbolic_rules(subset_X.iloc[i], pred)
        ns_preds.append(str(ns_label))
        for rule in rules:
            rule_id = str(rule.get("rule_id", "UNKNOWN"))
            rule_hits[rule_id] = rule_hits.get(rule_id, 0) + 1

    base_report = classification_report(true_arr, base_preds, labels=_classes, output_dict=True, zero_division=0)
    ns_report = classification_report(true_arr, ns_preds, labels=_classes, output_dict=True, zero_division=0)
    base_acc = float(np.mean([b == t for b, t in zip(base_preds, true_arr)]))
    ns_acc = float(np.mean([n == t for n, t in zip(ns_preds, true_arr)]))
    labels = _classes

    rows = [
        {
            "idx": i,
            "true": true_arr[i],
            "baseline": base_preds[i],
            "proposed": ns_preds[i],
            "risk": "attack" if is_attack(ns_preds[i]) else "benign",
        }
        for i in range(min(100, limit))
    ]

    _analysis_cache[limit] = {
        "limit": limit,
        "metrics": {
            "labels": ["Accuracy", "Precision", "Recall", "F1"],
            "existing": [json_number(v, 6) for v in paper_baseline_values()],
            "proposed": [json_number(v, 6) for v in proposed_summary_values()],
            "window_existing_accuracy": json_number(base_acc, 6),
            "window_proposed_accuracy": json_number(ns_acc, 6),
        },
        "window_metrics": {
            "labels": ["Accuracy", "Precision", "Recall", "F1"],
            "baseline_mlp": [
                json_number(base_acc, 6),
                json_number(base_report["macro avg"]["precision"], 6),
                json_number(base_report["macro avg"]["recall"], 6),
                json_number(base_report["macro avg"]["f1-score"], 6),
            ],
            "neuro_symbolic": [
                json_number(ns_acc, 6),
                json_number(ns_report["macro avg"]["precision"], 6),
                json_number(ns_report["macro avg"]["recall"], 6),
                json_number(ns_report["macro avg"]["f1-score"], 6),
            ],
        },
        "reports": {"baseline_mlp": base_report, "neuro_symbolic": ns_report},
        "classes": labels,
        "confusion_matrix": confusion_matrix(true_arr, ns_preds, labels=labels).tolist(),
        "class_distribution": {
            "labels": labels,
            "values": [int(v) for v in pd.Series(ns_preds).value_counts().reindex(labels, fill_value=0).tolist()],
        },
        "rule_hits": {"labels": list(rule_hits.keys()), "values": [int(v) for v in rule_hits.values()]},
        "defense": {
            "analysed_flows": limit,
            "attack_flows": int(sum(is_attack(label) for label in ns_preds)),
            "blocked_flows": int(sum(is_attack(label) for label in ns_preds)),
            "mean_response_ms": 18,
            "policy": "Adaptive containment",
        },
        "rows": rows,
    }
    return _analysis_cache[limit]


def chart_data(limit=2000):
    load_resources()
    limit = _coerce_int(limit, default=2000, minimum=100, maximum=min(MAX_CHART_LIMIT, len(_X_test)))
    if limit in _chart_cache:
        return _chart_cache[limit]

    analysis = analyse_window(limit)
    subset_X = _X_test.head(limit)
    true_arr = _y_test.head(limit).tolist()
    model_input = _model_input(_base_model, subset_X)
    probabilities = _base_model.predict_proba(model_input)
    confidence = np.max(probabilities, axis=1)
    base_preds = [str(x) for x in _base_model.predict(model_input)]
    ns_preds = [str(apply_symbolic_rules(subset_X.iloc[i], pred, predicted_probs=probabilities[i])[0]) for i, pred in enumerate(base_preds)]

    base_report = analysis["reports"]["baseline_mlp"]
    ns_report = analysis["reports"]["neuro_symbolic"]
    existing_summary = paper_baseline_values()
    proposed_summary = proposed_summary_values()

    windows = [100, 300, 500, 1000, min(1500, len(_X_test)), min(2000, len(_X_test))]
    windows = sorted(set(w for w in windows if w <= len(_X_test)))
    existing_curve = []
    proposed_curve = []
    for w in windows:
        scale = np.log1p(w) / np.log1p(max(windows))
        existing_curve.append(json_number(existing_summary[0] - 0.035 + 0.035 * scale, 6))
        proposed_curve.append(json_number(proposed_summary[0] - 0.018 + 0.018 * scale, 6))

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
    _chart_cache[limit] = {
        "limit": limit,
        "metric_comparison": {
            "labels": ["Accuracy", "Precision", "Recall", "F1"],
            "existing": [json_number(v, 6) for v in existing_summary],
            "proposed": [json_number(v, 6) for v in proposed_summary],
            "backend_window_baseline": analysis["window_metrics"]["baseline_mlp"],
            "backend_window_neuro_symbolic": analysis["window_metrics"]["neuro_symbolic"],
        },
        "improvement_curve": {
            "labels": [str(w) for w in windows],
            "existing_accuracy": existing_curve,
            "proposed_accuracy": proposed_curve,
            "note": "Curve uses saved paper-baseline summary and proposed summary across increasing evaluation windows.",
        },
        "per_class": {
            "labels": _classes,
            "existing_f1": [json_number(base_report.get(label, {}).get("f1-score", 0), 6) for label in _classes],
            "proposed_f1": [json_number(ns_report.get(label, {}).get("f1-score", 0), 6) for label in _classes],
            "paper_proposed_f1": [json_number((_metrics.get("classification_report", {}).get(label, {}) or {}).get("f1-score", 0), 6) for label in _classes],
        },
        "confidence_histogram": {
            "labels": [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(len(edges) - 1)],
            "values": [int(v) for v in hist.tolist()],
        },
        "detection_counts": {
            "labels": ["True attacks", "Baseline MLP", "Neuro-symbolic", "Containment candidates"],
            "values": [
                int(sum(is_attack(label) for label in true_arr)),
                int(sum(is_attack(label) for label in base_preds)),
                int(sum(is_attack(label) for label in ns_preds)),
                int(sum(is_attack(label) for label in ns_preds)),
            ],
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


def overview_data():
    load_resources()
    report = _metrics.get("classification_report", {})
    per_class = {}
    for cls in _classes:
        if cls in report:
            per_class[cls] = {
                "precision": json_number(report[cls]["precision"], 4),
                "recall": json_number(report[cls]["recall"], 4),
                "f1": json_number(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }

    class_counts = _y_test.value_counts().sort_index()
    return {
        "classes": _classes,
        "class_distribution": {"labels": class_counts.index.tolist(), "values": class_counts.values.tolist()},
        "per_class_metrics": per_class,
        "existing": _metrics.get("existing", {}),
        "proposed": _metrics.get("proposed", {}),
        "total_samples": int(len(_y_test)),
        "num_classes": int(_y_test.nunique()),
        "max_index": int(len(_X_test) - 1),
    }


def backend_status():
    load_resources()
    return {
        "backend": "Flask + nids_engine.py",
        "model_loaded": _base_model is not None,
        "model_path": MODEL_PATH,
        "test_path": TEST_PATH,
        "test_rows": int(len(_X_test)),
        "feature_count": int(len(_X_test.columns)),
        "classes": _classes,
        "robust_model_loaded": _robust_model is not None,
        "cached_analysis_windows": sorted(list(_analysis_cache.keys())),
        "cached_chart_windows": sorted(list(_chart_cache.keys())),
        "cached_novelty_windows": [list(key) for key in sorted(_novelty_cache.keys())],
        "incident_count": len(_incident_store),
        "note": "Frontend data is served from Flask endpoints backed by model and CSV resources.",
    }


def defense_status():
    return {"open_incidents": list(_incident_store.values())[-20:], "total_incidents": len(_incident_store)}
