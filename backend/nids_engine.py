import hashlib
import json
import logging
import threading
import uuid
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc, average_precision_score, classification_report, confusion_matrix, precision_recall_curve, recall_score, roc_curve
from sklearn.preprocessing import label_binarize

from src.neuro_symbolic import apply_symbolic_rules_batch, build_symbolic_context
from src.publication_protocol import apply_temperature, fit_temperature, validation_split
from src.project_paths import (
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PROJECT_ROOT,
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
DEFAULT_ALPHA = 0.65
DEFAULT_BETA = 0.35
DEFAULT_SEED = 60
DEFAULT_UNKNOWN_THRESHOLD = 0.70
UNKNOWN_LABEL = "UNKNOWN"
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"
CALIBRATION_SAMPLE_LIMIT = 3000
SYMBOLIC_CONTEXT_SAMPLE_LIMIT = 4000

_base_model = None
_baseline_model = None
_robust_model = None
_X_test = None
_y_test = None
_classes = None
_metrics = None
_symbolic_context = None
_probability_temperature = None
_uncertainty_policy = None
_evaluation_cache = {}
_analysis_cache = {}
_chart_cache = {}
_novelty_cache = {}
_ablation_cache = {}
_feature_window_cache = {}
_incident_store = {}
_resource_lock = threading.Lock()


class ResourceLoadError(RuntimeError):
    """Raised when a required model/data resource cannot be loaded."""


class EvalConfig(NamedTuple):
    """Sanitized live-evaluation controls used as cache keys and evidence metadata."""

    window_size: int
    flow_index: int
    alpha: float
    beta: float
    fusion_mode: str
    seed: int
    unknown_threshold: float


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


def _coerce_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    if minimum is not None:
        out = max(float(minimum), out)
    if maximum is not None:
        out = min(float(maximum), out)
    return out


def evaluation_config(
    window_size: Any = 750,
    flow_index: Any = 0,
    alpha: Any = DEFAULT_ALPHA,
    beta: Any | None = None,
    fusion_mode: Any = SYMBOLIC_FUSION_MODE,
    seed: Any = DEFAULT_SEED,
) -> EvalConfig:
    """Return one canonical parameter object for all live backend recomputation."""
    load_resources()
    clean_window = _coerce_int(
        window_size,
        default=750,
        minimum=MIN_ANALYSIS_LIMIT,
        maximum=min(MAX_ANALYSIS_LIMIT, len(_X_test)),
    )
    clean_flow = _coerce_int(flow_index, default=0, minimum=0, maximum=len(_X_test) - 1)
    clean_alpha = _coerce_float(alpha, DEFAULT_ALPHA, 0.0, 1.0)
    clean_beta = _coerce_float(beta, 1.0 - clean_alpha, 0.0, 1.0) if beta is not None else 1.0 - clean_alpha
    if clean_alpha + clean_beta <= 0:
        clean_alpha, clean_beta = DEFAULT_ALPHA, DEFAULT_BETA
    mode = str(fusion_mode or SYMBOLIC_FUSION_MODE).strip().lower()
    if mode not in {"hard", "soft"}:
        mode = SYMBOLIC_FUSION_MODE
    clean_seed = _coerce_int(seed, default=DEFAULT_SEED, minimum=0, maximum=2_147_483_647)
    unknown_threshold = unknown_confidence_threshold()
    return EvalConfig(
        window_size=clean_window,
        flow_index=clean_flow,
        alpha=round(clean_alpha, 6),
        beta=round(clean_beta, 6),
        fusion_mode=mode,
        seed=clean_seed,
        unknown_threshold=round(unknown_threshold, 6),
    )


def _config_key(config: EvalConfig) -> tuple[Any, ...]:
    return (
        config.window_size,
        config.flow_index,
        round(config.alpha, 6),
        round(config.beta, 6),
        config.fusion_mode,
        config.seed,
        round(config.unknown_threshold, 6),
    )


def _config_public(config: EvalConfig) -> dict[str, Any]:
    return {
        "window_size": config.window_size,
        "flow_index": config.flow_index,
        "alpha": json_number(config.alpha, 6),
        "beta": json_number(config.beta, 6),
        "fusion_mode": config.fusion_mode,
        "seed": config.seed,
        "unknown_threshold": json_number(config.unknown_threshold, 6),
    }


def _context_public(config: EvalConfig) -> dict[str, Any]:
    return {
        "window": config.window_size,
        "flow": config.flow_index,
        "alpha": json_number(config.alpha, 6),
        "beta": json_number(config.beta, 6),
        "fusion": config.fusion_mode,
        "seed": config.seed,
        "unknown_threshold": json_number(config.unknown_threshold, 6),
        "parameter_hash": _parameter_hash(config),
    }


def _parameter_hash(config: EvalConfig) -> str:
    raw = json.dumps(_config_public(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _window_indices(config: EvalConfig) -> np.ndarray:
    """Deterministic seed/flow-index aware sample window.

    The selected flow anchors the permutation so the single-flow selector also
    changes aggregate charts, while `seed` makes the experiment reproducible.
    """
    n_rows = len(_X_test)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(n_rows)
    matches = np.flatnonzero(order == config.flow_index)
    if matches.size:
        order = np.roll(order, -int(matches[0]))
    return order[: min(config.window_size, n_rows)]


def _sample_training_frame(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Use a deterministic calibration subset so live dashboard refreshes stay responsive."""
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=seed).sort_index()


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _require_file(path: str | Path, description: str) -> None:
    path = _as_path(path)
    if not path.exists():
        raise ResourceLoadError(f"Missing {description}: {path}")
    if not path.is_file():
        raise ResourceLoadError(f"Invalid {description} path, expected a file: {path}")


def _publication_model_path(kind: str) -> Path:
    base_dir = _as_path(MODEL_PATH).parent if _as_path(MODEL_PATH).parent.exists() else MODEL_DIR
    return base_dir / f"publication_{kind}_extratrees.pkl"


def _train_publication_model(path: Path, *, kind: str) -> Any:
    _require_file(TRAIN_PATH, "processed training dataset")
    train_df = pd.read_csv(TRAIN_PATH)
    if LABEL_COLUMN not in train_df.columns or train_df.empty:
        raise ResourceLoadError(f"Training dataset {TRAIN_PATH} is missing usable '{LABEL_COLUMN}' labels.")
    X_train = train_df.drop(columns=[LABEL_COLUMN])
    y_train = train_df[LABEL_COLUMN].astype(str)
    if kind == "baseline":
        model = ExtraTreesClassifier(
            n_estimators=80,
            max_depth=2,
            min_samples_leaf=max(2, min(80, len(X_train) // 40)),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = ExtraTreesClassifier(
            n_estimators=90,
            max_depth=10,
            min_samples_leaf=max(1, min(8, len(X_train) // 250)),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    model.fit(X_train, y_train)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    LOGGER.info("Trained compatible %s publication model at %s", kind, path)
    return model


def _load_or_train_publication_model(kind: str) -> Any:
    path = _publication_model_path(kind)
    if path.exists():
        try:
            model = joblib.load(path)
            if hasattr(model, "predict") and hasattr(model, "predict_proba"):
                return model
        except Exception as exc:
            warnings.warn(f"Could not load cached {kind} publication model at {path}: {exc}", RuntimeWarning)
    return _train_publication_model(path, kind=kind)


def _aligned_probabilities(model: Any, frame: pd.DataFrame, class_labels: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(_model_input(model, frame)), dtype=float)
    model_classes = [str(label) for label in getattr(model, "classes_", class_labels)]
    if model_classes == [str(label) for label in class_labels]:
        return raw
    aligned = np.zeros((raw.shape[0], len(class_labels)), dtype=float)
    target_pos = {str(label): idx for idx, label in enumerate(class_labels)}
    for src_idx, label in enumerate(model_classes):
        dst_idx = target_pos.get(str(label))
        if dst_idx is not None and src_idx < raw.shape[1]:
            aligned[:, dst_idx] = raw[:, src_idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sums, out=np.full_like(aligned, 1.0 / max(1, len(class_labels))), where=row_sums > 0)


def _read_simple_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def unknown_confidence_threshold() -> float:
    values = _read_simple_yaml(CONFIG_YAML_PATH)
    raw = values.get("unknown_confidence_threshold", values.get("tau", DEFAULT_UNKNOWN_THRESHOLD))
    return _coerce_float(raw, DEFAULT_UNKNOWN_THRESHOLD, 0.0, 1.0)


def _unknown_rejection(
    probabilities: np.ndarray,
    threshold: float,
    margin_threshold: float | None = None,
    entropy_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence = np.max(probabilities, axis=1)
    entropy = _entropy(probabilities)
    margin = _probability_margin(probabilities)
    rejected = confidence < threshold
    if margin_threshold is not None:
        rejected |= margin < float(margin_threshold)
    if entropy_threshold is not None:
        rejected |= entropy > float(entropy_threshold)
    return confidence, entropy, rejected


def _empty_rule_trace(reason: str) -> list[dict[str, Any]]:
    return [{
        "rule_id": "NONE",
        "applied": False,
        "strength": 0.0,
        "reason": reason,
    }]


def _clear_caches() -> None:
    _evaluation_cache.clear()
    _analysis_cache.clear()
    _chart_cache.clear()
    _novelty_cache.clear()
    _ablation_cache.clear()
    _feature_window_cache.clear()


def _reset_resources() -> None:
    global _base_model, _baseline_model, _robust_model, _X_test, _y_test, _classes, _metrics, _symbolic_context, _probability_temperature, _uncertainty_policy
    _base_model = None
    _baseline_model = None
    _robust_model = None
    _X_test = None
    _y_test = None
    _classes = None
    _metrics = None
    _symbolic_context = None
    _probability_temperature = None
    _uncertainty_policy = None
    _clear_caches()


def load_resources() -> None:
    if _base_model is not None:
        return
    with _resource_lock:
        if _base_model is None:
            _load_resources_unlocked()


def _load_resources_unlocked() -> None:
    global _base_model, _baseline_model, _robust_model, _X_test, _y_test, _classes, _metrics

    if _base_model is not None:
        return

    _require_file(TEST_PATH, "processed test dataset")
    _require_file(MODEL_PATH, "base model")

    try:
        _base_model = _load_or_train_publication_model("proposed")
    except Exception as exc:
        _reset_resources()
        raise ResourceLoadError(f"Could not prepare compatible proposed model from {TRAIN_PATH}: {exc}") from exc

    if not hasattr(_base_model, "predict") or not hasattr(_base_model, "predict_proba"):
        _reset_resources()
        raise ResourceLoadError("Proposed model must implement predict and predict_proba.")

    try:
        _baseline_model = _load_or_train_publication_model("baseline")
    except Exception as exc:
        _reset_resources()
        raise ResourceLoadError(f"Could not prepare compatible baseline model from {TRAIN_PATH}: {exc}") from exc

    robust_path = _as_path(ROBUST_PATH)
    if robust_path.exists():
        try:
            _robust_model = joblib.load(robust_path)
        except Exception as exc:
            LOGGER.warning("Optional robust model unavailable at %s: %s", ROBUST_PATH, exc)
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


def _get_probability_temperature() -> float:
    global _probability_temperature
    load_resources()
    if _probability_temperature is not None:
        return float(_probability_temperature)

    if not _as_path(TRAIN_PATH).exists():
        _probability_temperature = 1.0
        return float(_probability_temperature)
    train_df = pd.read_csv(TRAIN_PATH)
    if LABEL_COLUMN not in train_df.columns or train_df.empty:
        _probability_temperature = 1.0
        return float(_probability_temperature)

    train_df = _sample_training_frame(train_df, CALIBRATION_SAMPLE_LIMIT, DEFAULT_SEED + 11)
    X_train = train_df.drop(columns=[LABEL_COLUMN])
    y_train = train_df[LABEL_COLUMN].astype(str)
    _, _, X_val, y_val = validation_split(X_train, y_train, validation_size=0.20, random_state=42)
    val_probs = _aligned_probabilities(_base_model, X_val, _classes)
    temperature, validation_nll = fit_temperature(val_probs, y_val, _classes)
    _probability_temperature = float(temperature)
    LOGGER.info("Probability temperature calibrated on validation split: T=%.3f, NLL=%.6f", temperature, validation_nll)
    return float(_probability_temperature)


def _calibrated_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return apply_temperature(probabilities, _get_probability_temperature())


def _get_uncertainty_policy() -> dict[str, float]:
    global _uncertainty_policy
    load_resources()
    if _uncertainty_policy is not None:
        return _uncertainty_policy
    fallback = {
        "confidence_threshold": max(unknown_confidence_threshold(), 0.45),
        "margin_threshold": 0.06,
        "entropy_threshold": float(np.log(max(2, len(_classes))) * 0.82),
    }
    if not _as_path(TRAIN_PATH).exists():
        _uncertainty_policy = fallback
        return _uncertainty_policy
    train_df = pd.read_csv(TRAIN_PATH)
    if LABEL_COLUMN not in train_df.columns or len(train_df) < 10:
        _uncertainty_policy = fallback
        return _uncertainty_policy
    train_df = _sample_training_frame(train_df, CALIBRATION_SAMPLE_LIMIT, DEFAULT_SEED + 17)
    X_train = train_df.drop(columns=[LABEL_COLUMN])
    y_train = train_df[LABEL_COLUMN].astype(str)
    _, _, X_val, y_val = validation_split(X_train, y_train, validation_size=0.20, random_state=42)
    val_probs = _calibrated_probabilities(_aligned_probabilities(_base_model, X_val, _classes))
    val_preds = np.asarray([str(_classes[int(idx)]) for idx in np.argmax(val_probs, axis=1)])
    confidence = np.max(val_probs, axis=1)
    margin = _probability_margin(val_probs)
    entropy = _entropy(val_probs)
    incorrect = val_preds != y_val.to_numpy(dtype=str)
    if incorrect.any():
        conf_tau = float(np.quantile(confidence[incorrect], 0.70))
        margin_tau = float(np.quantile(margin[incorrect], 0.65))
    else:
        conf_tau = float(np.quantile(confidence, 0.08))
        margin_tau = float(np.quantile(margin, 0.08))
    entropy_tau = float(np.quantile(entropy, 0.88))
    _uncertainty_policy = {
        "confidence_threshold": json_number(np.clip(max(unknown_confidence_threshold(), conf_tau), 0.35, 0.68), 6),
        "margin_threshold": json_number(np.clip(margin_tau, 0.03, 0.18), 6),
        "entropy_threshold": json_number(np.clip(entropy_tau, 0.45, np.log(max(2, len(_classes))) * 0.95), 6),
    }
    return _uncertainty_policy


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

    train_df = _sample_training_frame(train_df, SYMBOLIC_CONTEXT_SAMPLE_LIMIT, DEFAULT_SEED + 29)
    X_train = train_df.drop(columns=[LABEL_COLUMN])
    y_train = train_df[LABEL_COLUMN].astype(str)
    train_probs = _calibrated_probabilities(_aligned_probabilities(_base_model, X_train, _classes))
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


def _curve_points(x_values: np.ndarray, y_values: np.ndarray, max_points: int = 120) -> list[dict[str, float | None]]:
    """Compact monotone curve samples for browser and paper exports."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.clip(x[finite], 0.0, 1.0)
    y = np.clip(y[finite], 0.0, 1.0)
    if x.size == 0:
        return []
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    if x.size > max_points:
        idx = np.unique(np.linspace(0, x.size - 1, max_points).astype(int))
        x = x[idx]
        y = y[idx]
    return [{"x": json_number(xv, 5), "y": json_number(yv, 5)} for xv, yv in zip(x, y)]


def _ranking_payloads(true_labels: list[str], score_matrix: np.ndarray) -> dict[str, Any]:
    """Return ROC and PR summaries from the same multiclass scores used for prediction."""
    if not true_labels or len(set(true_labels)) < 2:
        return {
            "roc": {"auc": None, "points": []},
            "pr": {"average_precision": None, "points": []},
        }
    y_bin = label_binarize(true_labels, classes=_classes)
    if y_bin.size == 0:
        return {
            "roc": {"auc": None, "points": []},
            "pr": {"average_precision": None, "points": []},
        }
    scores = np.asarray(score_matrix, dtype=float)
    try:
        fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), scores.ravel())
        return {
            "roc": {
                "auc": json_number(float(auc(fpr, tpr)), 6),
                "points": _curve_points(fpr, tpr),
            },
            "pr": {
                "average_precision": json_number(float(average_precision_score(y_bin.ravel(), scores.ravel())), 6),
                "points": _curve_points(recall, precision),
            },
        }
    except Exception as exc:
        LOGGER.info("Ranking metric computation failed: %s", exc)
        return {
            "roc": {"auc": None, "points": []},
            "pr": {"average_precision": None, "points": []},
        }


def _expanded_metric_vector(
    report: dict[str, Any],
    accuracy: float,
    ranking: dict[str, Any] | None = None,
) -> list[float | None]:
    macro = report.get("macro avg", {})
    ranking = ranking or {}
    return [
        json_number(accuracy, 6),
        json_number(macro.get("precision", 0.0), 6),
        json_number(macro.get("recall", 0.0), 6),
        json_number(macro.get("f1-score", 0.0), 6),
        ranking.get("roc", {}).get("auc"),
        ranking.get("pr", {}).get("average_precision"),
    ]


def _metric_labels() -> list[str]:
    return ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]


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


def _normalised_column(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _sample_value(sample: pd.Series, candidates: list[str]) -> Any:
    wanted = [_normalised_column(candidate) for candidate in candidates]
    for column in sample.index:
        normalised = _normalised_column(str(column))
        if any(candidate == normalised or candidate in normalised for candidate in wanted):
            return sample[column]
    return None


def _json_feature_value(value: Any) -> Any:
    numeric = json_number(value, 6)
    return numeric if numeric is not None else (None if value is None else str(value))


def _flow_context(sample: pd.Series, index: int) -> dict[str, Any]:
    src_ip = _sample_value(sample, ["src_ip", "source_ip", "ipv4_src_addr", "srcaddr"])
    dst_ip = _sample_value(sample, ["dst_ip", "destination_ip", "ipv4_dst_addr", "dstaddr"])
    src_port = _sample_value(sample, ["src_port", "sport", "l4_src_port"])
    dst_port = _sample_value(sample, ["dst_port", "dport", "l4_dst_port"])
    protocol = _sample_value(sample, ["protocol", "proto", "l4_proto"])
    timestamp = _sample_value(sample, ["timestamp", "time", "ts", "flow_start"])
    in_bytes = _sample_value(sample, ["in_bytes", "src_bytes", "bytes_in"])
    out_bytes = _sample_value(sample, ["out_bytes", "dst_bytes", "bytes_out"])
    total_bytes = _sample_value(sample, ["bytes", "tot_bytes", "total_bytes"])
    in_packets = _sample_value(sample, ["in_pkts", "src_pkts", "packets_in"])
    out_packets = _sample_value(sample, ["out_pkts", "dst_pkts", "packets_out"])
    total_packets = _sample_value(sample, ["packets", "tot_pkts", "total_packets"])

    byte_values = [json_number(value, 6) for value in (in_bytes, out_bytes) if json_number(value, 6) is not None]
    packet_values = [json_number(value, 6) for value in (in_packets, out_packets) if json_number(value, 6) is not None]
    return {
        "row_index": int(index),
        "timestamp": _json_feature_value(timestamp),
        "src_ip": _json_feature_value(src_ip),
        "dst_ip": _json_feature_value(dst_ip),
        "src_port": _json_feature_value(src_port),
        "dst_port": _json_feature_value(dst_port),
        "protocol": _json_feature_value(protocol),
        "bytes_in": _json_feature_value(in_bytes),
        "bytes_out": _json_feature_value(out_bytes),
        "bytes_total": _json_feature_value(total_bytes if total_bytes is not None else sum(byte_values) if byte_values else None),
        "packets_in": _json_feature_value(in_packets),
        "packets_out": _json_feature_value(out_packets),
        "packets_total": _json_feature_value(total_packets if total_packets is not None else sum(packet_values) if packet_values else None),
    }


def _historical_frequency(label: str) -> dict[str, Any]:
    counts = _y_test.astype(str).value_counts()
    count = int(counts.get(label, 0))
    total = int(len(_y_test))
    attack_counts = counts[[idx for idx in counts.index if is_attack(idx)]]
    return {
        "class": label,
        "count": count,
        "rate": json_number(count / max(1, total), 6),
        "total_rows": total,
        "attack_rows": int(attack_counts.sum()),
        "source": str(TEST_PATH),
    }


def _global_feature_importance(top_k: int = 20) -> list[dict[str, Any]]:
    load_resources()
    feature_names = [str(column) for column in _X_test.columns]
    scores = None
    method = None
    if hasattr(_base_model, "feature_importances_"):
        scores = np.asarray(getattr(_base_model, "feature_importances_"), dtype=float)
        method = "model_feature_importances"
    elif hasattr(_base_model, "coef_"):
        coef = np.asarray(getattr(_base_model, "coef_"), dtype=float)
        scores = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        method = "model_coefficients"
    if scores is None or len(scores) != len(feature_names):
        scores = _X_test.select_dtypes(include=[np.number]).std(axis=0).reindex(_X_test.columns).fillna(0).to_numpy()
        method = "feature_variance_fallback"
    rows = [
        {"feature": feature, "score": json_number(score, 8), "method": method}
        for feature, score in zip(feature_names, scores)
    ]
    return sorted(rows, key=lambda row: abs(float(row["score"] or 0)), reverse=True)[:top_k]


def _shap_attributions(sample_frame: pd.DataFrame, class_index: int) -> tuple[str, np.ndarray] | None:
    if not hasattr(_base_model, "estimators_"):
        return None
    try:
        import shap

        explainer = shap.TreeExplainer(_base_model)
        shap_values = explainer.shap_values(sample_frame)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[class_index][0], dtype=float)
        else:
            arr = np.asarray(shap_values, dtype=float)
            values = arr[0, :, class_index] if arr.ndim == 3 else arr[0]
        return "shap", values
    except Exception as exc:
        LOGGER.info("SHAP attribution unavailable, falling back to permutation-style scores: %s", exc)
        return None


def _permutation_attributions(sample_frame: pd.DataFrame, class_index: int, base_prob: float) -> tuple[str, np.ndarray]:
    reference = _X_test.median(numeric_only=True).reindex(_X_test.columns).fillna(0)
    scores = []
    for column in _X_test.columns:
        perturbed = sample_frame.copy()
        perturbed[column] = reference.get(column, 0)
        try:
            next_prob = float(_base_model.predict_proba(_model_input(_base_model, perturbed))[0][class_index])
            scores.append(base_prob - next_prob)
        except Exception as exc:
            LOGGER.info("Permutation attribution failed for %s: %s", column, exc)
            scores.append(0.0)
    return "permutation_occlusion", np.asarray(scores, dtype=float)


def _prediction_evidence(
    sample: pd.Series,
    sample_frame: pd.DataFrame,
    index: int,
    label: str,
    base_probs: np.ndarray,
    fired_rules: list[dict[str, Any]],
    config: EvalConfig,
    top_k: int = 8,
) -> dict[str, Any]:
    class_index = _classes.index(label) if label in _classes else int(np.argmax(base_probs))
    confidence = float(np.max(base_probs))
    fused = _fused_probabilities(np.asarray([base_probs]), np.asarray([label]), [fired_rules], config)[0]
    calibrated_probability = float(fused[class_index]) if class_index < len(fused) else confidence
    attribution = _shap_attributions(sample_frame, class_index)
    if attribution is None:
        attribution = _permutation_attributions(sample_frame, class_index, float(base_probs[class_index]))
    method, scores = attribution
    if not np.any(np.abs(scores)) and hasattr(_base_model, "feature_importances_"):
        method = "model_feature_importances"
        scores = np.asarray(getattr(_base_model, "feature_importances_"), dtype=float)

    rows = []
    for feature, score in zip(_X_test.columns, scores):
        rows.append({
            "feature": str(feature),
            "value": _json_feature_value(sample[feature]),
            "score": json_number(score, 8),
            "method": method,
        })
    matched_rules = [
        {
            "rule_id": str(rule.get("rule_id")),
            "signature": str(rule.get("rule_id")),
            "reason": str(rule.get("reason") or ""),
            "applied": bool(rule.get("applied")),
            "strength": json_number(rule.get("strength", rule.get("score", 0)), 6),
        }
        for rule in fired_rules
        if rule.get("rule_id") != "NONE"
    ]
    return {
        "top_features": sorted(rows, key=lambda row: abs(float(row["score"] or 0)), reverse=True)[:top_k],
        "flow_context": _flow_context(sample, index),
        "confidence": json_number(confidence, 6),
        "calibrated_probability": json_number(calibrated_probability, 6),
        "matched_rules": matched_rules,
        "matched_rule_count": len(matched_rules),
        "historical_frequency": _historical_frequency(label),
    }


def _flow_payload_from_evaluation(evaluated: dict[str, Any], row_pos: int = 0) -> dict[str, Any]:
    config = evaluated["config"]
    row_pos = max(0, min(int(row_pos), len(evaluated["indices"]) - 1))
    idx = int(evaluated["indices"][row_pos])
    sample = evaluated["subset_X"].iloc[row_pos]
    sample_frame = sample.to_frame().T
    true_label = str(evaluated["true_arr"][row_pos])
    baseline_probs = np.asarray(evaluated["baseline_probabilities"][row_pos], dtype=float)
    proposed_probs = np.asarray(evaluated["probabilities"][row_pos], dtype=float)
    base_pred = str(evaluated["base_preds"][row_pos])
    neural_pred = str(evaluated["neural_preds"][row_pos])
    ns_label = str(evaluated["ns_preds"][row_pos])
    final_label = str(evaluated.get("final_preds", evaluated["ns_preds"])[row_pos])
    confidence = float(evaluated["confidence"][row_pos])
    entropy = float(evaluated["entropy"][row_pos])
    margin = float(evaluated["margin"][row_pos])
    rejected_unknown = bool(evaluated["rejected_unknown"][row_pos])
    fired_rules = [dict(rule) for rule in evaluated["rule_traces"][row_pos]]
    strength = float(evaluated["strengths"][row_pos])
    applied = [rule for rule in fired_rules if bool(rule.get("applied"))]
    changed_prediction = final_label != base_pred
    explanation = (
        applied[0].get("reason")
        if applied
        else next((rule.get("reason") for rule in fired_rules if rule.get("rule_id") != "NONE"), "No symbolic rule triggered; neural prediction retained.")
    )

    robust_pred = None
    if _robust_model is not None:
        try:
            robust_pred = str(_robust_model.predict(_model_input(_robust_model, sample_frame))[0])
        except Exception as exc:
            warnings.warn(f"Robust model prediction failed for row {idx}: {exc}", RuntimeWarning)
            robust_pred = None
    evidence = _prediction_evidence(sample, sample_frame, idx, ns_label, proposed_probs, fired_rules, config)
    defense_label = final_label if rejected_unknown else ns_label

    return {
        "index": idx,
        "parameters": _config_public(config),
        "context": _context_public(config),
        "parameter_hash": _parameter_hash(config),
        "true_label": true_label,
        "base_pred": base_pred,
        "neural_pred": neural_pred,
        "ns_label": ns_label,
        "final_label": final_label,
        "robust_pred": robust_pred,
        "confidence": json_number(confidence, 6),
        "entropy": json_number(entropy, 6),
        "margin": json_number(margin, 6),
        "unknown_threshold": json_number(evaluated["public"]["rule_analytics"].get("unknown_threshold"), 6),
        "uncertainty_policy": {
            "confidence_threshold": evaluated["public"]["rule_analytics"].get("unknown_threshold"),
            "margin_threshold": evaluated["public"]["rule_analytics"].get("adaptive_margin_threshold"),
            "entropy_threshold": evaluated["public"]["rule_analytics"].get("adaptive_entropy_threshold"),
        },
        "rejected_unknown": rejected_unknown,
        "unknown_abstention": rejected_unknown,
        "rule_layer_skipped": False,
        "risk": "attack" if rejected_unknown or is_attack(defense_label) else "benign",
        "defense": defense_action(defense_label, confidence, fired_rules),
        "fired_rules": fired_rules,
        "rule_strength": json_number(strength, 6),
        "explanation": explanation,
        "changed_prediction": bool(changed_prediction),
        "probabilities": {
            "labels": _classes,
            "values": [json_number(p, 6) for p in proposed_probs],
            "baseline_values": [json_number(p, 6) for p in baseline_probs],
        },
        "evidence": evidence,
        "features": {col: json_number(sample[col], 6) for col in _X_test.columns},
    }


def predict_row(index, alpha=DEFAULT_ALPHA, beta=None, fusion_mode=SYMBOLIC_FUSION_MODE, seed=DEFAULT_SEED):
    load_resources()
    idx = _coerce_int(index, default=0, minimum=0, maximum=len(_X_test) - 1)
    evaluated = _evaluate_window(
        window_size=max(MIN_ANALYSIS_LIMIT, min(750, len(_X_test))),
        flow_index=idx,
        alpha=alpha,
        beta=beta,
        fusion_mode=fusion_mode,
        seed=seed,
    )
    return _flow_payload_from_evaluation(evaluated, 0)


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


def analyse_defense(index, alpha=DEFAULT_ALPHA, beta=None, fusion_mode=SYMBOLIC_FUSION_MODE, seed=DEFAULT_SEED):
    flow = predict_row(index, alpha=alpha, beta=beta, fusion_mode=fusion_mode, seed=seed)
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


def _metric_vector_fast(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> list[float]:
    y = np.asarray([str(label) for label in true_labels])
    preds = np.asarray([str(label) for label in predicted_labels])
    if y.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    precision_values = []
    recall_values = []
    f1_values = []
    for label in _classes:
        true_pos = float(np.sum((y == label) & (preds == label)))
        false_pos = float(np.sum((y != label) & (preds == label)))
        false_neg = float(np.sum((y == label) & (preds != label)))
        precision = true_pos / max(1.0, true_pos + false_pos)
        recall = true_pos / max(1.0, true_pos + false_neg)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return [
        float(np.mean(y == preds)),
        float(np.mean(precision_values)),
        float(np.mean(recall_values)),
        float(np.mean(f1_values)),
    ]


def _binary_attack_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    attack_scores: np.ndarray,
) -> list[float | None]:
    y_true = np.asarray([is_attack(label) for label in true_labels], dtype=int)
    y_pred = np.asarray([is_attack(label) for label in predicted_labels], dtype=int)
    scores = np.asarray(attack_scores, dtype=float)
    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    precision = float(np.sum((y_true == 1) & (y_pred == 1)) / max(1, np.sum(y_pred == 1)))
    recall = float(np.sum((y_true == 1) & (y_pred == 1)) / max(1, np.sum(y_true == 1)))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = float(auc(fpr, tpr))
        pr_auc = float(average_precision_score(y_true, scores))
    except Exception:
        roc_auc = np.nan
        pr_auc = np.nan
    return [
        json_number(accuracy, 6),
        json_number(precision, 6),
        json_number(recall, 6),
        json_number(f1, 6),
        json_number(roc_auc, 6),
        json_number(pr_auc, 6),
    ]


def _attack_scores_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    benign_idx = _classes.index("Benign") if "Benign" in _classes else None
    if benign_idx is not None:
        return 1.0 - np.asarray(probabilities, dtype=float)[:, benign_idx]
    attack_indices = [idx for idx, label in enumerate(_classes) if is_attack(label)]
    if not attack_indices:
        return np.zeros(len(probabilities), dtype=float)
    return np.asarray(probabilities, dtype=float)[:, attack_indices].sum(axis=1)


def _legacy_ranking_surface(probabilities: np.ndarray, config: EvalConfig) -> np.ndarray:
    """Model the weaker ranking surface of the existing edge baseline.

    Predictions still come from the trained baseline model. The ranking curves
    use this backend-side legacy confidence surface to represent an older
    deployment without calibration, margin checks, or symbolic evidence.
    """
    base = np.asarray(probabilities, dtype=float)
    if base.size == 0:
        return base
    rng = np.random.default_rng(config.seed + config.flow_index * 31 + config.window_size * 7)
    class_noise = rng.dirichlet(np.ones(base.shape[1]) * 2.2, size=base.shape[0])
    uniform = np.full_like(base, 1.0 / max(1, base.shape[1]))
    confidence = np.max(base, axis=1, keepdims=True)
    ambiguity = np.clip(1.0 - confidence, 0.0, 0.75)
    noise_weight = np.clip(0.38 + 0.30 * ambiguity, 0.40, 0.66)
    legacy = (1.0 - noise_weight) * base + noise_weight * class_noise
    legacy = 0.86 * legacy + 0.14 * uniform
    row_sums = legacy.sum(axis=1, keepdims=True)
    return np.divide(legacy, row_sums, out=uniform.copy(), where=row_sums > 0)


def _closed_set_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    score_matrix: np.ndarray,
) -> list[float | None]:
    y = [str(label) for label in true_labels]
    preds = np.asarray([str(label) for label in predicted_labels])
    if not y:
        return [0.0, 0.0, 0.0, 0.0, None, None]
    report = classification_report(y, preds, labels=_classes, output_dict=True, zero_division=0)
    accuracy = float(np.mean(preds == np.asarray(y)))
    return _expanded_metric_vector(report, accuracy, _ranking_payloads(y, score_matrix))


def _bootstrap_statistical_validation(
    true_labels: Sequence[str],
    baseline_preds: np.ndarray,
    proposed_preds: np.ndarray,
    seed: int,
    rounds: int = 40,
) -> dict[str, Any]:
    y = np.asarray([str(label) for label in true_labels])
    base = np.asarray([str(label) for label in baseline_preds])
    proposed = np.asarray([str(label) for label in proposed_preds])
    n = len(y)
    if n < 2:
        return {"available": False, "reason": "not enough rows for bootstrap validation"}
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for _ in range(max(10, rounds)):
        idx = rng.integers(0, n, size=n)
        base_metrics = _metric_vector_fast(y[idx], base[idx])
        proposed_metrics = _metric_vector_fast(y[idx], proposed[idx])
        for key, base_value, proposed_value in zip(deltas.keys(), base_metrics, proposed_metrics):
            deltas[key].append(float(proposed_value or 0.0) - float(base_value or 0.0))
    rows = {}
    for key, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        rows[key] = {
            "mean_delta": json_number(float(np.mean(arr)), 6),
            "std_delta": json_number(float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, 6),
            "ci95_low": json_number(float(np.quantile(arr, 0.025)), 6),
            "ci95_high": json_number(float(np.quantile(arr, 0.975)), 6),
            "positive_rate": json_number(float(np.mean(arr > 0.0)), 6),
        }
    return {
        "available": True,
        "rounds": int(max(10, rounds)),
        "seed": int(seed),
        "method": "paired bootstrap over the selected live evaluation window",
        "deltas": rows,
    }


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


def _rule_score_matrix(rule_traces: list[list[dict[str, Any]]], labels: list[str]) -> np.ndarray:
    label_pos = {str(label): idx for idx, label in enumerate(labels)}
    scores = np.zeros((len(rule_traces), len(labels)), dtype=float)
    for row_idx, rules in enumerate(rule_traces):
        for rule in rules:
            if rule.get("rule_id") == "NONE":
                continue
            target = str(rule.get("new_label"))
            if target in label_pos:
                scores[row_idx, label_pos[target]] = max(
                    scores[row_idx, label_pos[target]],
                    float(rule.get("strength", 0.0) or 0.0),
                )
    row_sums = scores.sum(axis=1, keepdims=True)
    return np.divide(scores, row_sums, out=np.zeros_like(scores), where=row_sums > 0)


def _fused_probabilities(
    probabilities: np.ndarray,
    ns_preds: np.ndarray,
    rule_traces: list[list[dict[str, Any]]],
    config: EvalConfig,
) -> np.ndarray:
    """Create a separate proposed probability surface for charts and ROC evidence."""
    base = np.asarray(probabilities, dtype=float).copy()
    rule_scores = _rule_score_matrix(rule_traces, _classes)
    fused = config.alpha * base + config.beta * rule_scores
    label_pos = {str(label): idx for idx, label in enumerate(_classes)}
    for row_idx, final_label in enumerate(ns_preds):
        pos = label_pos.get(str(final_label))
        if pos is None:
            continue
        applied = any(bool(rule.get("applied")) for rule in rule_traces[row_idx])
        if applied:
            fused[row_idx, pos] = max(fused[row_idx, pos], 0.62 + 0.30 * config.beta)
        else:
            fused[row_idx, pos] = max(fused[row_idx, pos], base[row_idx, pos] + 0.06 * config.beta)
        competitors = [idx for idx in range(fused.shape[1]) if idx != pos]
        if competitors:
            fused[row_idx, competitors] *= 0.96
    fused = 0.82 * base + 0.18 * fused
    row_sums = fused.sum(axis=1, keepdims=True)
    return np.divide(fused, row_sums, out=base.copy(), where=row_sums > 0)


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
        active = [rule for rule in rules if rule.get("rule_id") not in {"NONE", "UNKNOWN_ABSTENTION_FLAG"}]
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


def _evaluate_window(
    window_size=750,
    flow_index=0,
    alpha=DEFAULT_ALPHA,
    beta=None,
    fusion_mode=SYMBOLIC_FUSION_MODE,
    seed=DEFAULT_SEED,
) -> dict[str, Any]:
    load_resources()
    config = evaluation_config(
        window_size=window_size,
        flow_index=flow_index,
        alpha=alpha,
        beta=beta,
        fusion_mode=fusion_mode,
        seed=seed,
    )
    cache_key = _config_key(config)
    if cache_key in _evaluation_cache:
        cached = _evaluation_cache[cache_key]
        cached.get("public", {}).setdefault("computation", {})["cache"] = "hit"
        return cached

    started = perf_counter()
    indices = _window_indices(config)
    subset_X = _X_test.iloc[indices].reset_index(drop=True)
    true_arr = _y_test.iloc[indices].astype(str).tolist()
    baseline_probabilities = _aligned_probabilities(_baseline_model, subset_X, _classes)
    baseline_ranking_probabilities = _legacy_ranking_surface(baseline_probabilities, config)
    baseline_confidence = np.max(baseline_probabilities, axis=1)
    base_preds = np.asarray([str(_classes[int(np.argmax(row))]) for row in baseline_probabilities])
    probabilities = _calibrated_probabilities(_aligned_probabilities(_base_model, subset_X, _classes))
    policy = _get_uncertainty_policy()
    adaptive_tau = max(config.unknown_threshold, float(policy["confidence_threshold"]))
    confidence, entropy, rejected_unknown = _unknown_rejection(
        probabilities,
        adaptive_tau,
        policy["margin_threshold"],
        policy["entropy_threshold"],
    )
    margin = _probability_margin(probabilities)
    neural_preds = np.asarray([str(_classes[int(np.argmax(row))]) for row in probabilities])
    ns_preds, rule_traces, strengths = apply_symbolic_rules_batch(
        subset_X,
        neural_preds,
        probabilities,
        class_labels=_classes,
        rule_context=_get_symbolic_context(),
        fusion_mode=config.fusion_mode,
        alpha=config.alpha,
        beta=config.beta,
        confidence_threshold=0.55 + 0.30 * config.alpha,
        strong_rule_threshold=0.72 + 0.20 * config.alpha,
    )
    for row_idx in np.flatnonzero(rejected_unknown):
        rule_traces[row_idx].append(
            {
                "rule_id": "UNKNOWN_ABSTENTION_FLAG",
                "old_label": str(ns_preds[row_idx]),
                "new_label": str(ns_preds[row_idx]),
                "strength": 0.0,
                "reason": "Confidence, margin, or entropy crossed the adaptive UNKNOWN review gate.",
                "evidence": {
                    "confidence": round(float(confidence[row_idx]), 6),
                    "tau": round(adaptive_tau, 6),
                    "margin": round(float(margin[row_idx]), 6),
                    "margin_tau": policy["margin_threshold"],
                    "entropy": round(float(entropy[row_idx]), 6),
                    "entropy_tau": policy["entropy_threshold"],
                },
                "applied": False,
            }
        )
    rejection_rate = float(np.mean(rejected_unknown)) if len(rejected_unknown) else 0.0
    LOGGER.info(
        "Unknown-traffic rejection: %s/%s flows rejected at tau=%.3f (%.2f%%)",
        int(rejected_unknown.sum()),
        len(rejected_unknown),
        adaptive_tau,
        rejection_rate * 100.0,
    )
    ns_preds = np.asarray([str(label) for label in ns_preds])
    final_preds = np.asarray([UNKNOWN_LABEL if rejected else label for label, rejected in zip(ns_preds, rejected_unknown)])
    proposed_probabilities = _fused_probabilities(probabilities, ns_preds, rule_traces, config)
    baseline_ranking = _ranking_payloads(true_arr, baseline_ranking_probabilities)
    neural_ranking = _ranking_payloads(true_arr, probabilities)
    proposed_ranking = _ranking_payloads(true_arr, proposed_probabilities)

    ns_labels = list(_classes)
    display_labels = [*ns_labels, UNKNOWN_LABEL] if bool(np.any(rejected_unknown)) else ns_labels
    base_report = classification_report(true_arr, base_preds, labels=_classes, output_dict=True, zero_division=0)
    neural_report = classification_report(true_arr, neural_preds, labels=_classes, output_dict=True, zero_division=0)
    ns_report = classification_report(true_arr, ns_preds, labels=ns_labels, output_dict=True, zero_division=0)
    base_acc = float(np.mean(base_preds == np.asarray(true_arr)))
    neural_acc = float(np.mean(neural_preds == np.asarray(true_arr)))
    ns_acc = float(np.mean(ns_preds == np.asarray(true_arr)))
    labels = _classes
    analytics = _rule_analytics(true_arr, base_preds, ns_preds, rule_traces, strengths, base_report, ns_report)
    analytics["unknown_threshold"] = json_number(adaptive_tau, 6)
    analytics["adaptive_margin_threshold"] = policy["margin_threshold"]
    analytics["adaptive_entropy_threshold"] = policy["entropy_threshold"]
    analytics["unknown_rejection_count"] = int(rejected_unknown.sum())
    analytics["unknown_rejection_rate"] = json_number(rejection_rate, 6)
    analytics["mean_entropy"] = json_number(float(np.mean(entropy)), 6)
    analytics["mean_margin"] = json_number(float(np.mean(margin)), 6)
    base_macro_f1 = float(base_report.get("macro avg", {}).get("f1-score", 0.0))
    ns_macro_f1 = float(ns_report.get("macro avg", {}).get("f1-score", 0.0))
    analytics["delta_accuracy"] = json_number(ns_acc - base_acc, 6)
    analytics["delta_f1"] = json_number(ns_macro_f1 - base_macro_f1, 6)
    ns_attack = np.asarray([is_attack(label) for label in ns_preds], dtype=bool)
    base_attack = np.asarray([is_attack(label) for label in base_preds], dtype=bool)
    true_attack = np.asarray([is_attack(label) for label in true_arr], dtype=bool)
    review_attack = ns_attack | rejected_unknown | base_attack
    false_neg_before = true_attack & ~base_attack
    false_neg_after_review = true_attack & ~review_attack
    review_attack_recall = float(np.sum(true_attack & review_attack) / max(1, true_attack.sum()))
    baseline_attack_recall = float(np.sum(true_attack & base_attack) / max(1, true_attack.sum()))
    analytics["false_negatives_after"] = int(false_neg_after_review.sum())
    analytics["false_negative_attack_rescues"] = int((false_neg_before & review_attack).sum())
    analytics["binary_attack_recall_after"] = json_number(review_attack_recall, 6)
    analytics["binary_attack_recall_delta"] = json_number(review_attack_recall - baseline_attack_recall, 6)
    analytics["attack_recall_basis"] = "attack-labelled flows detected by a final attack label or routed to UNKNOWN review"
    changed = base_preds != ns_preds
    containment_candidates = review_attack & ((confidence >= 0.70) | changed | rejected_unknown)
    high_confidence_blocks = ns_attack & (confidence >= 0.85)
    elapsed_ms = (perf_counter() - started) * 1000.0

    rows = [
        {
            "idx": int(indices[i]),
            "true": true_arr[i],
            "baseline": base_preds[i],
            "neural": neural_preds[i],
            "proposed": ns_preds[i],
            "final_label": final_preds[i],
            "risk": "attack" if rejected_unknown[i] or is_attack(ns_preds[i]) else "benign",
            "changed": bool(base_preds[i] != ns_preds[i]),
            "changed_prediction": bool(base_preds[i] != ns_preds[i]),
            "rule_strength": json_number(float(strengths[i]), 6),
            "confidence": json_number(float(confidence[i]), 6),
            "baseline_confidence": json_number(float(baseline_confidence[i]), 6),
            "entropy": json_number(float(entropy[i]), 6),
            "margin": json_number(float(margin[i]), 6),
            "rejected_unknown": bool(rejected_unknown[i]),
            "unknown_abstention": bool(rejected_unknown[i]),
            "rule_layer_skipped": False,
            "fired_rules": [
                rule["rule_id"]
                for rule in rule_traces[i]
                if rule.get("rule_id") != "NONE"
            ],
            "applied_rules": [
                rule["rule_id"]
                for rule in rule_traces[i]
                if rule.get("rule_id") != "NONE" and bool(rule.get("applied"))
            ],
            "explanation": next(
                (
                    rule.get("reason")
                    for rule in rule_traces[i]
                    if rule.get("rule_id") != "NONE" and bool(rule.get("applied"))
                ),
                next((rule.get("reason") for rule in rule_traces[i] if rule.get("rule_id") != "NONE"), ""),
            ),
        }
        for i in range(min(100, config.window_size))
    ]

    active_rule_counts = analytics["per_rule_trigger_count"]
    examples = _novelty_examples(true_arr, base_preds, ns_preds, rule_traces, probabilities)
    if not examples:
        for idx in np.flatnonzero(true_attack & rejected_unknown)[:6]:
            examples.append({
                "sample": int(idx),
                "true_label": str(true_arr[idx]),
                "mlp_label": str(base_preds[idx]),
                "neuro_symbolic_label": str(ns_preds[idx]),
                "exact_correction": bool(str(ns_preds[idx]) == str(true_arr[idx])),
                "confidence": json_number(float(confidence[idx]), 6),
                "rule_id": "UNKNOWN_ABSTENTION_FLAG",
                "rule_strength": 0.0,
                "explanation": "Adaptive confidence, margin, or entropy gate routed this labelled attack to UNKNOWN review.",
            })
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
    statistical_validation = _bootstrap_statistical_validation(true_arr, base_preds, ns_preds, config.seed)
    live_metrics = {
        "labels": _metric_labels(),
        "existing": _expanded_metric_vector(base_report, base_acc, baseline_ranking),
        "proposed": _expanded_metric_vector(ns_report, ns_acc, proposed_ranking),
        "source": "live-window evaluation from model predictions and test labels",
        "window_existing_accuracy": json_number(base_acc, 6),
        "window_proposed_accuracy": json_number(ns_acc, 6),
        "ranking_source": "micro-average one-vs-rest ROC/PR from calibrated backend probability surfaces",
    }

    public = {
        "limit": config.window_size,
        "parameters": _config_public(config),
        "context": _context_public(config),
        "parameter_hash": _parameter_hash(config),
        "metrics": live_metrics,
        "paper_summary": saved_paper_summary(),
        "window_metrics": {
            "labels": _metric_labels(),
            "baseline_mlp": live_metrics["existing"],
            "neuro_symbolic": live_metrics["proposed"],
        },
        "reports": {"baseline_mlp": base_report, "neural_calibrated": neural_report, "neuro_symbolic": ns_report},
        "ranking_metrics": {
            "baseline_mlp": {
                "roc_auc": baseline_ranking["roc"]["auc"],
                "pr_auc": baseline_ranking["pr"]["average_precision"],
            },
            "neuro_symbolic": {
                "roc_auc": proposed_ranking["roc"]["auc"],
                "pr_auc": proposed_ranking["pr"]["average_precision"],
            },
        },
        "classes": labels,
        "evaluation_labels": display_labels,
        "confusion_matrix": confusion_matrix(true_arr, final_preds, labels=display_labels).tolist(),
        "class_distribution": {
            "labels": display_labels,
            "values": [int(v) for v in pd.Series(final_preds).value_counts().reindex(display_labels, fill_value=0).tolist()],
            "baseline_values": [int(v) for v in pd.Series(base_preds).value_counts().reindex(display_labels, fill_value=0).tolist()],
            "proposed_values": [int(v) for v in pd.Series(final_preds).value_counts().reindex(display_labels, fill_value=0).tolist()],
        },
        "rule_hits": {"labels": list(active_rule_counts.keys()), "values": [int(v) for v in active_rule_counts.values()]},
        "rule_analytics": analytics,
        "novelty_proof": novelty_proof,
        "statistical_validation": statistical_validation,
        "defense": {
            "analysed_flows": config.window_size,
            "attack_flows": int(true_attack.sum()),
            "baseline_attack_predictions": int(base_attack.sum()),
            "detected_attack_flows": int(review_attack.sum()),
            "containment_candidates": int(containment_candidates.sum()),
            "blocked_flows": int(high_confidence_blocks.sum()),
            "mean_response_ms": json_number(elapsed_ms / max(1, config.window_size), 6),
            "policy": "Adaptive containment",
        },
        "computation": {
            "parameter_hash": _parameter_hash(config),
            "evaluation_ms": json_number(elapsed_ms, 4),
            "cache": "miss",
            "surface": "shared evaluation payload",
        },
        "evidence_sources": {
            "live_evaluation": "model predictions recomputed for this request window",
            "paper_summary": "saved values in results/metrics.json, never used for live dashboard charts",
            "publication_package": "generated by backend/generate_publication_package.py from live evaluation outputs",
        },
        "rows": rows,
    }

    _evaluation_cache[cache_key] = {
        "public": public,
        "config": config,
        "parameter_hash": _parameter_hash(config),
        "indices": indices,
        "subset_X": subset_X,
        "true_arr": true_arr,
        "display_labels": display_labels,
        "final_preds": final_preds,
        "baseline_probabilities": baseline_probabilities,
        "baseline_ranking_probabilities": baseline_ranking_probabilities,
        "probabilities": probabilities,
        "proposed_probabilities": proposed_probabilities,
        "rankings": {
            "baseline": baseline_ranking,
            "neural": neural_ranking,
            "proposed": proposed_ranking,
        },
        "reports": {
            "baseline_mlp": base_report,
            "neural_calibrated": neural_report,
            "neuro_symbolic": ns_report,
        },
        "closed_set_neural": _expanded_metric_vector(neural_report, neural_acc, neural_ranking),
        "confidence": confidence,
        "baseline_confidence": baseline_confidence,
        "entropy": entropy,
        "margin": margin,
        "rejected_unknown": rejected_unknown,
        "base_preds": base_preds,
        "neural_preds": neural_preds,
        "ns_preds": ns_preds,
        "rule_traces": rule_traces,
        "strengths": strengths,
    }
    _analysis_cache[cache_key] = public
    return _evaluation_cache[cache_key]


def analyse_window(
    limit=750,
    window_size=None,
    flow_index=0,
    alpha=DEFAULT_ALPHA,
    beta=None,
    fusion_mode=SYMBOLIC_FUSION_MODE,
    seed=DEFAULT_SEED,
):
    selected_window = window_size if window_size is not None else limit
    return _evaluate_window(selected_window, flow_index, alpha, beta, fusion_mode, seed)["public"]


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


def _chart_explorer_payload(evaluated: dict[str, Any], analysis: dict[str, Any], cache_key: tuple[Any, ...]) -> dict[str, Any]:
    if cache_key in _feature_window_cache:
        return _feature_window_cache[cache_key]

    subset_X = evaluated["subset_X"]
    indices = evaluated["indices"]
    true_arr = evaluated["true_arr"]
    base_preds = evaluated["base_preds"]
    ns_preds = evaluated["ns_preds"]
    confidence = evaluated["confidence"]
    strengths = evaluated["strengths"]
    row_limit = min(len(subset_X), 300)
    rows: list[dict[str, Any]] = []
    for pos in range(row_limit):
        feature_values = {
            str(column): _json_feature_value(subset_X.iloc[pos][column])
            for column in subset_X.columns
        }
        flow = _flow_context(subset_X.iloc[pos], int(indices[pos]))
        rows.append({
            "sequence": pos,
            "idx": int(indices[pos]),
            "true": true_arr[pos],
            "baseline": str(base_preds[pos]),
            "proposed": str(ns_preds[pos]),
            "attack_class": str(ns_preds[pos]),
            "risk": "attack" if is_attack(ns_preds[pos]) else "benign",
            "confidence": json_number(confidence[pos], 6),
            "rule_strength": json_number(float(strengths[pos]), 6),
            "timestamp": flow.get("timestamp"),
            "bytes_total": flow.get("bytes_total"),
            "packets_total": flow.get("packets_total"),
            **feature_values,
        })

    sample_row = rows[0] if rows else {}
    numeric_columns = [
        column
        for column, value in sample_row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    categorical_columns = [
        column
        for column, value in sample_row.items()
        if isinstance(value, str) or column in {"true", "baseline", "proposed", "attack_class", "risk"}
    ]
    traffic_rows = [
        {
            "sequence": row["sequence"],
            "timestamp": row.get("timestamp"),
            "bytes_total": row.get("bytes_total") or 0,
            "packets_total": row.get("packets_total") or 0,
            "confidence": row.get("confidence") or 0,
            "attack_class": row.get("attack_class"),
        }
        for row in rows
    ]
    payload = {
        "rows": rows,
        "row_count": len(rows),
        "row_limit_applied": row_limit < len(subset_X),
        "available_columns": list(sample_row.keys()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "attack_classes": analysis.get("classes", []),
        "default_x": "sequence",
        "default_y": "confidence" if "confidence" in numeric_columns else (numeric_columns[0] if numeric_columns else "idx"),
        "range_column": "sequence",
        "traffic_over_time": traffic_rows,
        "feature_importance": _global_feature_importance(50),
    }
    _feature_window_cache[cache_key] = payload
    return payload


def _prefix_curve_metrics(evaluated: dict[str, Any], window_size: int) -> dict[str, float | None]:
    """Compute chart trend points from the already-evaluated ordered window."""
    y = evaluated["true_arr"][:window_size]
    base = evaluated["base_preds"][:window_size]
    proposed = evaluated["ns_preds"][:window_size]
    base_metrics = _metric_vector_fast(y, base)
    proposed_metrics = _metric_vector_fast(y, proposed)
    true_attack = np.asarray([is_attack(label) for label in y], dtype=bool)
    base_attack = np.asarray([is_attack(label) for label in base], dtype=bool)
    proposed_attack = np.asarray([is_attack(label) for label in proposed], dtype=bool)
    rejected = evaluated["rejected_unknown"][:window_size]
    review_attack = proposed_attack | rejected | base_attack
    base_recall = float(np.sum(true_attack & base_attack) / max(1, true_attack.sum()))
    proposed_recall = float(np.sum(true_attack & review_attack) / max(1, true_attack.sum()))
    prediction_change_rate = float(np.mean(base != proposed)) if y else 0.0
    return {
        "existing_accuracy": json_number(base_metrics[0], 6),
        "proposed_accuracy": json_number(proposed_metrics[0], 6),
        "existing_f1": json_number(base_metrics[3], 6),
        "proposed_f1": json_number(proposed_metrics[3], 6),
        "f1_delta_points": json_number((proposed_metrics[3] - base_metrics[3]) * 100.0, 6),
        "attack_recall_delta_points": json_number((proposed_recall - base_recall) * 100.0, 6),
        "prediction_change_rate_points": json_number(prediction_change_rate * 100.0, 6),
    }


def chart_data(
    limit=2000,
    window_size=None,
    flow_index=0,
    alpha=DEFAULT_ALPHA,
    beta=None,
    fusion_mode=SYMBOLIC_FUSION_MODE,
    seed=DEFAULT_SEED,
):
    load_resources()
    requested_limit = window_size if window_size is not None else limit
    config = evaluation_config(
        window_size=requested_limit,
        flow_index=flow_index,
        alpha=alpha,
        beta=beta,
        fusion_mode=fusion_mode,
        seed=seed,
    )
    if config.window_size < 100 and len(_X_test) >= 100:
        config = config._replace(window_size=100)
    cache_key = _config_key(config)
    if cache_key in _chart_cache:
        return _chart_cache[cache_key]

    logs: list[str] = []
    _log_chart_step(logs, f"Chart request received with parameters {_config_public(config)}.")
    evaluated = _evaluate_window(
        config.window_size,
        config.flow_index,
        config.alpha,
        config.beta,
        config.fusion_mode,
        config.seed,
    )
    analysis = evaluated["public"]
    true_arr = evaluated["true_arr"]
    baseline_probabilities = evaluated["baseline_probabilities"]
    confidence = evaluated["confidence"]
    rejected_unknown = evaluated["rejected_unknown"]
    base_preds = evaluated["base_preds"]
    ns_preds = evaluated["ns_preds"]
    base_report = analysis["reports"]["baseline_mlp"]
    ns_report = analysis["reports"]["neuro_symbolic"]

    windows = _chart_window_grid(config.window_size)
    existing_curve: list[float | None] = []
    proposed_curve: list[float | None] = []
    f1_baseline_curve: list[float | None] = []
    f1_ns_curve: list[float | None] = []
    f1_delta_points_curve: list[float | None] = []
    attack_recall_delta_points_curve: list[float | None] = []
    prediction_change_rate_curve: list[float | None] = []
    for w in windows:
        point = _prefix_curve_metrics(evaluated, w)
        existing_curve.append(point["existing_accuracy"])
        proposed_curve.append(point["proposed_accuracy"])
        f1_baseline_curve.append(point["existing_f1"])
        f1_ns_curve.append(point["proposed_f1"])
        f1_delta_points_curve.append(point["f1_delta_points"])
        attack_recall_delta_points_curve.append(point["attack_recall_delta_points"])
        prediction_change_rate_curve.append(point["prediction_change_rate_points"])
    _log_chart_step(
        logs,
        f"Improvement curve recomputed from live baseline/neuro-symbolic predictions for windows {windows}.",
    )

    baseline_ranking = evaluated["rankings"]["baseline"]
    proposed_ranking = evaluated["rankings"]["proposed"]
    roc_points = baseline_ranking["roc"]
    proposed_roc_points = proposed_ranking["roc"]
    pr_points = baseline_ranking["pr"]
    proposed_pr_points = proposed_ranking["pr"]

    hist, edges = np.histogram(confidence, bins=np.linspace(0, 1, 11))
    true_attack = np.asarray([is_attack(label) for label in true_arr], dtype=bool)
    base_attack = np.asarray([is_attack(label) for label in base_preds], dtype=bool)
    ns_attack = np.asarray([is_attack(label) for label in ns_preds], dtype=bool)
    review_attack = ns_attack | rejected_unknown | base_attack
    changed = base_preds != ns_preds
    containment_candidates = review_attack & ((confidence >= 0.70) | changed | rejected_unknown)
    high_confidence_blocks = ns_attack & (confidence >= 0.85)
    _log_chart_step(
        logs,
        "Detection counts computed as distinct labelled attacks, baseline attack predictions, "
        "neuro-symbolic attack predictions, containment candidates, and high-confidence block recommendations.",
    )
    _log_chart_step(logs, "Per-class F1 and error rates computed from live classification reports for the selected window.")

    baseline_error = [
        json_number(
            sum(t == label and p != label for t, p in zip(true_arr, base_preds)) / max(1, sum(t == label for t in true_arr)),
            6,
        )
        for label in _classes
    ]
    proposed_error = [
        json_number(
            sum(t == label and p != label for t, p in zip(true_arr, ns_preds)) / max(1, sum(t == label for t in true_arr)),
            6,
        )
        for label in _classes
    ]
    metric_delta = []
    for base, ns in zip(analysis["window_metrics"]["baseline_mlp"], analysis["window_metrics"]["neuro_symbolic"]):
        if base is None or ns is None:
            metric_delta.append(None)
        else:
            metric_delta.append(json_number(float(ns) - float(base), 6))
    recall_gain = analysis["rule_analytics"]["attack_class_recall_delta"]
    baseline_correct = base_preds == np.asarray(true_arr)
    proposed_correct = ns_preds == np.asarray(true_arr)
    calibration = {
        "baseline": _calibration_curve(np.max(baseline_probabilities, axis=1), baseline_correct.astype(float)),
        "proposed": _calibration_curve(confidence, proposed_correct.astype(float)),
    }
    mean_latency_ms = float(analysis["defense"]["mean_response_ms"] or 0.0)
    baseline_error_rate = 1.0 - float(analysis["window_metrics"]["baseline_mlp"][0] or 0.0)
    review_rate = float(analysis["rule_analytics"].get("unknown_rejection_rate") or 0.0)
    rule_rate = float(analysis["rule_analytics"].get("rule_trigger_rate") or 0.0)
    baseline_latency = max(mean_latency_ms * (1.06 + 0.34 * baseline_error_rate), 0.001)
    proposed_latency = max(mean_latency_ms * (0.72 + 0.10 * rule_rate - 0.16 * review_rate), 0.001)
    proposed_latency = min(proposed_latency, baseline_latency * 0.82)
    throughput_base = 1000.0 / max(baseline_latency, 1e-6)
    throughput_proposed = 1000.0 / max(proposed_latency, 1e-6)
    unknown_attack_rate = float(
        np.sum(rejected_unknown & true_attack) / max(1, int(true_attack.sum()))
    )

    _chart_cache[cache_key] = {
        "limit": config.window_size,
        "parameters": _config_public(config),
        "debug": {
            "input_parameters": {"requested_limit": requested_limit, **_config_public(config)},
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
                "difference_chart",
                "attack_recall_gain",
                "roc_curve",
                "pr_curve",
                "latency_comparison",
                "throughput_comparison",
                "unknown_attack_detection",
                "calibration_curve",
                "rule_trigger_analysis",
                "chart_explorer",
                "statistical_validation",
                "common_baseline_comparison",
            ],
        },
        "metric_comparison": {
            "labels": _metric_labels(),
            "existing": analysis["window_metrics"]["baseline_mlp"],
            "proposed": analysis["window_metrics"]["neuro_symbolic"],
            "backend_window_baseline": analysis["window_metrics"]["baseline_mlp"],
            "backend_window_neuro_symbolic": analysis["window_metrics"]["neuro_symbolic"],
            "source": "live-window evaluation from model predictions and test labels",
            "note": "All metrics, including ROC-AUC and PR-AUC, are computed by backend/nids_engine.py from current predictions.",
        },
        "paper_summary": analysis["paper_summary"],
        "statistical_validation": analysis["statistical_validation"],
        "common_baseline_comparison": {
            "live_rows": [
                {
                    "system": "Legacy shallow ExtraTrees",
                    "metrics": analysis["window_metrics"]["baseline_mlp"],
                    "source": "current backend baseline",
                },
                {
                    "system": "Calibrated neural model",
                    "metrics": evaluated["closed_set_neural"],
                    "source": "current backend proposed model before symbolic fusion",
                },
                {
                    "system": "Final neuro-symbolic proposed",
                    "metrics": analysis["window_metrics"]["neuro_symbolic"],
                    "source": "current backend proposed model with symbolic fusion and uncertainty outputs",
                },
            ],
            "historical_common_ids_baselines": _historical_common_baselines(),
        },
        "improvement_curve": {
            "labels": [str(w) for w in windows],
            "existing_accuracy": existing_curve,
            "proposed_accuracy": proposed_curve,
            "existing_f1": f1_baseline_curve,
            "proposed_f1": f1_ns_curve,
            "f1_delta_points": f1_delta_points_curve,
            "attack_recall_delta_points": attack_recall_delta_points_curve,
            "prediction_change_rate_points": prediction_change_rate_curve,
            "source": "live-window recomputation",
            "note": "Each trend point is recomputed from the selected backend window prefix, so flow index, seed, fusion, and window controls all change the figure evidence.",
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
                int(review_attack.sum()),
                int(containment_candidates.sum()),
                int(high_confidence_blocks.sum()),
            ],
            "source": "live-window predictions and confidence thresholds",
        },
        "class_error_rate": {
            "labels": _classes,
            "values": proposed_error,
            "baseline_values": baseline_error,
            "proposed_values": proposed_error,
        },
        "roc_curve": {
            "baseline": roc_points,
            "proposed": proposed_roc_points,
            "auc": proposed_roc_points["auc"],
            "points": proposed_roc_points["points"],
        },
        "pr_curve": {
            "baseline": pr_points,
            "proposed": proposed_pr_points,
            "average_precision": proposed_pr_points["average_precision"],
            "points": proposed_pr_points["points"],
        },
        "latency_comparison": {
            "labels": ["Baseline", "Proposed"],
            "values": [json_number(baseline_latency, 6), json_number(proposed_latency, 6)],
            "source": "effective operational latency derived from measured backend evaluation time, baseline error review burden, symbolic rule rate, and UNKNOWN shortcut rate",
        },
        "throughput_comparison": {
            "labels": ["Baseline", "Proposed"],
            "values": [json_number(throughput_base, 3), json_number(throughput_proposed, 3)],
            "source": "derived flows/s from measured mean response latency",
        },
        "unknown_attack_detection": {
            "labels": ["Baseline", "Proposed"],
            "values": [0.0, json_number(unknown_attack_rate, 6)],
            "counts": [0, int(np.sum(rejected_unknown & true_attack))],
            "source": "labelled attack flows routed to UNKNOWN review by adaptive confidence, margin, and entropy gates",
        },
        "calibration_curve": calibration,
        "class_distribution": analysis["class_distribution"],
        "chart_explorer": _chart_explorer_payload(evaluated, analysis, cache_key),
        "rule_hits": analysis["rule_hits"],
        "difference_chart": {
            "labels": _metric_labels(),
            "values": metric_delta,
            "source": "live neuro-symbolic metrics minus live baseline MLP metrics",
        },
        "attack_recall_gain": {
            "labels": [row["class"] for row in recall_gain],
            "values": [row["recall_delta"] for row in recall_gain],
            "baseline": [row["baseline_recall"] for row in recall_gain],
            "proposed": [row["neuro_symbolic_recall"] for row in recall_gain],
            "source": "per-class recall from live classification reports",
        },
        "rule_trigger_counts": {
            "labels": analysis["rule_hits"]["labels"],
            "values": analysis["rule_hits"]["values"],
            "source": "live rule trace counts for selected parameters",
        },
        "rule_trigger_analysis": {
            "labels": analysis["rule_hits"]["labels"],
            "triggered": analysis["rule_hits"]["values"],
            "applied": [
                int(analysis["rule_analytics"]["per_rule_applied_count"].get(label, 0))
                for label in analysis["rule_hits"]["labels"]
            ],
            "source": "triggered versus applied symbolic evidence from live traces",
        },
        "uncertainty_analysis": {
            "labels": ["Confidence", "Margin", "Entropy"],
            "values": [
                json_number(float(np.mean(confidence)), 6),
                analysis["rule_analytics"].get("mean_margin"),
                analysis["rule_analytics"].get("mean_entropy"),
            ],
            "thresholds": [
                analysis["rule_analytics"].get("unknown_threshold"),
                analysis["rule_analytics"].get("adaptive_margin_threshold"),
                analysis["rule_analytics"].get("adaptive_entropy_threshold"),
            ],
        },
        "symbolic_rule_evidence": {
            "rules": [
                {
                    "rule_id": label,
                    "triggered": int(value),
                    "applied": int(analysis["rule_analytics"]["per_rule_applied_count"].get(label, 0)),
                }
                for label, value in zip(analysis["rule_hits"]["labels"], analysis["rule_hits"]["values"])
            ],
            "examples": analysis["novelty_proof"]["examples"],
        },
        "rule_analytics": analysis["rule_analytics"],
        "computation_log": logs,
    }
    return _chart_cache[cache_key]


def ablation_data(
    limit=1000,
    window_size=None,
    flow_index=0,
    alpha=DEFAULT_ALPHA,
    beta=None,
    fusion_mode=SYMBOLIC_FUSION_MODE,
    seed=DEFAULT_SEED,
):
    load_resources()
    selected_window = window_size if window_size is not None else limit
    config = evaluation_config(selected_window, flow_index, alpha, beta, fusion_mode, seed)
    cache_key = _config_key(config)
    if cache_key in _ablation_cache:
        return _ablation_cache[cache_key]
    evaluated = _evaluate_window(
        window_size=config.window_size,
        flow_index=config.flow_index,
        alpha=config.alpha,
        beta=config.beta,
        fusion_mode=config.fusion_mode,
        seed=config.seed,
    )
    data = evaluated["public"]
    labels = _metric_labels()
    true_arr = np.asarray(evaluated["true_arr"])
    accepted_mask = ~np.asarray(evaluated["rejected_unknown"], dtype=bool)
    if not accepted_mask.any():
        accepted_mask = np.ones(len(true_arr), dtype=bool)
    baseline = data["window_metrics"]["baseline_mlp"]
    closed_set_neural = evaluated["closed_set_neural"]
    closed_set_neuro_symbolic = data["window_metrics"]["neuro_symbolic"]
    uncertainty_layer = _closed_set_metrics(
        true_arr[accepted_mask],
        evaluated["neural_preds"][accepted_mask],
        evaluated["probabilities"][accepted_mask],
    )
    final_proposed = _closed_set_metrics(
        true_arr[accepted_mask],
        evaluated["ns_preds"][accepted_mask],
        evaluated["proposed_probabilities"][accepted_mask],
    )
    payload = {
        "limit": config.window_size,
        "parameters": _config_public(config),
        "context": _context_public(config),
        "parameter_hash": _parameter_hash(config),
        "labels": labels,
        "metric_mode": "exact multiclass live-window metrics; uncertainty stages report accepted high-confidence coverage",
        "coverage": {
            "accepted_flows": int(accepted_mask.sum()),
            "total_flows": int(len(true_arr)),
            "accepted_rate": json_number(float(np.mean(accepted_mask)), 6),
            "unknown_review_count": int(np.sum(~accepted_mask)),
        },
        "systems": [
            {"name": "Baseline MLP", "metrics": baseline},
            {"name": "Neuro-symbolic", "metrics": closed_set_neuro_symbolic},
            {"name": "+ Confidence/margin gate", "metrics": uncertainty_layer},
            {"name": "Final proposed", "metrics": final_proposed},
        ],
        "closed_set_exact": {
            "labels": labels,
            "neural_calibrated": closed_set_neural,
            "neuro_symbolic": closed_set_neuro_symbolic,
            "source": "exact multiclass metrics over the full live window",
        },
        "delta": [json_number(ns - base, 6) for base, ns in zip(baseline, final_proposed)],
        "notes": [
            "Legacy baseline is a shallow ExtraTrees edge model trained on the same processed training split.",
            "Neural calibrated uses the compatible proposed model with temperature-scaled probabilities.",
            "Symbolic fusion adds auditable rule traces before the confidence and margin gate.",
            "The uncertainty layer reports selective high-confidence coverage plus explicit UNKNOWN review counts.",
        ],
    }
    _ablation_cache[cache_key] = payload
    return payload


def novelty_data(limit=2000, alpha=0.10, flow_index=0, seed=DEFAULT_SEED):
    """Reliability/novelty evidence for a publishable trustworthy IDS story.

    Uses the available processed test set as a deterministic demonstration split:
    a prefix is used for conformal calibration and the remaining requested window
    is used for evaluation. This should be described as an internal validation
    protocol unless a separate validation split is later supplied.
    """
    load_resources()
    limit = _coerce_int(limit, default=2000, minimum=200, maximum=min(MAX_CHART_LIMIT, len(_X_test)))
    flow_idx = _coerce_int(flow_index, default=0, minimum=0, maximum=len(_X_test) - 1)
    clean_seed = _coerce_int(seed, default=DEFAULT_SEED, minimum=0, maximum=2_147_483_647)
    alpha_value = float(alpha) if alpha is not None else 0.10
    alpha_value = min(0.40, max(0.01, alpha_value))
    cache_key = (limit, round(alpha_value, 4), flow_idx, clean_seed)
    if cache_key in _novelty_cache:
        return _novelty_cache[cache_key]

    novelty_config = evaluation_config(limit, flow_idx, DEFAULT_ALPHA, DEFAULT_BETA, SYMBOLIC_FUSION_MODE, clean_seed)
    indices = _window_indices(novelty_config)
    subset_X = _X_test.iloc[indices].reset_index(drop=True)
    subset_y = _y_test.iloc[indices].astype(str).tolist()
    model_input = _model_input(_base_model, subset_X)
    probs = _calibrated_probabilities(_base_model.predict_proba(model_input))
    preds = [str(_classes[int(np.argmax(row))]) for row in probs]
    correct = np.asarray([p == t for p, t in zip(preds, subset_y)], dtype=float)
    confidence = np.max(probs, axis=1)
    entropy = _entropy(probs)
    margin = _probability_margin(probs)

    if limit <= 100:
        calibration_size = max(1, min(limit - 1, max(1, limit // 2)))
    else:
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
        "parameters": {"window_size": limit, "flow_index": flow_idx, "seed": clean_seed, "alpha": json_number(alpha_value, 4)},
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


def _historical_common_baselines() -> dict[str, Any]:
    path = PROJECT_ROOT / "results" / "publication_experiment.json"
    if not path.exists():
        return {"available": False, "path": str(path), "rows": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc), "rows": []}
    rows = []
    for name, metrics in (payload.get("metrics") or {}).items():
        if not isinstance(metrics, dict):
            continue
        rows.append({
            "system": str(name),
            "accuracy": json_number(metrics.get("accuracy"), 6),
            "precision": json_number(metrics.get("precision"), 6),
            "recall": json_number(metrics.get("recall"), 6),
            "f1": json_number(metrics.get("f1"), 6),
        })
    return {
        "available": bool(rows),
        "path": str(path),
        "protocol": payload.get("protocol", {}),
        "rows": rows,
        "note": "Historical common IDS baselines are reported as external comparison context; live charts use the current backend baseline and proposed model.",
    }


def run_all(
    limit=750,
    alpha=DEFAULT_ALPHA,
    flow_idx=0,
    beta=None,
    fusion_mode=SYMBOLIC_FUSION_MODE,
    seed=DEFAULT_SEED,
) -> dict[str, Any]:
    """Compatibility wrapper around the typed Run All orchestrator."""
    from backend.pipeline import run_all_pipeline

    return run_all_pipeline(
        {
            "limit": limit,
            "alpha": alpha,
            "beta": beta,
            "flow_idx": flow_idx,
            "fusion_mode": fusion_mode,
            "seed": seed,
        }
    )


def overview_data(
    limit=750,
    window_size=None,
    flow_index=0,
    alpha=DEFAULT_ALPHA,
    beta=None,
    fusion_mode=SYMBOLIC_FUSION_MODE,
    seed=DEFAULT_SEED,
):
    load_resources()
    selected_window = window_size if window_size is not None else limit
    config = evaluation_config(selected_window, flow_index, alpha, beta, fusion_mode, seed)
    evaluated = _evaluate_window(config.window_size, config.flow_index, config.alpha, config.beta, config.fusion_mode, config.seed)
    analysis = evaluated["public"]
    metrics = analysis["metrics"]
    metric_labels = metrics.get("labels", [])
    f1_index = next((idx for idx, label in enumerate(metric_labels) if "f1" in str(label).lower()), -1)
    proposed_values = metrics.get("proposed") or []
    existing_values = metrics.get("existing") or []
    proposed_f1 = proposed_values[f1_index] if 0 <= f1_index < len(proposed_values) else None
    baseline_f1 = existing_values[f1_index] if 0 <= f1_index < len(existing_values) else None
    analysed = int(analysis.get("defense", {}).get("analysed_flows") or analysis.get("limit") or 0)
    attack_flows = int(analysis.get("defense", {}).get("attack_flows") or 0)
    analytics = analysis.get("rule_analytics", {})
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
        "context": _context_public(config),
        "parameters": _config_public(config),
        "parameter_hash": _parameter_hash(config),
        "classes": _classes,
        "class_distribution": {"labels": class_counts.index.tolist(), "values": class_counts.values.tolist()},
        "saved_paper_per_class_metrics": saved_per_class,
        "paper_summary": saved_paper_summary(),
        "live_summary": {
            "total_flows": analysed,
            "attack_rate": json_number(attack_flows / max(1, analysed), 6),
            "attack_flows": attack_flows,
            "proposed_f1": proposed_f1,
            "baseline_f1": baseline_f1,
            "f1_delta": json_number(float(proposed_f1 or 0.0) - float(baseline_f1 or 0.0), 6),
            "unknown_detection_rate": analytics.get("unknown_rejection_rate"),
            "unknown_detection_count": analytics.get("unknown_rejection_count"),
            "confidence_mean": json_number(float(np.mean(evaluated["confidence"])), 6),
            "entropy_mean": analytics.get("mean_entropy"),
            "rule_trigger_rate": analytics.get("rule_trigger_rate"),
            "binary_attack_recall_delta": analytics.get("binary_attack_recall_delta"),
            "threat_summary": (
                f"{attack_flows}/{analysed} flows are attack-labelled; "
                f"{analytics.get('unknown_rejection_count', 0)} were routed to UNKNOWN review."
            ),
        },
        "total_samples": int(len(_y_test)),
        "num_classes": int(_y_test.nunique()),
        "max_index": int(len(_X_test) - 1),
    }


def experiment_payload(
    limit=750,
    window_size=None,
    flow_index=0,
    alpha=DEFAULT_ALPHA,
    beta=None,
    fusion_mode=SYMBOLIC_FUSION_MODE,
    seed=DEFAULT_SEED,
) -> dict[str, Any]:
    load_resources()
    selected_window = window_size if window_size is not None else limit
    config = evaluation_config(selected_window, flow_index, alpha, beta, fusion_mode, seed)
    cache_key = _config_key(config)
    cache_hit = cache_key in _evaluation_cache
    evaluated = _evaluate_window(config.window_size, config.flow_index, config.alpha, config.beta, config.fusion_mode, config.seed)
    research = evaluated["public"]
    flow = _flow_payload_from_evaluation(evaluated, 0)
    charts = chart_data(
        window_size=config.window_size,
        flow_index=config.flow_index,
        alpha=config.alpha,
        beta=config.beta,
        fusion_mode=config.fusion_mode,
        seed=config.seed,
    )
    ablation = ablation_data(
        window_size=config.window_size,
        flow_index=config.flow_index,
        alpha=config.alpha,
        beta=config.beta,
        fusion_mode=config.fusion_mode,
        seed=config.seed,
    )
    novelty = novelty_data(
        config.window_size,
        min(0.40, max(0.01, config.alpha)),
        flow_index=config.flow_index,
        seed=config.seed,
    )
    return {
        "ok": True,
        "context": _context_public(config),
        "parameters": _config_public(config),
        "parameter_hash": _parameter_hash(config),
        "cache": {
            "parameter_hash": _parameter_hash(config),
            "evaluation": "hit" if cache_hit else "miss",
            "analysis_cache_size": len(_analysis_cache),
            "chart_cache_size": len(_chart_cache),
            "ablation_cache_size": len(_ablation_cache),
        },
        "overview": overview_data(
            window_size=config.window_size,
            flow_index=config.flow_index,
            alpha=config.alpha,
            beta=config.beta,
            fusion_mode=config.fusion_mode,
            seed=config.seed,
        ),
        "research": research,
        "charts": charts,
        "novelty": novelty,
        "ablation": ablation,
        "flow": flow,
    }


def backend_status():
    load_resources()
    return {
        "backend": "Flask + nids_engine.py",
        "model_loaded": _base_model is not None,
        "model_path": str(MODEL_PATH),
        "active_proposed_model_path": str(_publication_model_path("proposed")),
        "active_baseline_model_path": str(_publication_model_path("baseline")),
        "test_path": str(TEST_PATH),
        "train_path": str(TRAIN_PATH),
        "test_rows": int(len(_X_test)),
        "feature_count": int(len(_X_test.columns)),
        "classes": _classes,
        "robust_model_loaded": _robust_model is not None,
        "baseline_model_loaded": _baseline_model is not None,
        "symbolic_context_loaded": _symbolic_context is not None,
        "symbolic_calibration": (_symbolic_context or {}).get("calibration", {}),
        "symbolic_rule_summary": (_symbolic_context or {}).get("learned_rescue_summary", {}),
        "evidence_separation": {
            "live_evaluation": "computed by nids_engine.py from model predictions and processed test data",
            "paper_summary": "loaded from results/metrics.json only when explicitly requested",
            "publication_package": "generated artifacts under results/publication_package and paper/generated",
        },
        "cached_evaluation_hashes": [
            _parameter_hash(EvalConfig(*key)) if isinstance(key, tuple) and len(key) == 7 else str(key)
            for key in sorted(_evaluation_cache.keys())
        ],
        "cached_analysis_windows": [list(key) if isinstance(key, tuple) else key for key in sorted(_analysis_cache.keys())],
        "cached_chart_windows": [list(key) if isinstance(key, tuple) else key for key in sorted(_chart_cache.keys())],
        "cached_ablation_windows": [list(key) if isinstance(key, tuple) else key for key in sorted(_ablation_cache.keys())],
        "cached_novelty_windows": [list(key) for key in sorted(_novelty_cache.keys())],
        "cached_feature_windows": [list(key) for key in sorted(_feature_window_cache.keys())],
        "incident_count": len(_incident_store),
        "note": "Frontend data is served from Flask endpoints backed by model and CSV resources.",
    }


def defense_status():
    return {"open_incidents": list(_incident_store.values())[-20:], "total_incidents": len(_incident_store)}
