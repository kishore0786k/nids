from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.neuro_symbolic import apply_symbolic_rules_batch, build_symbolic_context


UNKNOWN_LABEL = "UNKNOWN"
TAU_GRID = tuple(round(value, 2) for value in np.arange(0.20, 0.8001, 0.05))
TEMPERATURE_GRID = tuple(round(value, 2) for value in np.arange(0.50, 5.0001, 0.05))
ALPHA_GRID = (0.55, 0.65, 0.75, 0.85, 0.95)
CONFIDENCE_THRESHOLD_GRID = (0.55, 0.65, 0.75)
STRONG_RULE_THRESHOLD_GRID = (0.82, 0.88, 0.94)


@dataclass(frozen=True)
class RuleFusionParams:
    fusion_mode: str = "soft"
    alpha: float = 0.85
    beta: float = 0.15
    confidence_threshold: float = 0.65
    strong_rule_threshold: float = 0.88
    validation_macro_f1: float = 0.0

    def call_kwargs(self) -> dict[str, float | str]:
        return {
            "fusion_mode": self.fusion_mode,
            "alpha": self.alpha,
            "beta": self.beta,
            "confidence_threshold": self.confidence_threshold,
            "strong_rule_threshold": self.strong_rule_threshold,
        }


@dataclass(frozen=True)
class TauSelection:
    tau: float
    score: float
    validation_macro_f1_with_unknown: float
    validation_rejection_rate: float
    validation_benign_unknown_fpr: float


def json_ready_dataclass(value: Any) -> dict[str, Any]:
    return asdict(value)


def model_input(model: Any, frame: pd.DataFrame) -> pd.DataFrame | np.ndarray:
    return frame if getattr(model, "feature_names_in_", None) is not None else frame.to_numpy()


def class_predictions(probabilities: np.ndarray, class_labels: Sequence[str]) -> np.ndarray:
    labels = [str(label) for label in class_labels]
    return np.asarray([labels[int(idx)] for idx in np.argmax(probabilities, axis=1)])


def validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    class_count = int(y.nunique())
    validation_count = int(np.ceil(len(y) * validation_size))
    stratify = (
        y
        if y.value_counts().min() >= 2
        and validation_count >= class_count
        and len(y) - validation_count >= class_count
        else None
    )
    X_ref, X_val, y_ref, y_val = train_test_split(
        X,
        y.astype(str),
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )
    return (
        X_ref.reset_index(drop=True),
        y_ref.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_val.reset_index(drop=True),
    )


def build_context_from_split(
    model: Any,
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
    class_labels: Sequence[str],
) -> dict[str, Any]:
    probabilities = model.predict_proba(model_input(model, X_ref))
    base_predictions = class_predictions(probabilities, class_labels)
    return build_symbolic_context(
        X_ref,
        reference_y=y_ref.astype(str).tolist(),
        class_labels=[str(label) for label in class_labels],
        predicted_probs=probabilities,
        base_predictions=base_predictions,
    )


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    logits = np.log(probs)
    scaled = logits / max(float(temperature), 1e-6)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)


def fit_temperature(
    probabilities: np.ndarray,
    y_true: Sequence[str],
    class_labels: Sequence[str],
    grid: Iterable[float] = TEMPERATURE_GRID,
) -> tuple[float, float]:
    label_to_idx = {str(label): idx for idx, label in enumerate(class_labels)}
    indices = np.asarray([label_to_idx.get(str(label), -1) for label in y_true], dtype=int)
    valid = indices >= 0
    if not valid.any():
        return 1.0, 0.0

    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in grid:
        calibrated = apply_temperature(probabilities[valid], float(temperature))
        chosen = np.clip(calibrated[np.arange(valid.sum()), indices[valid]], 1e-12, 1.0)
        nll = float(-np.mean(np.log(chosen)))
        if nll < best_nll:
            best_temperature = float(temperature)
            best_nll = nll
    return best_temperature, best_nll


def closed_set_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def benign_mask(y_true: Sequence[str]) -> np.ndarray:
    return np.asarray([str(label).lower() in {"benign", "normal"} for label in y_true], dtype=bool)


def false_positive_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    benign = benign_mask(y_true)
    if not benign.any():
        return 0.0
    predicted_attack = np.asarray([str(label).lower() not in {"benign", "normal"} for label in y_pred], dtype=bool)
    return float(np.mean(predicted_attack[benign]))


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
        if mask.any():
            ece += float(mask.sum() / n) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def tune_rule_fusion(
    X_val: pd.DataFrame,
    y_val: Sequence[str],
    base_predictions: np.ndarray,
    probabilities: np.ndarray,
    class_labels: Sequence[str],
    rule_context: Mapping[str, Any],
) -> RuleFusionParams:
    best = RuleFusionParams()
    best_key = (-1.0, -best.alpha)
    labels = [str(label) for label in class_labels]

    for alpha in ALPHA_GRID:
        beta = round(1.0 - alpha, 6)
        for confidence_threshold in CONFIDENCE_THRESHOLD_GRID:
            for strong_rule_threshold in STRONG_RULE_THRESHOLD_GRID:
                predictions, _, _ = apply_symbolic_rules_batch(
                    X_val,
                    base_predictions,
                    probabilities,
                    class_labels=labels,
                    rule_context=rule_context,
                    fusion_mode="soft",
                    alpha=float(alpha),
                    beta=beta,
                    confidence_threshold=float(confidence_threshold),
                    strong_rule_threshold=float(strong_rule_threshold),
                )
                macro_f1 = float(f1_score(y_val, predictions, average="macro", zero_division=0))
                key = (macro_f1, -abs(float(alpha) - 0.85))
                if key > best_key:
                    best_key = key
                    best = RuleFusionParams(
                        fusion_mode="soft",
                        alpha=float(alpha),
                        beta=beta,
                        confidence_threshold=float(confidence_threshold),
                        strong_rule_threshold=float(strong_rule_threshold),
                        validation_macro_f1=macro_f1,
                    )
    return best


def apply_rules_closed_set(
    X: pd.DataFrame,
    base_predictions: np.ndarray,
    probabilities: np.ndarray,
    class_labels: Sequence[str],
    rule_context: Mapping[str, Any],
    params: RuleFusionParams,
) -> np.ndarray:
    predictions, _, _ = apply_symbolic_rules_batch(
        X,
        base_predictions,
        probabilities,
        class_labels=[str(label) for label in class_labels],
        rule_context=rule_context,
        **params.call_kwargs(),
    )
    return np.asarray([str(label) for label in predictions])


def tune_tau(
    y_true: Sequence[str],
    closed_predictions: np.ndarray,
    probabilities: np.ndarray,
    class_labels: Sequence[str],
    grid: Iterable[float] = TAU_GRID,
) -> TauSelection:
    confidence = np.max(probabilities, axis=1)
    labels_with_unknown = sorted(set(str(label) for label in class_labels) | {UNKNOWN_LABEL})
    true_arr = np.asarray([str(label) for label in y_true])
    benign = benign_mask(true_arr)
    best: TauSelection | None = None
    for tau in grid:
        rejected = confidence < float(tau)
        abstention_predictions = closed_predictions.astype(object).copy()
        abstention_predictions[rejected] = UNKNOWN_LABEL
        macro_f1_unknown = float(
            f1_score(true_arr, abstention_predictions.astype(str), labels=labels_with_unknown, average="macro", zero_division=0)
        )
        rejection_rate = float(np.mean(rejected)) if len(rejected) else 0.0
        benign_unknown_fpr = float(np.mean(rejected[benign])) if benign.any() else 0.0
        score = macro_f1_unknown - 0.10 * benign_unknown_fpr
        candidate = TauSelection(
            tau=float(tau),
            score=score,
            validation_macro_f1_with_unknown=macro_f1_unknown,
            validation_rejection_rate=rejection_rate,
            validation_benign_unknown_fpr=benign_unknown_fpr,
        )
        if best is None or (candidate.score, -candidate.tau) > (best.score, -best.tau):
            best = candidate
    return best or TauSelection(0.20, 0.0, 0.0, 0.0, 0.0)


def abstention_metrics(
    y_true: Sequence[str],
    closed_predictions: Sequence[str],
    probabilities: np.ndarray,
    tau: float,
) -> dict[str, float]:
    confidence = np.max(probabilities, axis=1)
    rejected = confidence < tau
    benign = benign_mask(y_true)
    accepted = ~rejected
    accepted_macro_f1 = (
        float(f1_score(np.asarray(y_true)[accepted], np.asarray(closed_predictions)[accepted], average="macro", zero_division=0))
        if accepted.any()
        else 0.0
    )
    return {
        "unknown_rejection_rate": float(np.mean(rejected)) if len(rejected) else 0.0,
        "benign_unknown_false_positive_rate": float(np.mean(rejected[benign])) if benign.any() else 0.0,
        "accepted_macro_f1": accepted_macro_f1,
        "accepted_coverage": float(np.mean(accepted)) if len(accepted) else 0.0,
    }
