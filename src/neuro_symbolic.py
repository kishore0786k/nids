from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


BENIGN_LABELS = {"benign", "normal"}
DEFAULT_CLASS_LABELS = (
    "Backdoor",
    "Benign",
    "DoS/DDoS",
    "Injection",
    "Password",
    "Scanning",
    "XSS/MITM",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _sample_get(sample: Mapping[str, Any] | pd.Series, name: str, default: Any = None) -> Any:
    if isinstance(sample, pd.Series):
        return sample.get(name, default)
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return default


def _first_present(
    sample: Mapping[str, Any] | pd.Series,
    names: Sequence[str],
    default: float = 0.0,
) -> float:
    for name in names:
        value = _sample_get(sample, name)
        if value is not None and pd.notna(value):
            return _safe_float(value, default)
    return default


def _as_probability_vector(values: Optional[Sequence[float] | np.ndarray]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    total = float(arr.sum())
    if total > 0:
        arr = arr / total
    return arr


def _is_benign(label: str) -> bool:
    return str(label).strip().lower() in BENIGN_LABELS


def _is_attack(label: str) -> bool:
    return not _is_benign(label)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _threshold_score(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if not np.isfinite(low):
        low = value
    if not np.isfinite(high) or high <= low:
        return 1.0 if value >= low else 0.0
    return _clip01((value - low) / (high - low))


def _resolve_class_labels(
    class_labels: Optional[Sequence[str]],
    rule_context: Optional[Mapping[str, Any]],
    probs: Optional[np.ndarray],
) -> list[str]:
    if class_labels is not None:
        labels = [str(label) for label in class_labels]
    elif rule_context and rule_context.get("class_labels"):
        labels = [str(label) for label in rule_context["class_labels"]]
    elif probs is not None and len(DEFAULT_CLASS_LABELS) == len(probs):
        labels = list(DEFAULT_CLASS_LABELS)
    else:
        labels = list(DEFAULT_CLASS_LABELS)
    if probs is not None and len(labels) != len(probs):
        labels = [str(i) for i in range(len(probs))]
    return labels


def _label_index(labels: Sequence[str], target: str) -> Optional[int]:
    normalized = {str(label).lower(): idx for idx, label in enumerate(labels)}
    return normalized.get(target.lower())


def _engineer_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    in_pkts = col("IN_PKTS")
    out_pkts = col("OUT_PKTS")
    in_bytes = col("IN_BYTES")
    out_bytes = col("OUT_BYTES")
    flow_pkts = col("flow_pkts_s", np.nan).combine_first(col("flow_packetss", np.nan)).combine_first(col("FLOW_PKTS_S", np.nan))
    flow_bytes = col("flow_bytes_s", np.nan).combine_first(col("FLOW_BYTES_S", np.nan))
    flow_duration = col("flow_duration", np.nan).combine_first(col("FLOW_DURATION", np.nan)).combine_first(col("FLOW_DURATION_MILLISECONDS", np.nan))

    duration_parts = pd.concat(
        [
            flow_duration,
            col("DURATION_IN", np.nan),
            col("DURATION_OUT", np.nan),
        ],
        axis=1,
    )

    engineered = pd.DataFrame(index=frame.index)
    engineered["packet_rate"] = flow_pkts.combine_first(in_pkts + out_pkts).fillna(0.0)
    engineered["byte_rate"] = flow_bytes.combine_first(in_bytes + out_bytes).fillna(0.0)
    engineered["duration"] = duration_parts.max(axis=1).fillna(0.0)
    engineered["throughput"] = (
        col("SRC_TO_DST_AVG_THROUGHPUT")
        + col("DST_TO_SRC_AVG_THROUGHPUT")
        + col("SRC_TO_DST_SECOND_BYTES")
        + col("DST_TO_SRC_SECOND_BYTES")
    )
    engineered["sustained_rate"] = pd.concat(
        [engineered["packet_rate"], engineered["byte_rate"], engineered["throughput"]],
        axis=1,
    ).max(axis=1)
    engineered["l7_proto"] = col("L7_PROTO")
    engineered["tcp_flags"] = col("TCP_FLAGS")
    engineered["client_tcp_flags"] = col("CLIENT_TCP_FLAGS")
    engineered["server_tcp_flags"] = col("SERVER_TCP_FLAGS")
    engineered["min_ip_pkt_len"] = col("MIN_IP_PKT_LEN")
    engineered["dns_query_id"] = col("DNS_QUERY_ID")
    engineered["shortest_flow_pkt"] = col("SHORTEST_FLOW_PKT")
    engineered["longest_flow_pkt"] = col("LONGEST_FLOW_PKT")
    engineered["ttl_spread"] = (col("MAX_TTL") - col("MIN_TTL")).abs()
    return engineered.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _engineer_sample(sample: Mapping[str, Any] | pd.Series) -> dict[str, float]:
    frame = pd.DataFrame([dict(sample)]) if not isinstance(sample, pd.Series) else sample.to_frame().T
    return {key: _safe_float(value) for key, value in _engineer_dataframe(frame).iloc[0].items()}


def _quantile(series: pd.Series, q: float, default: float = 0.0) -> float:
    if series.empty:
        return default
    value = float(series.quantile(q))
    return value if np.isfinite(value) else default


def _scan_signature_score(features: Mapping[str, float], thresholds: Mapping[str, float]) -> float:
    l7_score = _threshold_score(features["l7_proto"], thresholds["scan_l7_p85"], thresholds["scan_l7_p95"])
    tcp_score = _threshold_score(features["tcp_flags"], thresholds["tcp_flags_p75"], thresholds["tcp_flags_p95"])
    server_score = _threshold_score(
        features["server_tcp_flags"],
        thresholds["server_tcp_flags_p50"],
        thresholds["server_tcp_flags_p90"],
    )
    min_ip_score = _threshold_score(
        features["min_ip_pkt_len"],
        thresholds["min_ip_pkt_len_p75"],
        thresholds["min_ip_pkt_len_p95"],
    )
    return _clip01(max(l7_score, min(tcp_score, server_score), min_ip_score))


def _dos_burst_score(features: Mapping[str, float], thresholds: Mapping[str, float]) -> float:
    packet_score = _threshold_score(features["packet_rate"], thresholds["packet_rate_p95"], thresholds["packet_rate_p99"])
    byte_score = _threshold_score(features["byte_rate"], thresholds["byte_rate_p95"], thresholds["byte_rate_p99"])
    throughput_score = _threshold_score(features["throughput"], thresholds["throughput_p95"], thresholds["throughput_p99"])
    short_duration = 1.0 if features["duration"] <= thresholds["duration_p80"] else 0.35
    return _clip01(max(packet_score, byte_score, throughput_score) * short_duration)


def _slow_dos_score(features: Mapping[str, float], thresholds: Mapping[str, float]) -> float:
    duration_score = _threshold_score(features["duration"], thresholds["duration_p80"], thresholds["duration_p95"])
    sustained_score = _threshold_score(features["sustained_rate"], thresholds["sustained_rate_p75"], thresholds["sustained_rate_p95"])
    return _clip01(min(duration_score, sustained_score))


def _signature_masks(engineered: pd.DataFrame, thresholds: Mapping[str, float]) -> dict[str, np.ndarray]:
    scan_mask = (
        (engineered["l7_proto"] >= thresholds["scan_l7_p85"])
        | (
            (engineered["tcp_flags"] >= thresholds["tcp_flags_p75"])
            & (engineered["server_tcp_flags"] >= thresholds["server_tcp_flags_p50"])
        )
        | (engineered["min_ip_pkt_len"] >= thresholds["min_ip_pkt_len_p75"])
    ).to_numpy(dtype=bool)
    burst_mask = (
        (engineered["packet_rate"] >= thresholds["packet_rate_p95"])
        | (engineered["byte_rate"] >= thresholds["byte_rate_p95"])
        | (engineered["throughput"] >= thresholds["throughput_p95"])
    ).to_numpy(dtype=bool)
    slow_dos_mask = (
        (engineered["duration"] >= thresholds["duration_p80"])
        & (engineered["sustained_rate"] >= thresholds["sustained_rate_p75"])
    ).to_numpy(dtype=bool)
    return {"scan": scan_mask, "dos": burst_mask | slow_dos_mask}


def _calibrate_target_threshold(
    target: str,
    probs: np.ndarray,
    labels: Sequence[str],
    base_predictions: Sequence[str],
    y_true: Optional[Sequence[str]],
    signature_mask: np.ndarray,
    fallback: float,
) -> float:
    target_idx = _label_index(labels, target)
    if target_idx is None:
        return fallback

    base_predictions = np.asarray([str(label) for label in base_predictions])
    pred_benign = np.asarray([_is_benign(label) for label in base_predictions], dtype=bool)
    eligible = pred_benign & signature_mask
    if not eligible.any():
        return fallback

    target_probs = probs[:, target_idx]
    eligible_scores = target_probs[eligible]
    candidates = np.unique(np.quantile(eligible_scores, np.linspace(0.50, 0.97, 12)))
    candidates = candidates[np.isfinite(candidates)]
    if candidates.size == 0:
        return fallback

    if y_true is None:
        return float(np.quantile(eligible_scores, 0.80))

    truth = np.asarray([str(label) for label in y_true])
    best_threshold = float(candidates[-1])
    best_score = -np.inf
    for threshold in candidates:
        changed = eligible & (target_probs >= threshold)
        true_target = changed & (truth == target)
        false_target = changed & (truth != target)
        corrected = int(true_target.sum())
        incorrect = int(false_target.sum())
        score = corrected - incorrect
        if score > best_score or (score == best_score and threshold < best_threshold):
            best_score = score
            best_threshold = float(threshold)

    return best_threshold if np.isfinite(best_threshold) else fallback


def _calibrate_attack_mass_threshold(
    probs: np.ndarray,
    labels: Sequence[str],
    base_predictions: Sequence[str],
    y_true: Optional[Sequence[str]],
    fallback: float = 0.50,
) -> float:
    benign_idx = _label_index(labels, "Benign")
    if benign_idx is None:
        return fallback

    base_predictions = np.asarray([str(label) for label in base_predictions])
    pred_benign = np.asarray([_is_benign(label) for label in base_predictions], dtype=bool)
    if not pred_benign.any():
        return fallback

    attack_indices = [idx for idx, label in enumerate(labels) if _is_attack(label)]
    if not attack_indices:
        return fallback

    attack_mass = 1.0 - probs[:, benign_idx]
    eligible_scores = attack_mass[pred_benign]
    finite_scores = eligible_scores[np.isfinite(eligible_scores)]
    if finite_scores.size == 0:
        return fallback

    if y_true is None:
        return _ceil_probability_threshold(max(fallback, float(np.quantile(finite_scores, 0.95))))

    truth = np.asarray([str(label) for label in y_true])
    best_attack_idx = np.argmax(
        np.where(np.asarray([_is_attack(label) for label in labels]), probs, -1.0),
        axis=1,
    )
    best_attack_labels = np.asarray([labels[int(idx)] for idx in best_attack_idx])
    grid = np.concatenate(
        [
            np.linspace(0.25, 0.75, 11),
            np.quantile(finite_scores, np.linspace(0.75, 0.99, 8)),
        ]
    )
    candidates = np.unique(grid[np.isfinite(grid)])
    candidates = candidates[(candidates > 0.0) & (candidates < 0.95)]
    if candidates.size == 0:
        return fallback

    best_threshold = fallback
    best_score = -np.inf
    best_precision = -np.inf
    for threshold in candidates:
        changed = pred_benign & (attack_mass >= threshold)
        changed_count = int(changed.sum())
        if changed_count == 0:
            continue
        exact_corrections = int(np.sum(changed & (truth == best_attack_labels)))
        benign_false_alarms = int(np.sum(changed & np.asarray([_is_benign(label) for label in truth], dtype=bool)))
        attack_rescues = int(np.sum(changed & np.asarray([_is_attack(label) for label in truth], dtype=bool)))
        precision = exact_corrections / max(1, changed_count)
        score = exact_corrections - benign_false_alarms + 0.05 * attack_rescues
        if score > best_score or (score == best_score and precision > best_precision):
            best_score = score
            best_precision = precision
            best_threshold = float(threshold)

    if best_score <= 0:
        return _ceil_probability_threshold(max(fallback, float(np.quantile(finite_scores, 0.95))))
    return _ceil_probability_threshold(best_threshold)


def _ceil_probability_threshold(value: float, step: float = 0.05) -> float:
    if not np.isfinite(value):
        return 0.5
    return float(np.clip(np.ceil(value / step) * step, 0.0, 0.95))


def build_symbolic_context(
    reference_X: pd.DataFrame,
    reference_y: Optional[Sequence[str]] = None,
    class_labels: Optional[Sequence[str]] = None,
    predicted_probs: Optional[np.ndarray] = None,
    base_predictions: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Fit data-driven symbolic thresholds from a reference split.

    The reference split should be training or calibration data, not the held-out
    test set. Feature thresholds are percentile based, and optional probability
    thresholds are calibrated against known weak false-negative patterns.
    """
    if reference_X.empty:
        raise ValueError("reference_X must contain at least one row.")

    labels = [str(label) for label in (class_labels or DEFAULT_CLASS_LABELS)]
    engineered = _engineer_dataframe(reference_X)
    thresholds = {
        "packet_rate_p75": _quantile(engineered["packet_rate"], 0.75),
        "packet_rate_p95": _quantile(engineered["packet_rate"], 0.95),
        "packet_rate_p99": _quantile(engineered["packet_rate"], 0.99),
        "byte_rate_p75": _quantile(engineered["byte_rate"], 0.75),
        "byte_rate_p95": _quantile(engineered["byte_rate"], 0.95),
        "byte_rate_p99": _quantile(engineered["byte_rate"], 0.99),
        "throughput_p75": _quantile(engineered["throughput"], 0.75),
        "throughput_p95": _quantile(engineered["throughput"], 0.95),
        "throughput_p99": _quantile(engineered["throughput"], 0.99),
        "duration_p50": _quantile(engineered["duration"], 0.50),
        "duration_p80": _quantile(engineered["duration"], 0.80),
        "duration_p90": _quantile(engineered["duration"], 0.90),
        "duration_p95": _quantile(engineered["duration"], 0.95),
        "sustained_rate_p75": _quantile(engineered["sustained_rate"], 0.75),
        "sustained_rate_p95": _quantile(engineered["sustained_rate"], 0.95),
        "scan_l7_p85": _quantile(engineered["l7_proto"], 0.85),
        "scan_l7_p95": _quantile(engineered["l7_proto"], 0.95),
        "tcp_flags_p75": _quantile(engineered["tcp_flags"], 0.75),
        "tcp_flags_p95": _quantile(engineered["tcp_flags"], 0.95),
        "server_tcp_flags_p50": _quantile(engineered["server_tcp_flags"], 0.50),
        "server_tcp_flags_p90": _quantile(engineered["server_tcp_flags"], 0.90),
        "min_ip_pkt_len_p75": _quantile(engineered["min_ip_pkt_len"], 0.75),
        "min_ip_pkt_len_p95": _quantile(engineered["min_ip_pkt_len"], 0.95),
        "dns_query_id_p90": _quantile(engineered["dns_query_id"], 0.90),
        "ttl_spread_p95": _quantile(engineered["ttl_spread"], 0.95, default=5.0),
    }
    probability_thresholds = {
        "scanning_from_benign": 0.25,
        "dos_from_benign": 0.20,
        "attack_mass_from_benign": 0.50,
    }

    probs = None if predicted_probs is None else np.asarray(predicted_probs, dtype=float)
    if probs is not None and probs.ndim == 2 and probs.shape[1] == len(labels):
        if base_predictions is None:
            base_predictions = [labels[int(idx)] for idx in np.argmax(probs, axis=1)]
        masks = _signature_masks(engineered, thresholds)
        probability_thresholds["scanning_from_benign"] = _ceil_probability_threshold(_calibrate_target_threshold(
            "Scanning",
            probs,
            labels,
            base_predictions,
            reference_y,
            masks["scan"],
            probability_thresholds["scanning_from_benign"],
        ))
        probability_thresholds["dos_from_benign"] = _ceil_probability_threshold(_calibrate_target_threshold(
            "DoS/DDoS",
            probs,
            labels,
            base_predictions,
            reference_y,
            masks["dos"],
            probability_thresholds["dos_from_benign"],
        ))

        probability_thresholds["attack_mass_from_benign"] = _calibrate_attack_mass_threshold(
            probs,
            labels,
            base_predictions,
            reference_y,
            probability_thresholds["attack_mass_from_benign"],
        )

    return {
        "class_labels": labels,
        "thresholds": thresholds,
        "probability_thresholds": probability_thresholds,
    }


@lru_cache(maxsize=1)
def load_default_symbolic_context() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "train_processed.csv"
    test_path = project_root / "data" / "test_processed.csv"
    data_path = train_path if train_path.exists() else test_path
    if not data_path.exists():
        empty = pd.DataFrame([{name: 0.0 for name in ("IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES")}])
        return build_symbolic_context(empty, class_labels=DEFAULT_CLASS_LABELS)
    df = pd.read_csv(data_path)
    X = df.drop(columns=["label"], errors="ignore")
    y = df["label"].astype(str).tolist() if "label" in df else None
    labels = sorted(df["label"].astype(str).unique().tolist()) if "label" in df else list(DEFAULT_CLASS_LABELS)
    return build_symbolic_context(X, reference_y=y, class_labels=labels)


def _add_rule_score(
    rule_scores: np.ndarray,
    labels: Sequence[str],
    target_label: str,
    strength: float,
) -> None:
    idx = _label_index(labels, target_label)
    if idx is not None:
        rule_scores[idx] = max(rule_scores[idx], _clip01(strength))


def _normalize_rule_scores(rule_scores: np.ndarray) -> np.ndarray:
    total = float(rule_scores.sum())
    if total <= 0:
        return rule_scores
    return rule_scores / total


def _override_allowed(
    current_label: str,
    target_label: str,
    confidence: float,
    strength: float,
    confidence_threshold: float,
    strong_rule_threshold: float,
) -> bool:
    if str(current_label) == str(target_label):
        return False
    if confidence < confidence_threshold and strength >= strong_rule_threshold - 0.05:
        return True
    if strength >= strong_rule_threshold:
        return True
    if _is_benign(current_label) and _is_attack(target_label) and strength >= strong_rule_threshold - 0.05:
        return True
    return False


def apply_symbolic_rules(
    sample: Mapping[str, Any] | pd.Series,
    predicted_label: str,
    predicted_probs: Optional[Sequence[float] | np.ndarray] = None,
    *,
    class_labels: Optional[Sequence[str]] = None,
    rule_context: Optional[Mapping[str, Any]] = None,
    fusion_mode: str = "hard",
    alpha: float = 0.65,
    beta: Optional[float] = None,
    confidence_threshold: float = 0.70,
    strong_rule_threshold: float = 0.85,
    adversarial_probs: Optional[Sequence[float] | np.ndarray] = None,
    gnn_anomaly_score: Optional[float] = None,
    adversarial_threshold: float = 0.15,
) -> tuple[str, list[dict[str, Any]], float]:
    """Apply confidence-aware symbolic NIDS rules to a neural prediction.

    Returns `(final_label, fired_rules, rule_strength_score)`. `fusion_mode`
    can be `hard` for direct overrides or `soft` for
    `alpha * neural_probs + beta * rule_scores`.
    """
    context = dict(rule_context or load_default_symbolic_context())
    thresholds = context.get("thresholds", {})
    probability_thresholds = context.get("probability_thresholds", {})
    probs = _as_probability_vector(predicted_probs)
    labels = _resolve_class_labels(class_labels, context, probs)
    confidence = float(np.max(probs)) if probs is not None else 1.0
    beta_value = (1.0 - alpha) if beta is None else beta
    alpha_value = float(alpha)
    beta_value = float(beta_value)
    if alpha_value < 0 or beta_value < 0 or alpha_value + beta_value <= 0:
        alpha_value, beta_value = 0.65, 0.35

    features = _engineer_sample(sample)
    final_label = str(predicted_label)
    fired_rules: list[dict[str, Any]] = []
    rule_scores = np.zeros(len(labels), dtype=float)

    def fire(rule_id: str, target_label: str, strength: float, reason: str, evidence: Mapping[str, Any]) -> None:
        strength_value = _clip01(strength)
        fired_rules.append(
            {
                "rule_id": rule_id,
                "old_label": str(predicted_label),
                "new_label": str(target_label),
                "strength": round(strength_value, 6),
                "reason": reason,
                "evidence": dict(evidence),
            }
        )
        _add_rule_score(rule_scores, labels, target_label, strength_value)

    burst_score = _dos_burst_score(features, thresholds)
    if burst_score >= 0.65:
        fire(
            "R1_HIGH_PACKET_RATE_BURST",
            "DoS/DDoS",
            0.70 + 0.09 * burst_score,
            "High packet or byte-rate burst exceeds the learned upper-tail traffic profile.",
            {
                "packet_rate": round(features["packet_rate"], 6),
                "byte_rate": round(features["byte_rate"], 6),
                "throughput": round(features["throughput"], 6),
                "burst_score": round(burst_score, 6),
            },
        )

    slow_dos_score = _slow_dos_score(features, thresholds)
    dos_idx = _label_index(labels, "DoS/DDoS")
    dos_prob = float(probs[dos_idx]) if probs is not None and dos_idx is not None else 0.0
    dos_prob_threshold = float(probability_thresholds.get("dos_from_benign", 0.20))
    if slow_dos_score >= 0.55 and (dos_prob >= dos_prob_threshold or confidence < confidence_threshold):
        fire(
            "R2_LONG_DURATION_SUSTAINED_RATE",
            "DoS/DDoS",
            0.70 + 0.08 * slow_dos_score,
            "Long duration with sustained rate matches the learned slow-DoS tail profile.",
            {
                "duration": round(features["duration"], 6),
                "sustained_rate": round(features["sustained_rate"], 6),
                "dos_probability": round(dos_prob, 6),
                "slow_dos_score": round(slow_dos_score, 6),
            },
        )

    if probs is not None and _is_benign(predicted_label):
        benign_idx = _label_index(labels, "Benign")
        attack_indices = [idx for idx, label in enumerate(labels) if _is_attack(label)]
        attack_mass = float(1.0 - probs[benign_idx]) if benign_idx is not None else float(np.sum(probs[attack_indices]))
        scan_idx = _label_index(labels, "Scanning")
        scan_prob = float(probs[scan_idx]) if scan_idx is not None else 0.0
        scan_prob_threshold = float(probability_thresholds.get("scanning_from_benign", 0.25))
        scan_score = _scan_signature_score(features, thresholds)

        if scan_idx is not None and scan_score >= 0.50 and scan_prob >= scan_prob_threshold:
            prob_excess = _threshold_score(scan_prob, scan_prob_threshold, 1.0)
            fire(
                "R3_SUSPICIOUS_BENIGN_SCANNING",
                "Scanning",
                0.68 + 0.08 * scan_score + 0.03 * prob_excess,
                "Benign prediction has a scanning-like signature and elevated scanning probability.",
                {
                    "confidence": round(confidence, 6),
                    "scan_probability": round(scan_prob, 6),
                    "scan_probability_threshold": round(scan_prob_threshold, 6),
                    "scan_signature_score": round(scan_score, 6),
                },
            )
        attack_mass_threshold = float(probability_thresholds.get("attack_mass_from_benign", 0.50))
        if attack_indices and attack_mass >= attack_mass_threshold:
            best_attack_idx = max(attack_indices, key=lambda idx: probs[idx])
            best_attack_label = labels[best_attack_idx]
            anomaly_score = max(scan_score, burst_score, slow_dos_score)
            mass_excess = _threshold_score(attack_mass, attack_mass_threshold, 1.0)
            fire(
                "R4_SUSPICIOUS_BENIGN_ATTACK_MASS",
                best_attack_label,
                0.86 + 0.10 * mass_excess + 0.02 * anomaly_score,
                "Benign prediction has calibrated high aggregate attack probability.",
                {
                    "confidence": round(confidence, 6),
                    "attack_mass": round(attack_mass, 6),
                    "attack_mass_threshold": round(attack_mass_threshold, 6),
                    "target_attack": best_attack_label,
                    "anomaly_score": round(anomaly_score, 6),
                },
            )

    original = _as_probability_vector(predicted_probs)
    perturbed = _as_probability_vector(adversarial_probs)
    if original is not None and perturbed is not None and original.shape == perturbed.shape:
        perturbation_score = float(np.max(np.abs(perturbed - original)))
        if perturbation_score > adversarial_threshold:
            fire(
                "R5_ADVERSARIAL_PROBABILITY_DRIFT",
                f"{final_label}_ADV",
                min(1.0, perturbation_score / max(adversarial_threshold, 1e-9)),
                "Adversarial probability drift exceeds the configured robustness tolerance.",
                {
                    "max_delta": round(perturbation_score, 6),
                    "threshold": round(adversarial_threshold, 6),
                },
            )

    anomaly = _safe_float(gnn_anomaly_score, default=-1.0) if gnn_anomaly_score is not None else None
    ttl_var = _first_present(sample, ("ttl_variance", "TTL_VARIANCE"), default=features.get("ttl_spread", 0.0))
    ttl_threshold = max(5.0, float(thresholds.get("ttl_spread_p95", 5.0)))
    if anomaly is not None and anomaly > 0.8 and ttl_var > ttl_threshold:
        fire(
            "R6_ZERO_DAY_ANOMALY",
            "ZeroDay",
            min(1.0, 0.80 + 0.20 * anomaly),
            "Graph anomaly score and TTL variation exceed the learned novelty profile.",
            {
                "gnn_anomaly_score": round(anomaly, 6),
                "ttl_variance": round(ttl_var, 6),
                "ttl_threshold": round(ttl_threshold, 6),
            },
        )

    active_rules = [rule for rule in fired_rules if rule["rule_id"] != "NONE"]
    strongest_rule = max(active_rules, key=lambda rule: float(rule["strength"]), default=None)
    strongest_strength = float(strongest_rule["strength"]) if strongest_rule else 0.0

    if strongest_rule is not None:
        mode = str(fusion_mode).strip().lower()
        if mode == "soft" and probs is not None and len(rule_scores) == len(probs):
            normalized_rule_scores = _normalize_rule_scores(rule_scores)
            fused = alpha_value * probs + beta_value * normalized_rule_scores
            fused = fused / max(float(fused.sum()), 1e-12)
            candidate_label = labels[int(np.argmax(fused))]
            candidate_strength = strongest_strength
            if _override_allowed(
                final_label,
                candidate_label,
                confidence,
                candidate_strength,
                confidence_threshold,
                strong_rule_threshold,
            ):
                final_label = candidate_label
        else:
            candidate_label = str(strongest_rule["new_label"])
            if _override_allowed(
                final_label,
                candidate_label,
                confidence,
                strongest_strength,
                confidence_threshold,
                strong_rule_threshold,
            ):
                final_label = candidate_label

    for rule in fired_rules:
        rule["applied"] = bool(str(rule["new_label"]) == final_label and final_label != str(predicted_label))

    if not fired_rules:
        fired_rules.append(
            {
                "rule_id": "NONE",
                "old_label": str(predicted_label),
                "new_label": final_label,
                "strength": 0.0,
                "reason": "No symbolic rule triggered; neural prediction retained.",
                "evidence": {"confidence": round(confidence, 6)},
                "applied": False,
            }
        )

    return final_label, fired_rules, strongest_strength


def apply_symbolic_rules_batch(
    samples: pd.DataFrame,
    predicted_labels: Sequence[str],
    predicted_probs: np.ndarray,
    *,
    class_labels: Sequence[str],
    rule_context: Mapping[str, Any],
    fusion_mode: str = "soft",
    alpha: float = 0.65,
    beta: Optional[float] = None,
    confidence_threshold: float = 0.70,
    strong_rule_threshold: float = 0.85,
) -> tuple[np.ndarray, list[list[dict[str, Any]]], list[float]]:
    """Vectorized publication evaluator for the core symbolic NIDS rules."""
    labels = [str(label) for label in class_labels]
    probs = np.asarray(predicted_probs, dtype=float)
    base = np.asarray([str(label) for label in predicted_labels])
    final = base.copy()
    features = _engineer_dataframe(samples)
    thresholds = rule_context.get("thresholds", {})
    prob_thresholds = rule_context.get("probability_thresholds", {})
    beta_value = 1.0 - alpha if beta is None else float(beta)

    confidence = np.max(probs, axis=1)
    benign_mask = np.asarray([_is_benign(label) for label in base], dtype=bool)
    traces: list[list[dict[str, Any]]] = [[] for _ in range(len(samples))]
    strengths = np.zeros(len(samples), dtype=float)

    def add_rule(row_mask: np.ndarray, rule_id: str, target: str, strength_values: np.ndarray, reason: str) -> None:
        for idx in np.flatnonzero(row_mask):
            strength = float(strength_values[idx]) if np.ndim(strength_values) else float(strength_values)
            strengths[idx] = max(strengths[idx], strength)
            traces[idx].append({
                "rule_id": rule_id,
                "old_label": str(base[idx]),
                "new_label": target,
                "strength": round(_clip01(strength), 6),
                "reason": reason,
                "evidence": {},
                "applied": False,
            })

    def threshold_vec(values: pd.Series, low: float, high: float) -> np.ndarray:
        values_arr = values.to_numpy(dtype=float)
        if not np.isfinite(high) or high <= low:
            return (values_arr >= low).astype(float)
        return np.clip((values_arr - low) / (high - low), 0.0, 1.0)

    burst_score = np.maximum.reduce([
        threshold_vec(features["packet_rate"], thresholds["packet_rate_p95"], thresholds["packet_rate_p99"]),
        threshold_vec(features["byte_rate"], thresholds["byte_rate_p95"], thresholds["byte_rate_p99"]),
        threshold_vec(features["throughput"], thresholds["throughput_p95"], thresholds["throughput_p99"]),
    ])
    burst_score *= np.where(features["duration"].to_numpy() <= thresholds["duration_p80"], 1.0, 0.35)
    r1_mask = burst_score >= 0.65
    add_rule(
        r1_mask,
        "R1_HIGH_PACKET_RATE_BURST",
        "DoS/DDoS",
        0.70 + 0.09 * burst_score,
        "High packet or byte-rate burst exceeds the learned upper-tail traffic profile.",
    )

    duration_score = threshold_vec(features["duration"], thresholds["duration_p80"], thresholds["duration_p95"])
    sustained_score = threshold_vec(features["sustained_rate"], thresholds["sustained_rate_p75"], thresholds["sustained_rate_p95"])
    slow_dos_score = np.minimum(duration_score, sustained_score)
    dos_idx = _label_index(labels, "DoS/DDoS")
    dos_prob = probs[:, dos_idx] if dos_idx is not None else np.zeros(len(samples))
    r2_mask = (slow_dos_score >= 0.55) & (
        (dos_prob >= float(prob_thresholds.get("dos_from_benign", 0.20))) | (confidence < confidence_threshold)
    )
    add_rule(
        r2_mask,
        "R2_LONG_DURATION_SUSTAINED_RATE",
        "DoS/DDoS",
        0.70 + 0.08 * slow_dos_score,
        "Long duration with sustained rate matches the learned slow-DoS tail profile.",
    )

    l7_score = threshold_vec(features["l7_proto"], thresholds["scan_l7_p85"], thresholds["scan_l7_p95"])
    tcp_score = threshold_vec(features["tcp_flags"], thresholds["tcp_flags_p75"], thresholds["tcp_flags_p95"])
    server_score = threshold_vec(features["server_tcp_flags"], thresholds["server_tcp_flags_p50"], thresholds["server_tcp_flags_p90"])
    min_ip_score = threshold_vec(features["min_ip_pkt_len"], thresholds["min_ip_pkt_len_p75"], thresholds["min_ip_pkt_len_p95"])
    scan_score = np.maximum.reduce([l7_score, np.minimum(tcp_score, server_score), min_ip_score])
    scan_idx = _label_index(labels, "Scanning")
    benign_idx = _label_index(labels, "Benign")
    scan_prob = probs[:, scan_idx] if scan_idx is not None else np.zeros(len(samples))
    scan_threshold = float(prob_thresholds.get("scanning_from_benign", 0.25))
    r3_mask = benign_mask & (scan_score >= 0.50) & (scan_prob >= scan_threshold)
    r3_strength = 0.68 + 0.08 * scan_score + 0.03 * threshold_vec(pd.Series(scan_prob), scan_threshold, 1.0)
    add_rule(
        r3_mask,
        "R3_SUSPICIOUS_BENIGN_SCANNING",
        "Scanning",
        r3_strength,
        "Benign prediction has a scanning-like signature and elevated scanning probability.",
    )

    if benign_idx is not None:
        attack_mass = 1.0 - probs[:, benign_idx]
    else:
        attack_indices = [idx for idx, label in enumerate(labels) if _is_attack(label)]
        attack_mass = probs[:, attack_indices].sum(axis=1)
    anomaly_score = np.maximum.reduce([scan_score, burst_score, slow_dos_score])
    attack_mass_threshold = float(prob_thresholds.get("attack_mass_from_benign", 0.50))
    r4_mask = benign_mask & (attack_mass >= attack_mass_threshold)
    best_attack_idx = np.argmax(np.where(np.asarray([_is_attack(label) for label in labels]), probs, -1.0), axis=1)
    mass_excess = np.clip((attack_mass - attack_mass_threshold) / max(1.0 - attack_mass_threshold, 1e-9), 0.0, 1.0)
    r4_strength = 0.86 + 0.10 * mass_excess + 0.02 * anomaly_score
    for idx in np.flatnonzero(r4_mask):
        target = labels[int(best_attack_idx[idx])]
        strength = float(r4_strength[idx])
        strengths[idx] = max(strengths[idx], strength)
        traces[idx].append({
            "rule_id": "R4_SUSPICIOUS_BENIGN_ATTACK_MASS",
            "old_label": str(base[idx]),
            "new_label": target,
            "strength": round(_clip01(strength), 6),
            "reason": "Benign prediction has calibrated high aggregate attack probability.",
            "evidence": {
                "attack_mass": round(float(attack_mass[idx]), 6),
                "attack_mass_threshold": round(attack_mass_threshold, 6),
                "target_attack": target,
            },
            "applied": False,
        })

    if scan_idx is not None and benign_idx is not None:
        if str(fusion_mode).lower() == "soft":
            fused_scan = alpha * probs[:, scan_idx] + beta_value
            fused_benign = alpha * probs[:, benign_idx]
            scan_wins = fused_scan > fused_benign
        else:
            scan_wins = np.ones(len(samples), dtype=bool)
        apply_scan = r3_mask & scan_wins & (r3_strength >= strong_rule_threshold - 0.05)
        final[apply_scan] = "Scanning"
        for idx in np.flatnonzero(apply_scan):
            for rule in traces[idx]:
                if rule["rule_id"] == "R3_SUSPICIOUS_BENIGN_SCANNING":
                    rule["applied"] = True

    apply_attack_mass = r4_mask & (r4_strength >= strong_rule_threshold)
    for idx in np.flatnonzero(apply_attack_mass):
        final[idx] = labels[int(best_attack_idx[idx])]
        for rule in traces[idx]:
            if rule["rule_id"] == "R4_SUSPICIOUS_BENIGN_ATTACK_MASS":
                rule["applied"] = True
            elif rule.get("applied") and rule.get("new_label") != final[idx]:
                rule["applied"] = False

    for idx, rules in enumerate(traces):
        if not rules:
            traces[idx].append({
                "rule_id": "NONE",
                "old_label": str(base[idx]),
                "new_label": str(final[idx]),
                "strength": 0.0,
                "reason": "No symbolic rule triggered; neural prediction retained.",
                "evidence": {},
                "applied": False,
            })

    return final, traces, strengths.tolist()
