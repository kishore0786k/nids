from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _first_present(sample: pd.Series, names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        if name in sample and pd.notna(sample.get(name)):
            return _safe_float(sample.get(name), default)
    return default


def _as_probability_vector(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    total = float(arr.sum())
    if total > 0:
        arr = arr / total
    return arr


def apply_symbolic_rules(
    sample: pd.Series,
    predicted_label: str,
    predicted_probs: Optional[np.ndarray] = None,
    adversarial_probs: Optional[np.ndarray] = None,
    gnn_anomaly_score: Optional[float] = None,
    adversarial_threshold: float = 0.15,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply auditable symbolic security rules to a neural prediction."""
    packet_rate = _first_present(sample, ("flow_pkts_s", "flow_packetss", "FLOW_PKTS_S", "IN_PKTS"))
    byte_rate = _first_present(sample, ("flow_bytes_s", "FLOW_BYTES_S", "IN_BYTES"))
    duration = _first_present(sample, ("flow_duration", "FLOW_DURATION", "FLOW_DURATION_MILLISECONDS"))
    ttl_var = _first_present(sample, ("ttl_variance", "TTL_VARIANCE"), default=1.0)

    final_label = str(predicted_label)
    fired_rules: list[dict[str, Any]] = []

    if packet_rate > 5000 and byte_rate > 1e6 and duration < 2:
        old_label = final_label
        final_label = "DoS/DDoS"
        fired_rules.append({
            "rule_id": "R1",
            "old_label": old_label,
            "new_label": final_label,
            "reason": (
                f"High-rate burst detected: packet_rate={packet_rate:.3f}, "
                f"byte_rate={byte_rate:.3f}, duration={duration:.3f}."
            ),
        })

    if duration > 60 and packet_rate > 1000 and final_label in {"Benign", "Scanning"}:
        old_label = final_label
        final_label = "DoS/DDoS"
        fired_rules.append({
            "rule_id": "R2",
            "old_label": old_label,
            "new_label": final_label,
            "reason": (
                f"Sustained high-rate flow detected: duration={duration:.3f}, "
                f"packet_rate={packet_rate:.3f}."
            ),
        })

    original = _as_probability_vector(predicted_probs)
    perturbed = _as_probability_vector(adversarial_probs)
    if original is not None and perturbed is not None and original.shape == perturbed.shape:
        perturbation_score = float(np.max(np.abs(perturbed - original)))
        if perturbation_score > adversarial_threshold:
            old_label = final_label
            final_label = f"{final_label}_ADV"
            fired_rules.append({
                "rule_id": "R3",
                "old_label": old_label,
                "new_label": final_label,
                "reason": (
                    f"Adversarial probability drift detected: max_delta={perturbation_score:.3f}, "
                    f"threshold={adversarial_threshold:.3f}."
                ),
            })

    anomaly = _safe_float(gnn_anomaly_score, default=-1.0) if gnn_anomaly_score is not None else None
    if anomaly is not None and anomaly > 0.8 and ttl_var > 5:
        old_label = final_label
        final_label = "ZeroDay"
        fired_rules.append({
            "rule_id": "R4",
            "old_label": old_label,
            "new_label": final_label,
            "reason": f"Novel threat signal detected: gnn_anomaly={anomaly:.3f}, ttl_variance={ttl_var:.3f}.",
        })

    if not fired_rules:
        fired_rules.append({
            "rule_id": "NONE",
            "old_label": str(predicted_label),
            "new_label": final_label,
            "reason": "No symbolic override triggered; neural prediction retained.",
        })

    return final_label, fired_rules
