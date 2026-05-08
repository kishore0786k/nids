from __future__ import annotations

import numpy as np
import pandas as pd

from src.journal_nids_upgrade import (
    UNKNOWN_LABEL,
    apply_unknown_threshold,
    binary_from_attack_types,
    clean_external_frame,
    clean_training_frame,
    get_severity,
)


def test_clean_training_frame_keeps_numeric_features_and_labels() -> None:
    frame = pd.DataFrame(
        {
            "L4_SRC_PORT": [1, 2, 3],
            "IN_BYTES": [100, 200, 300],
            "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
            "Label": [0, 1, 1],
            "Attack": ["Benign", "dos", "xss"],
        }
    )

    X, y_multi, y_bin, feature_cols = clean_training_frame(frame)

    assert feature_cols == ["L4_SRC_PORT", "IN_BYTES"]
    assert X.shape == (3, 2)
    assert y_multi.tolist() == ["Benign", "dos", "xss"]
    assert y_bin.tolist() == [0, 1, 1]


def test_external_alignment_fills_missing_train_columns() -> None:
    frame = pd.DataFrame(
        {
            "L4_SRC_PORT": [53, 443],
            "Label": [0, 1],
            "Attack": ["Benign", "Exploits"],
        }
    )

    X, y, attack_type = clean_external_frame(frame, ["L4_SRC_PORT", "IN_BYTES"])

    assert X["IN_BYTES"].tolist() == [0, 0]
    assert y.tolist() == [0, 1]
    assert attack_type.tolist() == ["Benign", "Exploits"]


def test_unknown_threshold_and_severity_bins() -> None:
    predicted = np.array(["Benign", "dos", "xss"], dtype=object)
    confidence = np.array([0.9, 0.6, 0.7])
    final = apply_unknown_threshold(predicted, confidence, threshold=0.65)

    assert final.tolist() == ["Benign", UNKNOWN_LABEL, "xss"]

    severity, score = get_severity(np.array([0.2, 0.7, 0.95]), np.array([0.1, 0.8, 1.0]), alpha=0.6)
    assert severity.tolist() == ["Low", "High", "Critical"]
    assert np.all(score >= 0)


def test_binary_from_attack_types_maps_unknown_attacks_to_one() -> None:
    assert binary_from_attack_types(["Benign", "normal", "xss", UNKNOWN_LABEL]).tolist() == [0, 0, 1, 1]
