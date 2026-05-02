# Research Contribution and Novelty

## Core Contribution

This project presents a confidence-aware neuro-symbolic NIDS for NF-ToN-IoT-V2 NetFlow traffic. The neural component learns multi-class attack patterns, while the symbolic component uses training-distribution percentiles and model probabilities to identify weak false-negative regions and selectively override or softly fuse predictions.

## Novelty

1. Data-calibrated symbolic rules: thresholds are learned from processed training features rather than fixed constants.
2. Confidence-aware fusion: rules override low-confidence predictions and use soft probability fusion when evidence is useful but not absolute.
3. Targeted false-negative reduction: rules focus on attack flows likely to be predicted benign, especially scanning-like traffic.
4. Auditable evaluation: the experiment reports rule trigger rate, prediction-change rate, before/after accuracy and F1, attack-class recall/F1 deltas, false negatives, hard/soft ablations, and confusion matrices.

## Defensible Claim

The system should be presented as a targeted detection-quality improvement over an MLP baseline, not as a universal replacement for RandomForest. The publishable claim is:

> Confidence-aware symbolic fusion can measurably reduce selected attack false negatives and improve per-class detection recall while preserving overall MLP-level accuracy.

## Publication Evidence Generated

Run `venv\Scripts\python.exe -m src.experiment_runner --fusion-mode soft --alpha 0.65`. The script exports model comparison tables, attack-class deltas, confusion matrices, rule diagnostics, hard/soft ablations, false-negative counts, and an approximate paired McNemar test to `results/`.
