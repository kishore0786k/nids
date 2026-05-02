# Neuro-Symbolic NIDS Research Package

Publication-oriented intrusion detection project for NF-ToN-IoT-V2 NetFlow traffic.

## Research Contribution

This project contributes a confidence-aware neuro-symbolic NIDS: an MLP detects NetFlow attack classes, while percentile-calibrated symbolic rules target weak false-negative regions such as benign-looking scanning flows. The rule layer supports hard override and soft probability fusion, records auditable rule traces, and reports rule firing rate, prediction-change rate, false-negative reduction, per-class recall/F1 deltas, and ablation results.

Novelty claim: instead of adding passive hand-written rules, the system calibrates symbolic thresholds from the training distribution and fuses rule evidence with neural probabilities so symbolic logic measurably changes predictions and improves targeted attack detection.

## Publication Experiment

```bat
venv\Scripts\python.exe -m src.experiment_runner --fusion-mode soft --alpha 0.65
```

Fast smoke run:

```bat
venv\Scripts\python.exe -m src.experiment_runner --quick-limit 500
```

The runner uses `data/train_processed.csv` and `data/test_processed.csv`, evaluates RandomForest, MLP, NeuroSymbolic hard fusion, NeuroSymbolic soft fusion, prints classification reports/confusion matrices, and saves `results/publication_experiment.json`.

Publication artifacts are saved under `results/`:

- `publication_experiment.json`
- `model_comparison.csv`
- `attack_class_deltas.csv`
- `rule_diagnostics.csv`
- `confusion_mlp.csv`
- `confusion_neurosymbolic.csv`
- `mcnemar_mlp_vs_neurosymbolic.csv`

Optional multi-seed run:

```bat
venv\Scripts\python.exe -m src.experiment_runner --seeds 42,43,44
```

## Run Dashboard

```bat
run_project.bat
```

Then open:

```text
http://127.0.0.1:5000
```

## Build IEEE/Overleaf Artifacts

```bat
build_publication_package.bat
```

Generated outputs:

- `paper/generated/*.tex`
- `paper/figures/generated/*.pdf`
- `results/publication_package/publication_package.json`
- `NIDS_IEEE_Overleaf_Package.zip`

## Structure

- `backend/app.py`: Flask API and dashboard server.
- `backend/nids_engine.py`: model, metrics, charts, reliability, OOD, and defence backend.
- `backend/neuro_symbolic.py`: auditable symbolic rules.
- `frontend/`: dashboard UI.
- `paper/`: IEEE support files and generated paper assets.
- `results/publication_package/`: reproducibility package.

## Verification

```bat
cd backend
..\venv\Scripts\python.exe -m unittest test_smoke.py
```

## Final Submission Notes

Before IEEE submission, manually verify the cited baseline method, bibliography metadata, author details, and manuscript claims against generated artifacts.
