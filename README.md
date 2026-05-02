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

The neuro-symbolic layer now reports `rules_fired`, `prediction_change_count`, accuracy/F1 deltas, binary attack-recall deltas, and concrete benign-to-attack correction examples. A smoke publication run (`--quick-limit 300`) demonstrates MLP accuracy/F1 improving from 0.8267/0.8224 to 0.8300/0.8257, with attack false negatives reduced from 6 to 4.

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

Use the dashboard **Run All** button to call `/api/run-all`, clear cached windows, recompute metrics/charts/reliability/defence state, and refresh the novelty proof panel from live model outputs.

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
- `src/neuro_symbolic.py`: auditable symbolic rules and rule analytics.
- `src/experiment_runner.py`: canonical offline experiment/evaluation runner.
- `src/project_paths.py`: shared project-root paths.
- `src/app.py` and `src/backend_engine.py`: compatibility wrappers for older imports.
- `frontend/`: dashboard UI.
- `paper/`: IEEE support files and generated paper assets.
- `results/publication_package/`: reproducibility package.
- `docs/REFACTOR_CHANGELOG.md`: static-to-empirical refactor notes.

## Verification

```bat
venv\Scripts\python.exe -m unittest backend.test_smoke
```

## Final Submission Notes

Before IEEE submission, manually verify the cited baseline method, bibliography metadata, author details, and manuscript claims against generated artifacts.
