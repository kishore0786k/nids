# Neuro-Symbolic NIDS

[![CI](https://github.com/kishore0786k/NIDS/actions/workflows/ci.yml/badge.svg)](https://github.com/kishore0786k/NIDS/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v1.0.0-blue)](https://github.com/kishore0786k/NIDS/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

Publication-ready Network Intrusion Detection System for NF-ToN-IoT-V2 NetFlow traffic. We propose a cross-dataset, uncertainty-aware neuro-symbolic NIDS that detects known attacks and rejects low-confidence unseen traffic.

![Demo GIF placeholder](docs/demo-placeholder.gif)

## Quickstart

```bash
cp .env.example .env
docker compose up --build
```

Open the dashboard at `http://127.0.0.1:8080`. The API is available at `http://127.0.0.1:5000`.

Local Python run:

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m backend.app
```

## Architecture

```mermaid
flowchart LR
  A["NetFlow window"] --> B["Preprocess and feature extraction"]
  B --> C["Sklearn model predict_proba"]
  C --> D["Symbolic rule fusion"]
  D --> E["Evidence builder"]
  D --> F["Run summary and charts"]
  E --> G["Flask API"]
  F --> G
  G --> H["Vanilla JS dashboard"]
  H --> I["Plotly chart explorer"]
  H --> J["Expandable evidence cards"]
```

## Run All Pipeline

The `Run All` button starts an asynchronous job:

1. capture processed NetFlow rows
2. preprocess the selected deterministic window
3. extract and cache feature payloads
4. batch predict and apply symbolic rules
5. write structured audit logs
6. build visualization payloads

Status is polled at `/api/run/status/<job_id>`. The latest completed summary is written to `runs/last_run.json`.

## API Reference

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Service health and last-run persistence state |
| `/api/run-all` | POST | Start full pipeline job; returns `job_id` |
| `/api/run/status/<job_id>` | GET | Poll progress, stage results, and final payload |
| `/api/single-flow` | GET | Predict one flow with evidence |
| `/api/charts` | GET | Chart and Plotly explorer data |
| `/api/defense/analyse` | POST | Create a defensive recommendation for one flow |
| `/api/export-charts` | POST | Persist rendered PNG chart exports |
| `/api/upload/validate` | POST | Validate upload size and file type |

## Evidence

Each prediction emits:

- top contributing features with SHAP or permutation-style attribution scores
- flow context such as IP, port, protocol, byte, packet, and timestamp fields when present
- raw confidence and calibrated/fused probability
- matched symbolic rule signatures
- historical frequency for the predicted attack class

The dashboard renders this as expandable evidence cards instead of a single paragraph.

## Model Card

**Dataset:** NF-ToN-IoT-V2 processed NetFlow CSV splits in `data/train_processed.csv` and `data/test_processed.csv`.

**Model:** sklearn-compatible classifier stored at `models/ns_nids_model.pkl`; optional robust model at `models/robust_nsnids.pkl`.

**Metrics:** Live dashboard metrics are recomputed from current model predictions. Publication summaries under `results/` are used only as saved reference artifacts.

**Limitations:** Results depend on the supplied processed split and may not generalize to unseen networks without external validation. Symbolic rules are audit aids, not a substitute for analyst review.

**Ethical Use:** Use for defensive monitoring, research reproducibility, and education. Do not deploy for unauthorized surveillance or automated punitive action without human oversight.

## Screenshots

- `results/dashboard-runall.png`
- `results/dashboard-impact-panel.png`
- `results/dashboard-architecture-final.png`

## Development

```bash
pytest
ruff check backend src tests
mypy --strict --ignore-missing-imports --follow-imports=skip backend/config.py backend/logging_config.py backend/run_manager.py
pip-audit -r requirements.txt
```

## Cross-dataset generalization (NF-UNSW-NB15)

Cross-dataset evaluation is provided by `evaluate_cross_dataset.py`. The script loads the trained model from `models/ns_nids_model.pkl`, discovers the NF-UNSW-NB15 NetFlow download from the UQ NIDS dataset page when `--data_path` is absent, aligns columns to the NF-ToN-IoT-V2 feature schema, and saves per-class F1, macro-F1, and confusion-matrix output to `results/cross_dataset_results.json`.

```bash
venv\Scripts\python evaluate_cross_dataset.py --model_path models\ns_nids_model.pkl --data_path data\NF-UNSW-NB15-v3.csv
```

Latest local smoke run: `20,000` NF-UNSW-NB15-v3 rows, macro-F1 `0.0860`. The low transfer score is intentional evidence for the generalization gap and should be reported honestly unless the training protocol is extended with domain adaptation or feature recalibration.

## Unknown-traffic handling

### Unknown-attack rejection (confidence thresholding)

The backend now computes softmax confidence and entropy before symbolic rule fusion. If `max_prob < tau`, the final label is `UNKNOWN`, the symbolic rule layer is skipped for that flow, and the batch rejection rate is logged. The default threshold is configured in `config.yaml`:

```yaml
unknown_confidence_threshold: 0.65
```

The same threshold is used by `ablation_study.py` and `calibration_analysis.py` unless `--tau` is provided.

## Ablation study results

Run:

```bash
venv\Scripts\python ablation_study.py --model_path models\ns_nids_model.pkl --data_path data\test_processed.csv
```

| Config | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| A) DNN only | 0.9232 | 0.9222 | 0.9215 |
| B) DNN + rules | 0.9234 | 0.9224 | 0.9217 |
| C) DNN + confidence | 0.8383 | 0.7615 | 0.7957 |
| D) full system | 0.8383 | 0.7615 | 0.7957 |

The CSV table is saved to `results/ablation_table.csv`.

## Calibration analysis

Run:

```bash
venv\Scripts\python calibration_analysis.py --model_path models\ns_nids_model.pkl --data_path data\test_processed.csv
```

The analysis computes 10-bin Expected Calibration Error (ECE) and writes a reliability diagram to `results/calibration_curve.png`. Latest local run: DNN-only ECE `0.0035`; proposed system ECE `0.0541`. The proposed ECE is higher because low-confidence rejection changes correctness without recalibrating the probability surface; this is useful paper evidence and a clear target for temperature scaling or conformal calibration.

## Citation

```bibtex
@software{kishore_nids_2026,
  author = {Kishore},
  title = {Neuro-Symbolic Network Intrusion Detection System},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/kishore0786k/NIDS}
}
```

## License

MIT License. See `LICENSE`.
