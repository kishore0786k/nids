from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = PROJECT_ROOT / "paper"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

TRAIN_PATH = DATA_DIR / "train_processed.csv"
TEST_PATH = DATA_DIR / "test_processed.csv"
MODEL_PATH = MODEL_DIR / "ns_nids_model.pkl"
ROBUST_MODEL_PATH = MODEL_DIR / "robust_nsnids.pkl"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PUBLICATION_EXPERIMENT_PATH = RESULTS_DIR / "publication_experiment.json"


def resolve_from_root(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

