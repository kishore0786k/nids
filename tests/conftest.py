from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from backend import nids_engine as engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "tiny_flows.csv"


@pytest.fixture()
def tiny_resources(tmp_path, monkeypatch):
    df = pd.read_csv(FIXTURE_CSV)
    model = RandomForestClassifier(n_estimators=8, random_state=7)
    model.fit(df.drop(columns=["label"]), df["label"].astype(str))
    model_path = tmp_path / "tiny_model.pkl"
    joblib.dump(model, model_path)
    test_path = tmp_path / "test_processed.csv"
    train_path = tmp_path / "train_processed.csv"
    metrics_path = tmp_path / "metrics.json"
    df.to_csv(test_path, index=False)
    df.to_csv(train_path, index=False)
    metrics_path.write_text(json.dumps({}), encoding="utf-8")

    monkeypatch.setattr(engine, "MODEL_PATH", model_path)
    monkeypatch.setattr(engine, "TEST_PATH", test_path)
    monkeypatch.setattr(engine, "TRAIN_PATH", train_path)
    monkeypatch.setattr(engine, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(engine, "ROBUST_PATH", tmp_path / "missing_robust.pkl")
    engine._reset_resources()
    yield {
        "model_path": model_path,
        "test_path": test_path,
        "train_path": train_path,
        "metrics_path": metrics_path,
    }
    engine._reset_resources()


@pytest.fixture()
def run_params():
    return {
        "window_size": 50,
        "flow_index": 0,
        "alpha": 0.65,
        "beta": 0.35,
        "fusion_mode": "soft",
        "seed": 7,
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def live_server(tiny_resources):
    from backend.app import app
    from werkzeug.serving import make_server

    port = free_port()
    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=5)
