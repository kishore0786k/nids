from __future__ import annotations

import time

import pytest

from backend import run_manager
from backend.pipeline import LAST_RUN_PATH, run_all_pipeline


@pytest.fixture()
def pipeline_result(tiny_resources, run_params):
    return run_all_pipeline(run_params)


@pytest.mark.parametrize("stage_name", ["capture", "preprocess", "feature-extract", "predict", "log", "visualize"])
def test_pipeline_stage_contracts(pipeline_result, stage_name):
    stage = next(item for item in pipeline_result["stages"] if item["name"] == stage_name)
    assert stage["status"] == "success"
    assert isinstance(stage["data"], dict)
    assert isinstance(stage["metrics"], dict)
    assert isinstance(stage["errors"], list)


def test_run_all_job_end_to_end(tiny_resources, run_params):
    job = run_manager.start_run(run_params)
    for _ in range(80):
        status = run_manager.get_status(job["job_id"])
        if status["state"] in {"succeeded", "failed"}:
            break
        time.sleep(0.1)
    assert status["state"] == "succeeded"
    assert status["result"]["ok"] is True
    assert status["result"]["research"]["limit"] == 6
    assert LAST_RUN_PATH.exists()


def test_single_prediction_evidence(tiny_resources, run_params):
    from backend import nids_engine as engine

    flow = engine.predict_row(**{
        "index": run_params["flow_index"],
        "alpha": run_params["alpha"],
        "beta": run_params["beta"],
        "fusion_mode": run_params["fusion_mode"],
        "seed": run_params["seed"],
    })
    evidence = flow["evidence"]
    assert evidence["top_features"]
    assert evidence["flow_context"]["src_port"] is not None
    assert evidence["calibrated_probability"] is not None
    assert evidence["historical_frequency"]["total_rows"] == 6
