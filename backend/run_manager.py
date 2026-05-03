"""In-memory Run All job manager with status polling."""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Mapping

from backend.pipeline import PipelineStageError, run_all_pipeline


LOGGER = logging.getLogger(__name__)
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="run-all")
_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _empty_stage(name: str) -> dict[str, Any]:
    return {"name": name, "status": "pending", "data": {}, "metrics": {}, "errors": []}


STAGE_NAMES = ["capture", "preprocess", "feature-extract", "predict", "log", "visualize"]


def _progress_percent(job: dict[str, Any]) -> int:
    completed = sum(1 for stage in job["stages"] if stage.get("status") == "success")
    running = any(stage.get("status") == "running" for stage in job["stages"])
    base = int((completed / max(1, len(job["stages"]))) * 100)
    if running and base < 96:
        return max(base, int(((completed + 0.35) / max(1, len(job["stages"]))) * 100))
    return base


def _update_job(job_id: str, patch: Mapping[str, Any]) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        job.update(patch)
        job["updated_at"] = _utc_now()
        job["progress"] = _progress_percent(job)


def _update_stage(job_id: str, stage_name: str, patch: Mapping[str, Any]) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        for stage in job["stages"]:
            if stage["name"] == stage_name:
                stage.update(patch)
                break
        job["current_stage"] = stage_name
        job["updated_at"] = _utc_now()
        job["progress"] = _progress_percent(job)


def _progress_callback(job_id: str):
    def callback(event: dict[str, Any]) -> None:
        name = event.get("stage")
        if event.get("event") == "pipeline_started":
            _update_job(job_id, {"state": "running", "current_stage": "capture"})
        elif event.get("event") == "stage_started" and name:
            _update_stage(job_id, str(name), {"status": "running"})
        elif event.get("event") == "stage_completed" and name:
            _update_stage(job_id, str(name), event.get("result") or {"status": "success"})
        elif event.get("event") == "stage_failed" and name:
            _update_stage(job_id, str(name), event.get("result") or {"status": "failed"})
        elif event.get("event") == "pipeline_completed":
            _update_job(job_id, {"state": "succeeded", "current_stage": "complete", "progress": 100})

    return callback


def _execute(job_id: str, params: Mapping[str, Any]) -> None:
    try:
        result = run_all_pipeline({**dict(params), "run_id": job_id}, progress=_progress_callback(job_id))
        _update_job(job_id, {"state": "succeeded", "progress": 100, "current_stage": "complete", "result": result})
    except PipelineStageError as exc:
        _update_job(
            job_id,
            {
                "state": "failed",
                "error": {"stage": exc.stage, "message": str(exc), "details": exc.details},
            },
        )
    except Exception as exc:
        LOGGER.exception("run-all job failed unexpectedly: %s", job_id)
        _update_job(job_id, {"state": "failed", "error": {"message": str(exc), "type": exc.__class__.__name__}})


def start_run(params: Mapping[str, Any]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "state": "queued",
        "progress": 0,
        "current_stage": "queued",
        "stages": [_empty_stage(name) for name in STAGE_NAMES],
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "params": dict(params),
        "result": None,
        "error": None,
    }
    with _LOCK:
        _JOBS[job_id] = job
    EXECUTOR.submit(_execute, job_id, dict(params))
    return get_status(job_id)


def get_status(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        snapshot = copy.deepcopy(job)
    if snapshot.get("state") == "succeeded":
        snapshot["progress"] = 100
    return snapshot


def latest_status() -> dict[str, Any] | None:
    with _LOCK:
        if not _JOBS:
            return None
        latest = max(_JOBS.values(), key=lambda job: job["created_at"])
        return copy.deepcopy(latest)
