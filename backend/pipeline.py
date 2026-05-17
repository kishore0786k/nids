"""Typed Run All orchestration for the dashboard pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from backend.config import settings
from backend import nids_engine as engine
from src.project_paths import TEST_PATH, TRAIN_PATH, MODEL_PATH


LOGGER = logging.getLogger(__name__)
RUNS_DIR = settings.runs_dir
LAST_RUN_PATH = RUNS_DIR / "last_run.json"
ProgressCallback = Callable[[dict[str, Any]], None]


class PipelineError(RuntimeError):
    """Base exception for run-all orchestration failures."""


class PipelineStageError(PipelineError):
    """Raised when one named stage fails."""

    def __init__(self, stage: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = dict(details or {})


@dataclass(frozen=True)
class PipelineParams:
    """Sanitized user inputs for one full pipeline run."""

    window_size: int
    flow_index: int
    alpha: float
    beta: float
    fusion_mode: str
    seed: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PipelineParams":
        config = engine.evaluation_config(
            window_size=payload.get("window_size", payload.get("limit", 750)),
            flow_index=payload.get("flow_index", payload.get("flow_idx", payload.get("idx", 0))),
            alpha=payload.get("alpha", engine.DEFAULT_ALPHA),
            beta=payload.get("beta"),
            fusion_mode=payload.get("fusion_mode", engine.SYMBOLIC_FUSION_MODE),
            seed=payload.get("seed", engine.DEFAULT_SEED),
        )
        return cls(
            window_size=config.window_size,
            flow_index=config.flow_index,
            alpha=config.alpha,
            beta=config.beta,
            fusion_mode=config.fusion_mode,
            seed=config.seed,
        )

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageResult:
    """Public result contract for every pipeline stage."""

    name: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _stage_result(name: str, data: dict[str, Any] | None = None, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": "success", "data": data or {}, "metrics": metrics or {}, "errors": []}


def _run_stage(
    index: int,
    total: int,
    name: str,
    context: dict[str, Any],
    func: Callable[[dict[str, Any]], dict[str, Any]],
    progress: ProgressCallback | None,
) -> StageResult:
    started = utc_now()
    started_perf = perf_counter()
    if progress:
        progress({"event": "stage_started", "stage": name, "stage_index": index, "total_stages": total})
    LOGGER.info("run_all stage started: %s", name)
    try:
        payload = func(context)
        elapsed_ms = (perf_counter() - started_perf) * 1000.0
        result = StageResult(
            name=name,
            status=str(payload.get("status", "success")),
            data=dict(payload.get("data") or {}),
            metrics={**dict(payload.get("metrics") or {}), "elapsed_ms": round(elapsed_ms, 3)},
            errors=list(payload.get("errors") or []),
            started_at=started,
            finished_at=utc_now(),
        )
        if result.status != "success":
            raise PipelineStageError(name, f"{name} returned status {result.status}", details=result.to_dict())
        if progress:
            progress(
                {
                    "event": "stage_completed",
                    "stage": name,
                    "stage_index": index,
                    "total_stages": total,
                    "result": result.to_dict(),
                }
            )
        LOGGER.info("run_all stage completed: %s in %.2f ms", name, elapsed_ms)
        return result
    except PipelineStageError:
        raise
    except Exception as exc:
        elapsed_ms = (perf_counter() - started_perf) * 1000.0
        error = {"type": exc.__class__.__name__, "message": str(exc)}
        result = StageResult(
            name=name,
            status="failed",
            errors=[error],
            metrics={"elapsed_ms": round(elapsed_ms, 3)},
            started_at=started,
            finished_at=utc_now(),
        )
        if progress:
            progress(
                {
                    "event": "stage_failed",
                    "stage": name,
                    "stage_index": index,
                    "total_stages": total,
                    "result": result.to_dict(),
                }
            )
        LOGGER.exception("run_all stage failed: %s", name)
        raise PipelineStageError(name, str(exc), details=result.to_dict()) from exc


def _capture(context: dict[str, Any]) -> dict[str, Any]:
    engine.load_resources()
    params = PipelineParams.from_mapping(context["raw_params"])
    context["params"] = params
    if hasattr(engine, "_clear_caches"):
        engine._clear_caches()
    row_count = int(len(engine._X_test))
    return _stage_result(
        "capture",
        {
            "source": "processed NetFlow test window",
            "test_path": str(TEST_PATH),
            "train_path": str(TRAIN_PATH),
            "model_path": str(MODEL_PATH),
            "rows_available": row_count,
            "parameters": params.public(),
        },
        {"rows_available": row_count},
    )


def _preprocess(context: dict[str, Any]) -> dict[str, Any]:
    params: PipelineParams = context["params"]
    config = engine.evaluation_config(**params.public())
    indices = engine._window_indices(config)
    subset = engine._X_test.iloc[indices].reset_index(drop=True)
    context["config"] = config
    context["indices"] = indices
    context["subset"] = subset
    return _stage_result(
        "preprocess",
        {
            "window_size": int(config.window_size),
            "flow_index": int(config.flow_index),
            "selected_indices_preview": [int(value) for value in indices[:10]],
        },
        {"rows_selected": int(len(subset))},
    )


def _feature_extract(context: dict[str, Any]) -> dict[str, Any]:
    subset = context["subset"]
    numeric_columns = [str(col) for col in subset.select_dtypes(include=[np.number]).columns]
    context["feature_columns"] = [str(col) for col in subset.columns]
    return _stage_result(
        "feature-extract",
        {
            "feature_columns": context["feature_columns"],
            "numeric_feature_columns": numeric_columns,
        },
        {"feature_count": int(len(subset.columns)), "numeric_feature_count": int(len(numeric_columns))},
    )


def _predict(context: dict[str, Any]) -> dict[str, Any]:
    params: PipelineParams = context["params"]
    novelty_alpha = min(0.40, max(0.01, params.alpha))
    overview = engine.overview_data()
    research = engine.analyse_window(
        window_size=params.window_size,
        flow_index=params.flow_index,
        alpha=params.alpha,
        beta=params.beta,
        fusion_mode=params.fusion_mode,
        seed=params.seed,
    )
    novelty = engine.novelty_data(params.window_size, novelty_alpha, flow_index=params.flow_index, seed=params.seed)
    defense = engine.analyse_defense(
        params.flow_index,
        alpha=params.alpha,
        beta=params.beta,
        fusion_mode=params.fusion_mode,
        seed=params.seed,
    )
    context.update({"overview": overview, "research": research, "novelty": novelty, "defense": defense})
    analytics = research.get("rule_analytics", {})
    return _stage_result(
        "predict",
        {"overview": overview, "research": research, "novelty": novelty, "defense": defense},
        {
            "window_size": research.get("limit"),
            "rule_trigger_count": analytics.get("rule_trigger_count"),
            "prediction_change_count": analytics.get("prediction_change_count"),
        },
    )


def _log(context: dict[str, Any]) -> dict[str, Any]:
    params: PipelineParams = context["params"]
    research = context["research"]
    summary = {
        "parameters": params.public(),
        "window_size": research.get("limit"),
        "rule_analytics": research.get("rule_analytics", {}),
        "defense_label": context["defense"]["flow"].get("ns_label"),
        "defense_risk": context["defense"]["flow"].get("risk"),
        "logged_at": utc_now(),
    }
    context["run_log_summary"] = summary
    LOGGER.info("run_all audit summary: %s", summary)
    return _stage_result("log", {"summary": summary}, {"audit_fields": len(summary)})


def _visualize(context: dict[str, Any]) -> dict[str, Any]:
    params: PipelineParams = context["params"]
    charts = engine.chart_data(
        window_size=params.window_size,
        flow_index=params.flow_index,
        alpha=params.alpha,
        beta=params.beta,
        fusion_mode=params.fusion_mode,
        seed=params.seed,
    )
    backend = engine.backend_status()
    publication = {}
    try:
        if int(backend.get("test_rows", 0)) < 1000:
            raise RuntimeError("publication package generation skipped for tiny test fixture")
        from backend.generate_publication_package import build_package

        package = build_package(
            limit=params.window_size,
            alpha=params.alpha,
            beta=params.beta,
            fusion_mode=params.fusion_mode,
            seed=params.seed,
            flow_index=params.flow_index,
        )
        publication = {
            "package_json": package.get("figures") and "results/publication_package/publication_package.json",
            "figure_count": len(package.get("figures", [])),
        }
    except Exception as exc:
        LOGGER.warning("Publication package generation skipped: %s", exc)
        publication = {"warning": str(exc)}
    context.update({"charts": charts, "backend": backend, "publication": publication})
    chart_keys = [key for key, value in charts.items() if isinstance(value, dict)]
    return _stage_result(
        "visualize",
        {"charts": charts, "backend": backend, "publication": publication},
        {"chart_payloads": len(chart_keys), "publication_figures": publication.get("figure_count", 0)},
    )


def _persist_last_run(result: dict[str, Any]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")


def run_all_pipeline(raw_params: Mapping[str, Any], progress: ProgressCallback | None = None) -> dict[str, Any]:
    """Execute capture -> preprocess -> feature-extract -> predict -> log -> visualize."""
    started = perf_counter()
    run_id = str(raw_params.get("run_id") or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}")
    context: dict[str, Any] = {"raw_params": dict(raw_params), "run_id": run_id}
    stages: list[StageResult] = []
    stage_defs: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = [
        ("capture", _capture),
        ("preprocess", _preprocess),
        ("feature-extract", _feature_extract),
        ("predict", _predict),
        ("log", _log),
        ("visualize", _visualize),
    ]

    if progress:
        progress({"event": "pipeline_started", "run_id": run_id, "total_stages": len(stage_defs)})
    LOGGER.info("run_all pipeline started: %s", run_id)
    try:
        for index, (name, func) in enumerate(stage_defs, start=1):
            stages.append(_run_stage(index, len(stage_defs), name, context, func, progress))
    except PipelineStageError as exc:
        failed_result = {
            "ok": False,
            "run_id": run_id,
            "message": str(exc),
            "error": {"stage": exc.stage, "message": str(exc), "details": exc.details},
            "stages": [stage.to_dict() for stage in stages],
            "finished_at": utc_now(),
        }
        _persist_last_run(failed_result)
        raise

    elapsed_ms = (perf_counter() - started) * 1000.0
    params: PipelineParams = context["params"]
    debug = {
        "input_parameters": {**dict(raw_params), **params.public()},
        "api_output_summary": {
            "overview_samples": context["overview"]["total_samples"],
            "research_window": context["research"]["limit"],
            "chart_window": context["charts"]["limit"],
            "rule_trigger_count": context["research"]["rule_analytics"]["rule_trigger_count"],
            "prediction_change_count": context["research"]["rule_analytics"]["prediction_change_count"],
            "delta_accuracy": context["research"]["rule_analytics"]["delta_accuracy"],
            "delta_f1": context["research"]["rule_analytics"]["delta_f1"],
            "novelty_verdict": context["research"]["novelty_proof"]["verdict"],
            "elapsed_ms": round(elapsed_ms, 3),
        },
        "datasets_changed": ["overview", "research_metrics", "charts", "defense_analysis", "novelty_panel", "backend_status"],
    }
    result = {
        "ok": True,
        "run_id": run_id,
        "message": "Full pipeline completed.",
        "parameters": params.public(),
        "overview": context["overview"],
        "research": context["research"],
        "charts": context["charts"],
        "novelty": context["novelty"],
        "defense": context["defense"],
        "backend": context["backend"],
        "stages": [stage.to_dict() for stage in stages],
        "debug": debug,
        "finished_at": utc_now(),
    }
    _persist_last_run(result)
    if progress:
        progress({"event": "pipeline_completed", "run_id": run_id, "result": result})
    LOGGER.info("run_all pipeline completed: %s", run_id)
    return result
