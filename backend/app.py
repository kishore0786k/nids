"""
Flask backend API for the Neuro-Symbolic NIDS research package.

The browser UI is served from the repository frontend directory, while all model/data/evaluation
logic is isolated in nids_engine.py.
"""

import base64
import binascii
import json
import logging
import os
import re
from datetime import datetime

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from backend import nids_engine as engine
from backend import run_manager
from backend.pipeline import LAST_RUN_PATH, PipelineStageError
from src.project_paths import FRONTEND_DIR, PROJECT_ROOT


app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static",
)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _raw_param(name, default=None, *aliases):
    payload = request.get_json(silent=True) or {}
    for key in (name, *aliases):
        if key in payload:
            return payload.get(key)
        if key in request.args:
            return request.args.get(key)
    return default


def _int_param(name, default, minimum=None, maximum=None, *aliases):
    raw = _raw_param(name, default, *aliases)
    try:
        value = int(float(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return value


def _float_param(name, default, minimum=None, maximum=None, *aliases):
    raw = _raw_param(name, default, *aliases)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return value


def _fusion_mode_param(default="hard"):
    mode = str(_raw_param("fusion_mode", default)).strip().lower()
    if mode not in {"hard", "soft"}:
        raise ValueError("fusion_mode must be 'hard' or 'soft'.")
    return mode


def _eval_params(default_window=750):
    return {
        "window_size": _int_param("window_size", default_window, 50, None, "limit", "n"),
        "flow_index": _int_param("flow_index", 0, 0, None, "flow_idx", "idx"),
        "alpha": _float_param("alpha", engine.DEFAULT_ALPHA, 0.0, 1.0),
        "beta": _float_param("beta", 1.0 - _float_param("alpha", engine.DEFAULT_ALPHA, 0.0, 1.0), 0.0, 1.0),
        "fusion_mode": _fusion_mode_param(engine.SYMBOLIC_FUSION_MODE),
        "seed": _int_param("seed", engine.DEFAULT_SEED, 0, 2_147_483_647),
    }


@app.errorhandler(engine.ResourceLoadError)
def handle_resource_error(exc):
    return jsonify({"error": "resource_load_error", "message": str(exc)}), 503


@app.errorhandler(ValueError)
def handle_value_error(exc):
    return jsonify({"error": "invalid_request", "message": str(exc)}), 400


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.name, "message": exc.description}), exc.code
    if isinstance(exc, PipelineStageError):
        app.logger.exception("Pipeline stage failed: %s", exc.stage)
        return jsonify({"error": "pipeline_stage_failed", "message": str(exc), "stage": exc.stage}), 500
    app.logger.exception("Unhandled backend error")
    return jsonify({"error": "internal_error", "message": str(exc)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "service": "neuro-symbolic-nids",
        "status": "healthy",
        "last_run_persisted": LAST_RUN_PATH.exists(),
        "last_run_path": str(LAST_RUN_PATH),
    })


@app.route("/single-flow")
@app.route("/comparison")
def legacy_spa_routes():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    return jsonify(engine.overview_data())


@app.route("/api/research")
def api_research():
    return jsonify(engine.analyse_window(**_eval_params(750)))


@app.route("/api/charts")
def api_charts():
    return jsonify(engine.chart_data(**_eval_params(2000)))


@app.route("/api/ablation")
def api_ablation():
    return jsonify(engine.ablation_data(**_eval_params(1000)))


@app.route("/api/novelty")
def api_novelty():
    params = _eval_params(2000)
    return jsonify(engine.novelty_data(
        params["window_size"],
        min(0.40, max(0.01, params["alpha"])),
        flow_index=params["flow_index"],
        seed=params["seed"],
    ))


@app.route("/api/run-all", methods=["GET", "POST"])
def api_run_all():
    params = _eval_params(750)
    payload = {
        "window_size": params["window_size"],
        "alpha": params["alpha"],
        "beta": params["beta"],
        "flow_index": params["flow_index"],
        "fusion_mode": params["fusion_mode"],
        "seed": params["seed"],
    }
    if str(_raw_param("sync", "false")).lower() in {"1", "true", "yes"}:
        return jsonify(engine.run_all(
            limit=params["window_size"],
            alpha=params["alpha"],
            beta=params["beta"],
            flow_idx=params["flow_index"],
            fusion_mode=params["fusion_mode"],
            seed=params["seed"],
        ))
    job = run_manager.start_run(payload)
    return jsonify(job), 202


@app.route("/run/status")
@app.route("/api/run/status")
def api_latest_run_status():
    status = run_manager.latest_status()
    if status is None:
        return jsonify({"error": "not_found", "message": "No Run All job has been started."}), 404
    return jsonify(status)


@app.route("/run/status/<job_id>")
@app.route("/api/run/status/<job_id>")
def api_run_status(job_id):
    status = run_manager.get_status(job_id)
    if status is None:
        return jsonify({"error": "not_found", "message": f"Unknown Run All job: {job_id}"}), 404
    return jsonify(status)


@app.route("/api/export-charts", methods=["POST"])
def api_export_charts():
    payload = request.get_json(silent=True) or {}
    charts = payload.get("charts") or []
    if not isinstance(charts, list) or not charts:
        raise ValueError("No charts were provided for export.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    export_dir = PROJECT_ROOT / "results" / "dashboard_chart_exports" / f"charts_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for position, chart in enumerate(charts, start=1):
        if not isinstance(chart, dict):
            continue
        raw_name = str(chart.get("name") or f"chart_{position}")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._") or f"chart_{position}"
        image = str(chart.get("image") or "")
        if "," in image:
            header, image = image.split(",", 1)
            if "image/png" not in header:
                raise ValueError(f"{raw_name} is not a PNG chart export.")

        try:
            image_bytes = base64.b64decode(image, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{raw_name} contains invalid chart image data.") from exc

        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{raw_name} did not decode to a PNG image.")

        filename = f"{position:02d}_{safe_name}.png"
        path = export_dir / filename
        path.write_bytes(image_bytes)
        saved.append({"name": raw_name, "file": str(path), "bytes": len(image_bytes)})

    if not saved:
        raise ValueError("No valid charts were provided for export.")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": payload.get("metadata") or {},
        "saved": saved,
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return jsonify({
        "ok": True,
        "export_dir": str(export_dir),
        "manifest": str(manifest_path),
        "saved": saved,
    })


@app.route("/api/backend/status")
def api_backend_status():
    return jsonify(engine.backend_status())


@app.route("/api/single-flow")
def api_single_flow():
    params = _eval_params(750)
    return jsonify(engine.predict_row(
        params["flow_index"],
        alpha=params["alpha"],
        beta=params["beta"],
        fusion_mode=params["fusion_mode"],
        seed=params["seed"],
    ))


@app.route("/api/comparison")
def api_comparison():
    params = _eval_params(200)
    data = engine.analyse_window(**params)
    return jsonify({
        "n_samples": data["limit"],
        "base_accuracy": data["window_metrics"]["baseline_mlp"][0],
        "ns_accuracy": data["window_metrics"]["neuro_symbolic"][0],
        "paper_existing_accuracy": data["paper_summary"]["existing"][0],
        "paper_proposed_accuracy": data["paper_summary"]["proposed"][0],
        "table": [
            {
                "idx": row["idx"],
                "true": row["true"],
                "baseline": row["baseline"],
                "neuro_symbolic": row["proposed"],
            }
            for row in data["rows"][:50]
        ],
    })


@app.route("/api/defense/analyse", methods=["POST"])
def api_defense_analyse():
    params = _eval_params(750)
    return jsonify(engine.analyse_defense(
        params["flow_index"],
        alpha=params["alpha"],
        beta=params["beta"],
        fusion_mode=params["fusion_mode"],
        seed=params["seed"],
    ))


@app.route("/api/defense/contain", methods=["POST"])
def api_defense_contain():
    payload = request.get_json(silent=True) or {}
    incident, message = engine.contain_incident(payload.get("incident_id"))
    if incident is None:
        return jsonify({"error": message}), 404
    return jsonify({"incident": incident, "message": message})


@app.route("/api/defense/status")
def api_defense_status():
    return jsonify(engine.defense_status())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("Neuro-Symbolic NIDS backend running")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dashboard: http://127.0.0.1:{port}")
    app.run(debug=False, threaded=True, port=port)
