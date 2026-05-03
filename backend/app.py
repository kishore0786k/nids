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
from src.project_paths import FRONTEND_DIR, PROJECT_ROOT


app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static",
)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


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
    return jsonify({"error": "internal_error", "message": str(exc)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/single-flow")
@app.route("/comparison")
def legacy_spa_routes():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    return jsonify(engine.overview_data())


@app.route("/api/research")
def api_research():
    return jsonify(engine.analyse_window(request.args.get("limit", 750, type=int)))


@app.route("/api/charts")
def api_charts():
    return jsonify(engine.chart_data(request.args.get("limit", 2000, type=int)))


@app.route("/api/ablation")
def api_ablation():
    return jsonify(engine.ablation_data(request.args.get("limit", 1000, type=int)))


@app.route("/api/novelty")
def api_novelty():
    return jsonify(engine.novelty_data(
        request.args.get("limit", 2000, type=int),
        request.args.get("alpha", 0.10, type=float),
    ))


@app.route("/api/run-all", methods=["GET", "POST"])
def api_run_all():
    payload = request.get_json(silent=True) or {}
    limit = payload.get("limit", request.args.get("limit", 750, type=int))
    alpha = payload.get("alpha", request.args.get("alpha", 0.10, type=float))
    flow_idx = payload.get("flow_idx", request.args.get("flow_idx", 0, type=int))
    return jsonify(engine.run_all(limit=limit, alpha=alpha, flow_idx=flow_idx))


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
    return jsonify(engine.predict_row(request.args.get("idx", 0, type=int)))


@app.route("/api/comparison")
def api_comparison():
    data = engine.analyse_window(request.args.get("n", 200, type=int))
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
    payload = request.get_json(silent=True) or {}
    return jsonify(engine.analyse_defense(payload.get("idx", request.args.get("idx", 0))))


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
