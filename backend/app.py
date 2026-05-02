"""
Flask backend API for the Neuro-Symbolic NIDS research package.

The browser UI is served from ../frontend, while all model/data/evaluation
logic is isolated in nids_engine.py.
"""

from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

import nids_engine as engine


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static",
)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


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
        "paper_existing_accuracy": data["metrics"]["existing"][0],
        "paper_proposed_accuracy": data["metrics"]["proposed"][0],
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
    print("Neuro-Symbolic NIDS backend running")
    print("Dashboard: http://127.0.0.1:5000")
    app.run(debug=False, threaded=True, port=5000)
