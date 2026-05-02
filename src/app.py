"""
Flask API/server for the Neuro-Symbolic IoT NIDS dashboard.

The backend computation lives in backend_engine.py. This file only exposes
page routes and JSON endpoints consumed by the browser dashboard.
"""

from flask import Flask, jsonify, render_template, request

import backend_engine as engine


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


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
    limit = request.args.get("limit", 750, type=int)
    return jsonify(engine.analyse_window(limit))


@app.route("/api/charts")
def api_charts():
    limit = request.args.get("limit", 2000, type=int)
    return jsonify(engine.chart_data(limit))


@app.route("/api/backend/status")
def api_backend_status():
    return jsonify(engine.backend_status())


@app.route("/api/single-flow")
def api_single_flow():
    idx = request.args.get("idx", 0, type=int)
    return jsonify(engine.predict_row(idx))


@app.route("/api/comparison")
def api_comparison():
    n = request.args.get("n", 200, type=int)
    data = engine.analyse_window(n)
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
    idx = payload.get("idx", request.args.get("idx", 0))
    return jsonify(engine.analyse_defense(idx))


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
    print("Neuro-Symbolic IoT NIDS - Flask backend + dashboard")
    print("Backend API and frontend dashboard: http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
