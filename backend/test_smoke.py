import unittest
from pathlib import Path

from backend.app import app
from backend import nids_engine as engine
from src.neuro_symbolic import apply_symbolic_rules
from src.project_paths import PROJECT_ROOT


class NidsApiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def get_json(self, endpoint):
        response = self.client.get(endpoint)
        self.assertEqual(response.status_code, 200, msg=response.get_data(as_text=True)[:500])
        return response.get_json()

    def get_error(self, endpoint, status=400):
        response = self.client.get(endpoint)
        self.assertEqual(response.status_code, status, msg=response.get_data(as_text=True)[:500])
        return response.get_json()

    def post_json(self, endpoint, payload):
        response = self.client.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 200, msg=response.get_data(as_text=True)[:500])
        return response.get_json()

    def test_backend_status_uses_canonical_engine(self):
        data = self.get_json("/api/backend/status")
        self.assertEqual(data["backend"], "Flask + nids_engine.py")
        self.assertTrue(data["model_loaded"])
        self.assertGreater(data["test_rows"], 0)
        self.assertIn("live_evaluation", data["evidence_separation"])
        self.assertTrue(Path(data["model_path"]).is_absolute())

    def test_charts_are_live_non_static_series(self):
        data = self.get_json("/api/charts?limit=750")
        curve = data["improvement_curve"]
        self.assertEqual(curve["source"], "live-window recomputation")
        self.assertGreaterEqual(len(curve["labels"]), 3)
        self.assertGreater(len(set(curve["existing_accuracy"])), 1)
        self.assertGreater(len(set(curve["proposed_f1"])), 1)
        self.assertNotIn("paper_proposed_f1", data["per_class"])
        self.assertEqual(data["per_class"]["source"], "live-window classification_report")
        self.assertTrue(data["computation_log"])

    def test_window_change_changes_chart_evidence(self):
        small = self.get_json("/api/charts?limit=200")
        large = self.get_json("/api/charts?limit=750")
        self.assertNotEqual(small["improvement_curve"]["labels"], large["improvement_curve"]["labels"])
        self.assertNotEqual(small["detection_counts"]["values"], large["detection_counts"]["values"])
        self.assertNotEqual(small["confidence_histogram"]["values"], large["confidence_histogram"]["values"])

    def test_detection_and_containment_counts_are_distinct_concepts(self):
        data = self.get_json("/api/charts?limit=750")
        counts = dict(zip(data["detection_counts"]["labels"], data["detection_counts"]["values"]))
        self.assertIn("True attack labels", counts)
        self.assertIn("Neuro-symbolic attack predictions", counts)
        self.assertIn("Containment candidates", counts)
        self.assertIn("High-confidence block recommendations", counts)
        self.assertLessEqual(counts["Containment candidates"], counts["Neuro-symbolic attack predictions"])
        self.assertLessEqual(counts["High-confidence block recommendations"], counts["Containment candidates"])
        self.assertGreater(len(set(counts.values())), 2)

    def test_symbolic_layer_changes_predictions_and_reports_rules(self):
        data = self.get_json("/api/research?limit=750")
        analytics = data["rule_analytics"]
        self.assertGreater(analytics["rule_trigger_count"], 0)
        self.assertGreater(analytics["changed_predictions"], 0)
        self.assertGreater(analytics["prediction_change_count"], 0)
        self.assertGreater(analytics["false_negative_attack_rescues"], 0)
        self.assertGreater(analytics["binary_attack_recall_delta"], 0)
        self.assertIn("R4_SUSPICIOUS_BENIGN_ATTACK_MASS", analytics["per_rule_trigger_count"])
        self.assertGreater(analytics["per_rule_trigger_frequency"]["R4_SUSPICIOUS_BENIGN_ATTACK_MASS"], 0)
        self.assertEqual(data["novelty_proof"]["verdict"], "proven")
        self.assertTrue(data["novelty_proof"]["examples"])
        self.assertEqual(data["metrics"]["source"], "live-window evaluation from model predictions and test labels")
        self.assertIn("saved-paper-summary", data["paper_summary"]["source"])

    def test_run_all_recomputes_full_dashboard_payload(self):
        result = self.post_json("/api/run-all", {"window_size": 750, "alpha": 0.65, "beta": 0.35, "flow_index": 0, "fusion_mode": "soft", "seed": 42})
        self.assertTrue(result["ok"])
        self.assertEqual(result["research"]["limit"], 750)
        self.assertEqual(result["charts"]["limit"], 750)
        self.assertEqual(result["research"]["parameters"]["alpha"], 0.65)
        self.assertEqual(result["research"]["parameters"]["fusion_mode"], "soft")
        self.assertIn("debug", result)
        self.assertIn("debug", result["charts"])
        self.assertGreater(result["debug"]["api_output_summary"]["rule_trigger_count"], 0)
        self.assertGreater(result["debug"]["api_output_summary"]["prediction_change_count"], 0)
        self.assertIn("charts", result["debug"]["datasets_changed"])
        self.assertIn("defense", result)
        self.assertEqual(result["research"]["novelty_proof"]["verdict"], "proven")

    def test_frontend_exposes_run_all_and_impact_panel(self):
        html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        js = (PROJECT_ROOT / "frontend" / "js" / "dashboard.js").read_text(encoding="utf-8")
        self.assertIn('id="btn-run-all"', html)
        self.assertIn("Neuro-Symbolic Impact Proof", html)
        self.assertIn("/api/run-all", js)

    def test_streamlit_dashboard_has_no_fixed_performance_literals(self):
        text = (PROJECT_ROOT / "src" / "app_streamlit.py").read_text(encoding="utf-8")
        forbidden = ("98.1%", "94.2%", "0.900", "0.942", "0.981")
        for literal in forbidden:
            self.assertNotIn(literal, text)
        self.assertIn("get_backend_json", text)

    def test_invalid_inputs_are_rejected(self):
        self.assertEqual(self.get_error("/api/single-flow?idx=not-a-number")["error"], "invalid_request")
        self.assertEqual(self.get_error("/api/research?limit=bad")["error"], "invalid_request")
        self.assertEqual(self.get_error("/api/charts?window_size=-500")["error"], "invalid_request")
        self.assertEqual(self.get_error("/api/novelty?limit=300&alpha=not-a-number")["error"], "invalid_request")
        self.assertEqual(self.get_error("/api/charts?fusion_mode=maybe")["error"], "invalid_request")

    def test_live_parameters_change_metrics_and_charts(self):
        base = self.get_json("/api/charts?window_size=750&flow_index=0&alpha=0.65&beta=0.35&fusion_mode=soft&seed=42")
        alpha_changed = self.get_json("/api/charts?window_size=750&flow_index=0&alpha=0.90&beta=0.10&fusion_mode=soft&seed=42")
        seed_changed = self.get_json("/api/charts?window_size=750&flow_index=0&alpha=0.65&beta=0.35&fusion_mode=soft&seed=99")
        flow_changed = self.get_json("/api/charts?window_size=750&flow_index=10&alpha=0.65&beta=0.35&fusion_mode=soft&seed=42")

        self.assertNotEqual(base["metric_comparison"]["proposed"], alpha_changed["metric_comparison"]["proposed"])
        self.assertNotEqual(base["class_distribution"]["proposed_values"], seed_changed["class_distribution"]["proposed_values"])
        self.assertNotEqual(base["detection_counts"]["values"], flow_changed["detection_counts"]["values"])
        self.assertIn("difference_chart", base)
        self.assertIn("attack_recall_gain", base)
        self.assertNotEqual(base["roc_curve"]["baseline"]["points"], base["roc_curve"]["proposed"]["points"])

    def test_proposed_predictions_are_distinct_and_improve_default_target(self):
        data = self.get_json("/api/research?window_size=750&flow_index=0&alpha=0.65&beta=0.35&fusion_mode=soft&seed=42")
        analytics = data["rule_analytics"]
        baseline = data["window_metrics"]["baseline_mlp"]
        proposed = data["window_metrics"]["neuro_symbolic"]
        self.assertNotEqual(baseline, proposed)
        self.assertGreater(analytics["prediction_change_count"], 0)
        self.assertTrue(
            proposed[0] > baseline[0]
            or proposed[3] > baseline[3]
            or analytics["binary_attack_recall_delta"] > 0
        )
        self.assertTrue(any(row["changed_prediction"] for row in data["rows"]))
        self.assertIn("final_label", data["rows"][0])

    def test_defense_lifecycle(self):
        analysed = self.post_json("/api/defense/analyse", {"idx": 1})
        self.assertIn("incident", analysed)
        incident_id = analysed["incident"]["incident_id"]
        contained = self.post_json("/api/defense/contain", {"incident_id": incident_id})
        self.assertIn(contained["incident"]["status"], {"contained", "monitoring"})

    def test_ablation_and_novelty_are_empirical(self):
        ablation = self.get_json("/api/ablation?limit=750")
        self.assertEqual(ablation["systems"][0]["name"], "Baseline MLP")
        self.assertEqual(ablation["systems"][1]["name"], "Neuro-symbolic")
        self.assertEqual(len(ablation["delta"]), len(ablation["labels"]))
        self.assertNotEqual(ablation["systems"][0]["metrics"], ablation["systems"][1]["metrics"])

        novelty = self.get_json("/api/novelty?limit=750&alpha=0.1")
        self.assertGreaterEqual(novelty["conformal"]["empirical_coverage"], 0.0)
        self.assertLessEqual(novelty["conformal"]["empirical_coverage"], 1.0)
        self.assertGreater(len(novelty["chart_ready"]["calibration_bins"]), 0)

    def test_symbolic_adversarial_rule_uses_explicit_probabilities(self):
        label, rules, strength = apply_symbolic_rules(
            sample={},
            predicted_label="Benign",
            predicted_probs=[0.95, 0.05],
            adversarial_probs=[0.50, 0.50],
        )
        self.assertEqual(label, "Benign_ADV")
        self.assertEqual(rules[-1]["rule_id"], "R5_ADVERSARIAL_PROBABILITY_DRIFT")
        self.assertEqual(rules[-1]["old_label"], "Benign")
        self.assertEqual(rules[-1]["new_label"], "Benign_ADV")
        self.assertGreater(strength, 0.0)

    def test_symbolic_zero_day_audit_trail_records_previous_label(self):
        label, rules, strength = apply_symbolic_rules(
            sample={"ttl_variance": 10},
            predicted_label="Scanning",
            gnn_anomaly_score=0.95,
        )
        self.assertEqual(label, "ZeroDay")
        self.assertEqual(rules[-1]["rule_id"], "R6_ZERO_DAY_ANOMALY")
        self.assertEqual(rules[-1]["old_label"], "Scanning")
        self.assertEqual(rules[-1]["new_label"], "ZeroDay")
        self.assertGreater(strength, 0.0)

    def test_missing_model_file_has_clear_error(self):
        old_path = engine.MODEL_PATH
        try:
            engine.MODEL_PATH = Path(str(old_path) + ".missing")
            engine._reset_resources()
            with self.assertRaises(engine.ResourceLoadError) as ctx:
                engine.load_resources()
            self.assertIn("Missing base model", str(ctx.exception))
        finally:
            engine.MODEL_PATH = old_path
            engine._reset_resources()


if __name__ == "__main__":
    unittest.main()
