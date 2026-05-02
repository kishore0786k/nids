import unittest

from app import app
import nids_engine as engine
from neuro_symbolic import apply_symbolic_rules


class NidsApiSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def get_json(self, endpoint):
        response = self.client.get(endpoint)
        self.assertEqual(response.status_code, 200, msg=response.get_data(as_text=True)[:300])
        return response.get_json()

    def post_json(self, endpoint, payload):
        response = self.client.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 200, msg=response.get_data(as_text=True)[:300])
        return response.get_json()

    def test_backend_status(self):
        data = self.get_json("/api/backend/status")
        self.assertEqual(data["backend"], "Flask + nids_engine.py")
        self.assertTrue(data["model_loaded"])
        self.assertGreater(data["test_rows"], 0)
        self.assertGreater(data["feature_count"], 0)

    def test_charts_contract(self):
        data = self.get_json("/api/charts?limit=200")
        for key in (
            "improvement_curve",
            "per_class",
            "confidence_histogram",
            "detection_counts",
            "class_error_rate",
            "roc_curve",
        ):
            self.assertIn(key, data)
        self.assertGreater(len(data["improvement_curve"]["labels"]), 2)

    def test_research_contract(self):
        data = self.get_json("/api/research?limit=200")
        self.assertIn("metrics", data)
        self.assertIn("window_metrics", data)
        self.assertIn("confusion_matrix", data)
        self.assertIn("rows", data)

    def test_defense_lifecycle(self):
        analysed = self.post_json("/api/defense/analyse", {"idx": 1})
        self.assertIn("incident", analysed)
        incident_id = analysed["incident"]["incident_id"]
        contained = self.post_json("/api/defense/contain", {"incident_id": incident_id})
        self.assertIn(contained["incident"]["status"], {"contained", "monitoring"})

    def test_invalid_inputs_are_sanitized(self):
        flow = self.get_json("/api/single-flow?idx=not-a-number")
        self.assertEqual(flow["index"], 0)
        research = self.get_json("/api/research?limit=bad")
        self.assertGreaterEqual(research["limit"], 50)
        charts = self.get_json("/api/charts?limit=-500")
        self.assertGreaterEqual(charts["limit"], 100)

    def test_ablation_contract(self):
        data = self.get_json("/api/ablation?limit=200")
        self.assertIn("systems", data)
        self.assertEqual(data["systems"][0]["name"], "Baseline MLP")
        self.assertEqual(data["systems"][1]["name"], "Neuro-symbolic")
        self.assertEqual(len(data["delta"]), len(data["labels"]))

    def test_novelty_contract(self):
        data = self.get_json("/api/novelty?limit=300&alpha=0.1")
        self.assertIn("uncertainty", data)
        self.assertIn("calibration", data)
        self.assertIn("conformal", data)
        self.assertIn("ood_drift", data)
        self.assertIn("review_queue", data)
        self.assertGreaterEqual(data["conformal"]["empirical_coverage"], 0.0)
        self.assertLessEqual(data["conformal"]["empirical_coverage"], 1.0)
        self.assertIn("ece", data["calibration"])

    def test_symbolic_adversarial_rule_uses_explicit_probabilities(self):
        label, rules = apply_symbolic_rules(
            sample={},
            predicted_label="Benign",
            predicted_probs=[0.95, 0.05],
            adversarial_probs=[0.50, 0.50],
        )
        self.assertEqual(label, "Benign_ADV")
        self.assertEqual(rules[-1]["rule_id"], "R3")
        self.assertEqual(rules[-1]["old_label"], "Benign")
        self.assertEqual(rules[-1]["new_label"], "Benign_ADV")

    def test_symbolic_zero_day_audit_trail_records_previous_label(self):
        label, rules = apply_symbolic_rules(
            sample={"ttl_variance": 10},
            predicted_label="Scanning",
            gnn_anomaly_score=0.95,
        )
        self.assertEqual(label, "ZeroDay")
        self.assertEqual(rules[-1]["rule_id"], "R4")
        self.assertEqual(rules[-1]["old_label"], "Scanning")
        self.assertEqual(rules[-1]["new_label"], "ZeroDay")

    def test_missing_model_file_has_clear_error(self):
        old_path = engine.MODEL_PATH
        try:
            engine.MODEL_PATH = old_path + ".missing"
            engine._reset_resources()
            with self.assertRaises(engine.ResourceLoadError) as ctx:
                engine.load_resources()
            self.assertIn("Missing base model", str(ctx.exception))
        finally:
            engine.MODEL_PATH = old_path
            engine._reset_resources()


if __name__ == "__main__":
    unittest.main()
