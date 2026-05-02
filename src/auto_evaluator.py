"""Generate paper-ready metrics from the canonical live backend pipeline."""

from __future__ import annotations

import pandas as pd

from backend import nids_engine as engine
from src.project_paths import RESULTS_DIR


def run_comprehensive_evaluation(limit: int = 2000) -> dict:
    """Generate empirical dashboard/publication metrics without placeholders."""
    output_dir = RESULTS_DIR / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    research = engine.analyse_window(limit)
    charts = engine.chart_data(limit)
    metrics = {
        "limit": research["limit"],
        "baseline_accuracy": research["window_metrics"]["baseline_mlp"][0],
        "neuro_symbolic_accuracy": research["window_metrics"]["neuro_symbolic"][0],
        "baseline_macro_f1": research["window_metrics"]["baseline_mlp"][3],
        "neuro_symbolic_macro_f1": research["window_metrics"]["neuro_symbolic"][3],
        "rule_triggers": research["rule_analytics"]["rule_trigger_count"],
        "changed_predictions": research["rule_analytics"]["changed_predictions"],
        "false_negative_attack_rescues": research["rule_analytics"]["false_negative_attack_rescues"],
    }

    pd.DataFrame([metrics]).to_csv(output_dir / "final_metrics.csv", index=False)
    pd.DataFrame(research["rows"]).to_csv(output_dir / "unified_results.csv", index=False)
    pd.DataFrame(charts["computation_log"], columns=["chart_computation_step"]).to_csv(
        output_dir / "chart_computation_log.csv",
        index=False,
    )

    print("Paper-ready evaluation complete.")
    print(metrics)
    return metrics


if __name__ == "__main__":
    run_comprehensive_evaluation()
