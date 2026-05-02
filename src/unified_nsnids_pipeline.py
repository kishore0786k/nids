"""Compatibility facade over the canonical backend evaluation pipeline."""

from __future__ import annotations

import pandas as pd

from backend import nids_engine as engine


class UnifiedNeuroSymbolicNIDS:
    """Small adapter retained for older notebooks/scripts."""

    def predict_full_pipeline(self, flows: pd.DataFrame) -> pd.DataFrame:
        limit = min(len(flows), 5000)
        data = engine.analyse_window(limit)
        return pd.DataFrame(data["rows"]).rename(
            columns={
                "baseline": "baseline_label",
                "proposed": "final_ns_label",
            }
        )


if __name__ == "__main__":
    result = UnifiedNeuroSymbolicNIDS().predict_full_pipeline(pd.DataFrame(index=range(100)))
    print(result.head().to_string(index=False))
