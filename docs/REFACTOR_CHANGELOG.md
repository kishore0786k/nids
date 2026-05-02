# Refactor Changelog

## Static Before

- `src/app_streamlit.py` displayed fixed performance values such as `98.1%`, `94.2%`, and fixed comparison arrays.
- `backend/nids_engine.py` built the improvement curve by interpolating saved summary metrics instead of recomputing metrics per window.
- Detection coverage duplicated the neuro-symbolic attack prediction count as containment count.
- `src/backend_engine.py` duplicated backend logic from `backend/nids_engine.py`, and `src/app.py` duplicated Flask routes.
- `src/ieee_plots.py` generated formula-shaped curves from placeholder values.
- Several scripts depended on the current working directory or `sys.path` edits.

## Empirical Now

- Flask/API logic is canonical in `backend/app.py` and `backend/nids_engine.py`.
- Offline experiments remain canonical in `src/experiment_runner.py`.
- `src/app.py` and `src/backend_engine.py` are compatibility wrappers only.
- Dashboard metrics, Streamlit metrics, publication tables, and publication figures use backend recomputation from model predictions and test labels.
- Saved paper-summary values are separated under `paper_summary` and labeled as `saved-paper-summary evidence`.
- Improvement curves are recomputed over multiple sample windows.
- Detection counts now distinguish true attack labels, baseline attack predictions, neuro-symbolic attack predictions, containment candidates, and high-confidence block recommendations.
- Symbolic rules expose trigger counts, applied counts, changed predictions, false-negative rescues, exact corrections, introduced false positives, and per-class attack recall deltas.
- Shared paths live in `src/project_paths.py`; scripts use project-root-derived paths.

## Graphs Changed

- Accuracy improvement curve: formula/interpolation removed; each point is a live window evaluation.
- Per-class F1: saved paper F1 removed from the live chart; baseline and neuro-symbolic F1 come from current classification reports.
- Detection coverage: changed from duplicated doughnut counts to distinct bar-series concepts.
- IEEE figures: generated through `backend.generate_publication_package` from the corrected chart pipeline.

## Tests Added

- Non-static chart series and live evidence-source assertions.
- Window-change assertions for chart and histogram data.
- Detection/containment semantic count checks.
- Streamlit hardcoded-metric literal checks.
- Neuro-symbolic prediction-change and rule-counter checks.
- False-negative rescue checks.
- Invalid input sanitization checks.
