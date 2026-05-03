# Changelog

## v1.0.0 - 2026-05-03

### Added

- Asynchronous Run All pipeline with per-stage status, polling, and persisted `runs/last_run.json`.
- Enriched evidence cards with feature attribution, flow context, calibrated probability, matched rule signatures, and class history.
- Plotly chart explorer with chart type, axis, range, class, top-N, PNG/SVG/CSV export, and localStorage persistence.
- Pydantic settings, rotating logs, upload validation, rate limiting, Docker, CI, pytest coverage, and dependency audit workflow.

### Fixed

- Run All no longer blocks as an opaque single request and now exposes backend errors to the dashboard.
- Evidence is no longer shallow/static.
- Graph selection is no longer hard-coded to one view.
