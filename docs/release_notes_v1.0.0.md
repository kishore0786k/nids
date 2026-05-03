# Release Notes Draft: v1.0.0

## Highlights

- Full Run All orchestration across capture, preprocess, feature extraction, prediction, audit logging, and visualization.
- Evidence cards suitable for analyst and publication review.
- Interactive Plotly graph explorer with persisted chart configuration.
- Docker Compose startup for API and frontend.
- CI for lint, type check, tests, audit, and Docker build.

## Verification

- `pytest`
- `ruff check backend src tests`
- `mypy --strict --ignore-missing-imports --follow-imports=skip backend/config.py backend/logging_config.py backend/run_manager.py`
- `pip-audit -r requirements.txt`
- `docker compose build`

## Upgrade Notes

- Copy `.env.example` to `.env` for environment overrides.
- Keep model and processed data files mounted at `models/` and `data/` when using Docker Compose.
