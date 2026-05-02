"""Compatibility entry point for the canonical Flask backend.

Run from the repository root with:
    python -m backend.app
"""

from backend.app import app  # noqa: F401


if __name__ == "__main__":
    print("Delegating to canonical backend.app")
    app.run(debug=False, threaded=True, port=5000)
