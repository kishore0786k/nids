"""Compatibility wrapper for the canonical backend engine.

The single source of truth for API/evaluation logic is `backend.nids_engine`.
This module remains only for older imports that referenced `src.backend_engine`.
"""

from backend.nids_engine import *  # noqa: F401,F403

