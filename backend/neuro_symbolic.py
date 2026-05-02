from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.neuro_symbolic import (  # noqa: E402,F401
    build_symbolic_context,
    load_default_symbolic_context,
    apply_symbolic_rules,
)
