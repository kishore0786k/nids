"""Logging setup for console and rotating file output."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from backend.config import Settings


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    if getattr(root, "_nids_logging_configured", False):
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_dir / settings.log_file,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    setattr(root, "_nids_logging_configured", True)
