"""Structured logging configuration.

A single function installs a stderr handler with a consistent format so every
module gets the same shape. Idempotent — safe to call multiple times in tests
or from CLI entry points.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_HANDLER_NAME = "annrag-stderr"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(level: LogLevel = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)

    # Idempotency: if our handler is already attached, just refresh its level.
    for existing in root.handlers:
        if existing.get_name() == _HANDLER_NAME:
            existing.setLevel(level)
            return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
    handler.setLevel(level)
    root.addHandler(handler)
