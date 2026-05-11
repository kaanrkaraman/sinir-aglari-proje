"""Smoke tests for logging_setup."""

from __future__ import annotations

import logging

from annrag.logging_setup import configure_logging


def test_handler_installed_once():
    configure_logging("INFO")
    configure_logging("INFO")
    root = logging.getLogger()
    annrag_handlers = [h for h in root.handlers if h.get_name() == "annrag-stderr"]
    assert len(annrag_handlers) == 1


def test_level_refreshed_on_recall():
    configure_logging("INFO")
    configure_logging("DEBUG")
    root = logging.getLogger()
    handler = next(h for h in root.handlers if h.get_name() == "annrag-stderr")
    assert handler.level == logging.DEBUG
    assert root.level == logging.DEBUG
