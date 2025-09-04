from __future__ import annotations

import logging


_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a single stream handler.

    - Level: INFO
    - Handler: StreamHandler to stderr with readable format
    - Idempotent: does not add duplicate handlers on repeated calls
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate logs via root

    # Use a sentinel to prevent adding multiple handlers for this logger
    if getattr(logger, "_rt_stream_attached", False):
        return logger

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(handler)

    # Mark as initialized
    setattr(logger, "_rt_stream_attached", True)
    return logger
