from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    resolved_level = getattr(logging, level.upper(), logging.INFO)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)

    root_logger.setLevel(resolved_level)
