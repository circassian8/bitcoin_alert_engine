from __future__ import annotations

import logging
import os


def configure_logging(level: str | None = None) -> None:
    chosen = (level or os.getenv("BTC_ALERT_ENGINE_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, chosen, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
