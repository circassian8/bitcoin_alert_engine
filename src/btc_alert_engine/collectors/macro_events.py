from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

from btc_alert_engine.schemas import RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter


class MacroCsvIngestor:
    """Ingests a canonical CSV into the raw-event store.

    CSV columns:
      ts_utc_ms,event_type,source,importance,notes
    """

    def __init__(self, writer: RawEventWriter, logger: logging.Logger | None = None) -> None:
        self.writer = writer
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def ingest(self, csv_path: str | Path) -> list[Path]:
        paths: list[Path] = []
        with Path(csv_path).open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts = int(row["ts_utc_ms"])
                event = RawEvent(
                    source="macro_csv",
                    topic=f"macro.{row['event_type']}",
                    symbol="MACRO",
                    exchange_ts=ts,
                    local_received_ts=ts,
                    payload=row,
                )
                paths.append(self.writer.write(event))
        self.logger.info("ingested %s macro events", len(paths))
        return paths


def sample_rows() -> Iterable[dict[str, str]]:
    yield {
        "ts_utc_ms": "1742947200000",
        "event_type": "fomc",
        "source": "fed",
        "importance": "high",
        "notes": "Example row",
    }
