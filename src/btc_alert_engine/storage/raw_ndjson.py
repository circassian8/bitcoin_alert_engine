from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

from btc_alert_engine.schemas import RawEvent


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


class RawEventWriter:
    """Writes raw events into a date/hour partitioned NDJSON tree."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self._handles: dict[Path, TextIO] = {}

    def _path_for(self, event: RawEvent) -> Path:
        dt = datetime.fromtimestamp(event.local_received_ts / 1000, tz=UTC)
        topic_dir = event.topic.replace("/", "_")
        return (
            self.base_dir
            / event.source
            / topic_dir
            / f"date={dt:%Y-%m-%d}"
            / f"hour={dt:%H}.ndjson"
        )

    def write(self, event: RawEvent) -> Path:
        path = self._path_for(event)
        if path not in self._handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles[path] = path.open("a", encoding="utf-8")
        handle = self._handles[path]
        handle.write(json.dumps(event.model_dump(mode="json"), default=_json_default) + "\n")
        handle.flush()
        return path

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "RawEventWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _discover_ndjson_files(paths: Iterable[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    for path_like in paths:
        path = Path(path_like)
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*.ndjson")))
        else:
            discovered.append(path)
    return sorted(set(discovered))


def iter_raw_events(paths: Iterable[str | Path]) -> Iterator[RawEvent]:
    for path in _discover_ndjson_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield RawEvent.model_validate_json(line)


def _partition_key(path: Path) -> tuple[str, str, str]:
    date_key = "date=9999-99-99"
    hour_key = "hour=99.ndjson"
    for part in path.parts:
        if part.startswith("date="):
            date_key = part
        elif part.startswith("hour="):
            hour_key = part
    return (date_key, hour_key, str(path))


def _event_sort_key(event: RawEvent) -> tuple[int, int, str, str]:
    primary_ts = event.exchange_ts if event.exchange_ts is not None else event.local_received_ts
    return (primary_ts, event.local_received_ts, event.source, event.topic)


def iter_raw_events_sorted(paths: Iterable[str | Path]) -> Iterator[RawEvent]:
    """Yield raw events ordered by exchange/local timestamp.

    To stay memory-bounded, files are grouped by the existing date/hour partition. All files
    for a single hour are loaded, merged, sorted, and then yielded before moving on.
    """

    files = _discover_ndjson_files(paths)
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for file_path in files:
        date_key, hour_key, _ = _partition_key(file_path)
        groups[(date_key, hour_key)].append(file_path)

    for key in sorted(groups):
        hour_events: list[RawEvent] = []
        for file_path in sorted(groups[key], key=_partition_key):
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    hour_events.append(RawEvent.model_validate_json(line))
        hour_events.sort(key=_event_sort_key)
        for event in hour_events:
            yield event
