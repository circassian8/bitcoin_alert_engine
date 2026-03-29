from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


class PartitionedNdjsonWriter:
    """Writes arbitrary records into a date/hour partitioned NDJSON tree.

    Layout:
      <base_dir>/<namespace>/<symbol>/date=YYYY-MM-DD/hour=HH.ndjson

    A bounded handle cache is used so long backfills do not exhaust the OS
    file-descriptor limit while still avoiding an open/close cycle on every row.
    """

    def __init__(self, base_dir: str | Path, *, max_open_handles: int = 64) -> None:
        self.base_dir = Path(base_dir)
        self.max_open_handles = max_open_handles
        self._handles: OrderedDict[Path, TextIO] = OrderedDict()

    def _path_for(self, *, namespace: str, symbol: str, ts_ms: int) -> Path:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
        namespace_path = Path(*[part for part in namespace.split("/") if part])
        return (
            self.base_dir
            / namespace_path
            / symbol
            / f"date={dt:%Y-%m-%d}"
            / f"hour={dt:%H}.ndjson"
        )

    def _evict_if_needed(self) -> None:
        while self.max_open_handles > 0 and len(self._handles) > self.max_open_handles:
            _, stale_handle = self._handles.popitem(last=False)
            stale_handle.close()

    def _get_handle(self, path: Path) -> TextIO:
        handle = self._handles.get(path)
        if handle is not None and not handle.closed:
            self._handles.move_to_end(path)
            return handle

        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
        self._handles[path] = handle
        self._handles.move_to_end(path)
        self._evict_if_needed()
        return handle

    def write(self, *, namespace: str, symbol: str, ts_ms: int, record: BaseModel | dict[str, Any]) -> Path:
        path = self._path_for(namespace=namespace, symbol=symbol, ts_ms=ts_ms)
        handle = self._get_handle(path)
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        handle.write(json.dumps(payload, default=_json_default) + "\n")
        handle.flush()
        return path

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "PartitionedNdjsonWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def iter_json_records(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    for path_like in paths:
        path = Path(path_like)
        if path.is_dir():
            for file_path in sorted(path.rglob("*.ndjson")):
                yield from iter_json_records([file_path])
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
