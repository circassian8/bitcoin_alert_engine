from __future__ import annotations

import csv
import gzip
import io
import json
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from btc_alert_engine.schemas import RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter


class BybitHistoryImportError(ValueError):
    pass


_TRADE_TS_ALIASES = ["T", "ts", "time", "timestamp", "trade_time", "execTime"]
_TRADE_SYMBOL_ALIASES = ["s", "symbol"]
_TRADE_PRICE_ALIASES = ["p", "price"]
_TRADE_SIZE_ALIASES = ["v", "size", "qty", "quantity"]
_TRADE_SIDE_ALIASES = ["S", "side"]
_TRADE_ID_ALIASES = ["i", "trade_id", "tradeId", "execId", "id"]
_TRADE_SEQ_ALIASES = ["seq", "cross_seq", "crossSequence"]
_TRADE_BLOCK_ALIASES = ["BT", "block_trade", "isBlockTrade"]
_TRADE_RPI_ALIASES = ["RPI", "rpi_trade", "isRPITrade"]

_ORDERBOOK_TS_ALIASES = ["ts", "T", "timestamp", "time"]
_ORDERBOOK_SYMBOL_ALIASES = ["s", "symbol"]
_ORDERBOOK_UPDATE_ALIASES = ["u", "update_id", "updateId"]
_ORDERBOOK_SEQ_ALIASES = ["seq", "cross_seq", "crossSequence"]
_ORDERBOOK_CTS_ALIASES = ["cts", "client_ts", "clientTimestamp"]
_ORDERBOOK_BIDS_ALIASES = ["b", "bids", "bid"]
_ORDERBOOK_ASKS_ALIASES = ["a", "asks", "ask"]
_ORDERBOOK_TYPE_ALIASES = ["type", "message_type", "msg_type"]


def _discover_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for path_like in paths:
        path = Path(path_like)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            files.append(path)
    return sorted(set(files))


def _open_text_handle(path: Path) -> io.TextIOBase:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_zip_members(path: Path) -> Iterator[tuple[str, io.TextIOBase]]:
    with zipfile.ZipFile(path, "r") as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/"):
                continue
            with archive.open(name, "r") as handle:
                yield name, io.TextIOWrapper(handle, encoding="utf-8")


def _guess_text_format(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".jsonl") or lowered.endswith(".ndjson") or lowered.endswith(".data"):
        return "jsonl"
    if lowered.endswith(".csv") or lowered.endswith(".txt"):
        return "csv"
    return "jsonl"


def _iter_records_from_text(name: str, handle: io.TextIOBase) -> Iterator[dict[str, Any]]:
    fmt = _guess_text_format(name)
    if fmt == "csv":
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            yield dict(row)
        return
    for line in handle:
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload
        else:
            raise BybitHistoryImportError(f"Expected JSON object rows in {name}")


def _iter_records(paths: Iterable[str | Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in _discover_files(paths):
        if path.suffix.lower() == ".zip":
            for member_name, handle in _iter_zip_members(path):
                with handle:
                    for record in _iter_records_from_text(member_name, handle):
                        yield path, record
            continue
        with _open_text_handle(path) as handle:
            for record in _iter_records_from_text(path.name, handle):
                yield path, record


def _first_value(record: dict[str, Any], aliases: Iterable[str]) -> Any | None:
    for key in aliases:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", 0, "0", "false", "False", "FALSE", "N", "n"):
        return False
    if value in (1, "1", "true", "True", "TRUE", "Y", "y"):
        return True
    return bool(value)


def _filename_symbol(path: Path) -> str | None:
    stem = path.stem
    if stem.endswith(".data"):
        stem = Path(stem).stem
    for token in stem.replace("-", "_").split("_"):
        token = token.strip().upper()
        if token and token.endswith(("USDT", "USD", "USDC")):
            return token
    return None


def _normalize_trade_record(record: dict[str, Any], *, symbol_hint: str | None = None) -> dict[str, Any]:
    if "data" in record and isinstance(record["data"], list):
        raise BybitHistoryImportError("Trade importer expects row records, not websocket envelopes")
    ts = _first_value(record, _TRADE_TS_ALIASES)
    price = _first_value(record, _TRADE_PRICE_ALIASES)
    size = _first_value(record, _TRADE_SIZE_ALIASES)
    side = _first_value(record, _TRADE_SIDE_ALIASES)
    if ts is None or price is None or size is None or side is None:
        raise BybitHistoryImportError(f"Trade row missing one of required fields ts/price/size/side: {record}")
    symbol = _first_value(record, _TRADE_SYMBOL_ALIASES) or symbol_hint
    if symbol is None:
        raise BybitHistoryImportError(f"Trade row missing symbol and no symbol hint available: {record}")
    trade_id = _first_value(record, _TRADE_ID_ALIASES)
    if trade_id is None:
        trade_id = f"{symbol}-{ts}-{price}-{size}-{side}"
    normalized = {
        "T": int(str(ts)),
        "s": str(symbol).upper(),
        "p": str(price),
        "v": str(size),
        "S": str(side).title(),
        "i": str(trade_id),
    }
    seq = _first_value(record, _TRADE_SEQ_ALIASES)
    if seq is not None:
        normalized["seq"] = int(str(seq))
    normalized["BT"] = _to_bool(_first_value(record, _TRADE_BLOCK_ALIASES))
    normalized["RPI"] = _to_bool(_first_value(record, _TRADE_RPI_ALIASES))
    return normalized


def _parse_book_levels(raw_value: Any) -> list[list[str]]:
    if raw_value in (None, ""):
        return []
    value = raw_value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise BybitHistoryImportError(f"Could not parse book levels: {raw_value}") from exc
    if not isinstance(value, list):
        raise BybitHistoryImportError(f"Book levels must be a list, got {type(value)!r}")
    levels: list[list[str]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            levels.append([str(item[0]), str(item[1])])
        elif isinstance(item, dict):
            px = item.get("price") or item.get("p")
            qty = item.get("size") or item.get("qty") or item.get("q")
            if px is None or qty is None:
                raise BybitHistoryImportError(f"Unsupported book level object: {item}")
            levels.append([str(px), str(qty)])
        else:
            raise BybitHistoryImportError(f"Unsupported book level row: {item}")
    return levels


def _normalize_orderbook_record(record: dict[str, Any], *, symbol_hint: str | None = None, depth: int | None = None) -> dict[str, Any]:
    if {"topic", "type", "ts", "data"}.issubset(record):
        payload = dict(record)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise BybitHistoryImportError(f"Order book payload data must be an object: {record}")
        payload.setdefault("cts", _first_value(data, _ORDERBOOK_CTS_ALIASES) or payload.get("ts"))
        if "s" not in data and symbol_hint is not None:
            data["s"] = symbol_hint
        payload["data"] = data
        return payload

    ts = _first_value(record, _ORDERBOOK_TS_ALIASES)
    bids = _first_value(record, _ORDERBOOK_BIDS_ALIASES)
    asks = _first_value(record, _ORDERBOOK_ASKS_ALIASES)
    if ts is None or bids is None or asks is None:
        raise BybitHistoryImportError(f"Order book row missing ts/bids/asks: {record}")
    symbol = _first_value(record, _ORDERBOOK_SYMBOL_ALIASES) or symbol_hint
    if symbol is None:
        raise BybitHistoryImportError(f"Order book row missing symbol and no symbol hint available: {record}")
    update_id = _first_value(record, _ORDERBOOK_UPDATE_ALIASES)
    if update_id is None:
        update_id = ts
    seq = _first_value(record, _ORDERBOOK_SEQ_ALIASES)
    cts = _first_value(record, _ORDERBOOK_CTS_ALIASES) or ts
    message_type = str(_first_value(record, _ORDERBOOK_TYPE_ALIASES) or "delta")
    if message_type not in {"snapshot", "delta"}:
        message_type = "delta"
    if depth is None:
        topic = str(record.get("topic") or "")
        if topic.startswith("orderbook."):
            try:
                depth = int(topic.split(".")[1])
            except Exception:
                depth = 500
        else:
            depth = 500
    return {
        "topic": f"orderbook.{depth}.{str(symbol).upper()}",
        "type": message_type,
        "ts": int(str(ts)),
        "cts": int(str(cts)),
        "data": {
            "s": str(symbol).upper(),
            "u": int(str(update_id)),
            "seq": int(str(seq)) if seq is not None else None,
            "b": _parse_book_levels(bids),
            "a": _parse_book_levels(asks),
        },
    }


def import_bybit_history_trades(
    paths: Iterable[str | Path],
    *,
    writer: RawEventWriter,
    symbol: str | None = None,
    batch_size: int = 500,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_rows = 0
    total_events = 0
    pending: list[dict[str, Any]] = []
    pending_symbol: str | None = symbol.upper() if symbol else None

    def _flush() -> None:
        nonlocal pending, total_events, pending_symbol
        if not pending:
            return
        batch_symbol = pending_symbol or str(pending[0]["s"]).upper()
        ts = max(int(item["T"]) for item in pending)
        event = RawEvent(
            source="bybit_ws",
            topic=f"publicTrade.{batch_symbol}",
            symbol=batch_symbol,
            exchange_ts=ts,
            local_received_ts=ts,
            payload={"topic": f"publicTrade.{batch_symbol}", "ts": ts, "data": pending},
            metadata={"imported": True, "importer": "bybit_history_trades"},
        )
        writer.write(event)
        total_events += 1
        pending = []

    for path, record in _iter_records(paths):
        normalized = _normalize_trade_record(record, symbol_hint=(symbol.upper() if symbol else _filename_symbol(path)))
        row_symbol = str(normalized["s"]).upper()
        if pending and row_symbol != (pending_symbol or row_symbol):
            _flush()
        pending_symbol = row_symbol
        pending.append(normalized)
        total_rows += 1
        if len(pending) >= batch_size:
            _flush()
    _flush()
    return {"trade_rows": total_rows, "trade_events": total_events}


def import_bybit_history_orderbook(
    paths: Iterable[str | Path],
    *,
    writer: RawEventWriter,
    symbol: str | None = None,
    depth: int | None = None,
) -> dict[str, int]:
    total_rows = 0
    total_events = 0
    for path, record in _iter_records(paths):
        normalized = _normalize_orderbook_record(record, symbol_hint=(symbol.upper() if symbol else _filename_symbol(path)), depth=depth)
        data = normalized["data"]
        ts = int(normalized["ts"])
        event = RawEvent(
            source="bybit_ws",
            topic=str(normalized["topic"]),
            symbol=str(data["s"]).upper(),
            exchange_ts=ts,
            local_received_ts=int(normalized.get("cts") or ts),
            payload=normalized,
            metadata={"imported": True, "importer": "bybit_history_orderbook", "source_file": str(path)},
        )
        writer.write(event)
        total_rows += 1
        total_events += 1
    return {"orderbook_rows": total_rows, "orderbook_events": total_events}
