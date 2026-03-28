from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from btc_alert_engine.normalize.bybit_public import parse_orderbook_payload, parse_trade_payload
from btc_alert_engine.normalize.orderbook import BybitOrderBookBuilder
from btc_alert_engine.schemas import TopOfBookState
from btc_alert_engine.storage.raw_ndjson import iter_raw_events


def replay_top_of_book(paths: Iterable[str | Path], *, symbol: str) -> list[TopOfBookState]:
    builder = BybitOrderBookBuilder(symbol=symbol)
    states: list[TopOfBookState] = []
    for event in iter_raw_events(paths):
        if not event.topic.startswith("orderbook."):
            continue
        message = parse_orderbook_payload(event.payload)
        builder.apply(message)
        states.append(builder.top_of_book(ts=message.ts))
    return states


def replay_trades(paths: Iterable[str | Path], *, symbol: str) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for event in iter_raw_events(paths):
        if not event.topic.startswith("publicTrade."):
            continue
        for trade in parse_trade_payload(event.payload):
            if trade.symbol == symbol:
                trades.append(trade.model_dump(mode="json"))
    return trades


def deterministic_replay_hash(paths: Iterable[str | Path], *, symbol: str) -> str:
    top = [state.model_dump(mode="json") for state in replay_top_of_book(paths, symbol=symbol)]
    trades = replay_trades(paths, symbol=symbol)
    payload = {"symbol": symbol, "top_of_book": top, "trades": trades}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
