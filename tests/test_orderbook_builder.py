import json
from pathlib import Path

import pytest

from btc_alert_engine.normalize.bybit_public import parse_orderbook_payload
from btc_alert_engine.normalize.orderbook import (
    BybitOrderBookBuilder,
    OrderBookGapError,
    OrderBookOutOfOrderError,
    OrderBookStateError,
)


def load_json(name: str) -> dict:
    path = Path(__file__).parent / "data" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_orderbook_builder_snapshot_then_delta() -> None:
    snapshot = parse_orderbook_payload(load_json("bybit_orderbook_snapshot.json"))
    delta = parse_orderbook_payload(load_json("bybit_orderbook_delta.json"))

    builder = BybitOrderBookBuilder(symbol="BTCUSDT")
    builder.apply(snapshot)
    top = builder.top_of_book(ts=snapshot.ts)
    assert str(top.best_bid_price) == "16493.50"
    assert str(top.best_ask_price) == "16611.00"

    builder.apply(delta)
    top = builder.top_of_book(ts=delta.ts)
    assert str(top.best_bid_price) == "16494.00"
    assert str(top.best_ask_price) == "16610.80"
    assert str(top.spread) == "116.80"


def test_orderbook_builder_rejects_delta_before_snapshot() -> None:
    delta = parse_orderbook_payload(load_json("bybit_orderbook_delta.json"))
    builder = BybitOrderBookBuilder(symbol="BTCUSDT")
    with pytest.raises(OrderBookStateError):
        builder.apply(delta)


def test_orderbook_builder_detects_gap() -> None:
    snapshot = parse_orderbook_payload(load_json("bybit_orderbook_snapshot.json"))
    gap_payload = load_json("bybit_orderbook_delta.json")
    gap_payload["data"]["u"] = snapshot.update_id + 2
    gap_payload["data"]["seq"] = snapshot.seq + 2
    gap_delta = parse_orderbook_payload(gap_payload)

    builder = BybitOrderBookBuilder(symbol="BTCUSDT")
    builder.apply(snapshot)
    with pytest.raises(OrderBookGapError):
        builder.apply(gap_delta)


def test_orderbook_builder_detects_out_of_order_update() -> None:
    snapshot = parse_orderbook_payload(load_json("bybit_orderbook_snapshot.json"))
    stale_payload = load_json("bybit_orderbook_delta.json")
    stale_payload["data"]["u"] = snapshot.update_id
    stale_payload["data"]["seq"] = snapshot.seq
    stale_delta = parse_orderbook_payload(stale_payload)

    builder = BybitOrderBookBuilder(symbol="BTCUSDT")
    builder.apply(snapshot)
    with pytest.raises(OrderBookOutOfOrderError):
        builder.apply(stale_delta)
