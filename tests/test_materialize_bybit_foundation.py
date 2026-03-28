import math
from pathlib import Path

from btc_alert_engine.materialize.bybit_foundation import materialize_bybit_bars, materialize_micro_buckets
from btc_alert_engine.schemas import RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter


def _write_events(base_dir: Path, events: list[RawEvent]) -> None:
    with RawEventWriter(base_dir) as writer:
        for event in events:
            writer.write(event)


def test_materialize_micro_buckets(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    events = [
        RawEvent(
            source="bybit_ws",
            topic="orderbook.200.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=base_ts,
            local_received_ts=base_ts,
            payload={
                "topic": "orderbook.200.BTCUSDT",
                "type": "snapshot",
                "ts": base_ts,
                "cts": base_ts,
                "data": {
                    "s": "BTCUSDT",
                    "u": 1,
                    "seq": 1,
                    "b": [["100", "1.0"], ["99", "2.0"]],
                    "a": [["101", "1.5"], ["102", "2.0"]],
                },
            },
        ),
        RawEvent(
            source="bybit_ws",
            topic="publicTrade.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=base_ts + 100,
            local_received_ts=base_ts + 100,
            payload={
                "topic": "publicTrade.BTCUSDT",
                "ts": base_ts + 100,
                "data": [{"T": base_ts + 100, "s": "BTCUSDT", "S": "Buy", "v": "0.5", "p": "101", "i": "t1", "seq": 1, "L": "PlusTick"}],
            },
        ),
        RawEvent(
            source="bybit_ws",
            topic="orderbook.200.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=base_ts + 500,
            local_received_ts=base_ts + 500,
            payload={
                "topic": "orderbook.200.BTCUSDT",
                "type": "delta",
                "ts": base_ts + 500,
                "cts": base_ts + 500,
                "data": {
                    "s": "BTCUSDT",
                    "u": 2,
                    "seq": 2,
                    "b": [["100", "2.0"]],
                    "a": [["101", "1.0"]],
                },
            },
        ),
        RawEvent(
            source="bybit_ws",
            topic="allLiquidation.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=base_ts + 800,
            local_received_ts=base_ts + 800,
            payload={
                "topic": "allLiquidation.BTCUSDT",
                "ts": base_ts + 800,
                "data": [{"T": base_ts + 800, "s": "BTCUSDT", "S": "Sell", "v": "0.3", "p": "99"}],
            },
        ),
        RawEvent(
            source="bybit_ws",
            topic="publicTrade.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=base_ts + 1200,
            local_received_ts=base_ts + 1200,
            payload={
                "topic": "publicTrade.BTCUSDT",
                "ts": base_ts + 1200,
                "data": [{"T": base_ts + 1200, "s": "BTCUSDT", "S": "Sell", "v": "0.2", "p": "100.5", "i": "t2", "seq": 2, "L": "MinusTick"}],
            },
        ),
    ]
    _write_events(tmp_path / "raw", events)

    buckets = materialize_micro_buckets([tmp_path / "raw"], symbol="BTCUSDT")
    assert len(buckets) == 2
    first, second = buckets
    assert first.trade_count == 1
    assert math.isclose(first.buy_volume, 0.5)
    assert math.isclose(first.long_liq_size, 0.3)
    assert first.first_trade_price == 101.0
    assert first.trade_high == 101.0
    assert first.trade_low == 101.0
    assert first.best_bid_low == 100.0
    assert first.best_bid_high == 100.0
    assert first.bookimb_l10 is not None and first.bookimb_l10 > 0
    assert first.replenish_notional > 0
    assert second.trade_count == 1
    assert math.isclose(second.sell_volume, 0.2)
    assert second.spread_bps is not None and second.spread_bps > 0


def test_materialize_bybit_bars(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    rows_trade = []
    rows_index = []
    rows_premium = []
    for i in range(12):
        start = base_ts + i * 900_000
        open_px = 100 + i
        close_px = open_px + 0.5
        high_px = close_px + 0.2
        low_px = open_px - 0.2
        rows_trade.append([str(start), str(open_px), str(high_px), str(low_px), str(close_px), "10", "1000"])
        rows_index.append([str(start), str(open_px - 0.1), str(high_px - 0.1), str(low_px - 0.1), str(close_px - 0.1)])
        rows_premium.append([str(start), "0.0010", "0.0012", "0.0008", "0.0011"])
    event_ts = base_ts + 12 * 900_000 + 1_000
    events = [
        RawEvent(
            source="bybit_rest",
            topic="rest.kline_15.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=event_ts,
            local_received_ts=event_ts,
            payload={"time": event_ts, "result": {"list": rows_trade}},
        ),
        RawEvent(
            source="bybit_rest",
            topic="rest.index_price_kline_15.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=event_ts,
            local_received_ts=event_ts,
            payload={"time": event_ts, "result": {"list": rows_index}},
        ),
        RawEvent(
            source="bybit_rest",
            topic="rest.premium_index_price_kline_15.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=event_ts,
            local_received_ts=event_ts,
            payload={"time": event_ts, "result": {"list": rows_premium}},
        ),
    ]
    _write_events(tmp_path / "raw", events)
    bars = materialize_bybit_bars([tmp_path / "raw"], symbol="BTCUSDT")
    assert len(bars["trade_15m"]) == 12
    assert len(bars["index_price_15m"]) == 12
    assert len(bars["premium_index_15m"]) == 12
    assert bars["trade_15m"][-1].close > bars["trade_15m"][0].close
