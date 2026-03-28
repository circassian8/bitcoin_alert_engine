from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from btc_alert_engine.backfill import backfill_bybit_rest_history, backfill_deribit_dvol_history, default_bybit_history_specs
from btc_alert_engine.research.labeling import label_candidates
from btc_alert_engine.schemas import CandidateEvent, MicroBucket1s, PriceBar, RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter, iter_raw_events


def _write_events(base_dir: Path, events: list[RawEvent]) -> None:
    with RawEventWriter(base_dir) as writer:
        for event in events:
            writer.write(event)


def test_label_candidates_prefers_raw_event_tape_for_entry_and_exit(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    signal_ts = 1_000
    events = [
        RawEvent(
            source="bybit_ws",
            topic="orderbook.200.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=1_500,
            local_received_ts=1_500,
            payload={
                "topic": "orderbook.200.BTCUSDT",
                "type": "snapshot",
                "ts": 1_500,
                "cts": 1_500,
                "data": {
                    "s": "BTCUSDT",
                    "u": 1,
                    "seq": 1,
                    "b": [["100.0", "1.0"]],
                    "a": [["101.0", "1.0"]],
                },
            },
        ),
        RawEvent(
            source="bybit_ws",
            topic="publicTrade.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=1_800,
            local_received_ts=1_800,
            payload={
                "topic": "publicTrade.BTCUSDT",
                "ts": 1_800,
                "data": [{"T": 1_800, "s": "BTCUSDT", "S": "Buy", "v": "0.1", "p": "102.1", "i": "t1", "seq": 1}],
            },
        ),
        RawEvent(
            source="bybit_ws",
            topic="publicTrade.BTCUSDT",
            symbol="BTCUSDT",
            exchange_ts=1_900,
            local_received_ts=1_900,
            payload={
                "topic": "publicTrade.BTCUSDT",
                "ts": 1_900,
                "data": [{"T": 1_900, "s": "BTCUSDT", "S": "Sell", "v": "0.1", "p": "99.0", "i": "t2", "seq": 2}],
            },
        ),
    ]
    _write_events(raw_dir, events)

    bars = [
        PriceBar(ts=signal_ts, symbol="BTCUSDT", interval="15", open=100.0, high=100.8, low=99.8, close=100.5, volume=1, turnover=100),
        PriceBar(ts=901_000, symbol="BTCUSDT", interval="15", open=100.5, high=103.0, low=99.0, close=102.0, volume=1, turnover=100),
    ]
    candidate = CandidateEvent(
        ts=signal_ts,
        venue="bybit",
        symbol="BTCUSDT",
        module="continuation_v1",
        side="long",
        entry=100.5,
        stop=100.0,
        tp=101.0,
        target_r_multiple=1.0,
        timeout_bars=96,
        rule_reasons=["test"],
        veto_reasons=[],
    )

    labels = label_candidates([candidate], bars, raw_paths=[raw_dir], slippage_bps=0.0)
    assert labels
    label = labels[0]
    assert label.entry_source == "raw_quote"
    assert label.path_source == "raw_events"
    assert label.executed_entry == 101.0
    assert label.outcome == "tp"
    assert label.exit_ts == 1_800
    assert label.executed_exit == 102.0


def test_label_candidates_fallback_to_micro_bucket_path() -> None:
    bars = [
        PriceBar(ts=1_000, symbol="BTCUSDT", interval="15", open=100.0, high=100.8, low=99.8, close=100.5, volume=1, turnover=100),
        PriceBar(ts=901_000, symbol="BTCUSDT", interval="15", open=100.5, high=103.0, low=99.0, close=102.0, volume=1, turnover=100),
    ]
    micro = [
        MicroBucket1s(ts=2_000, symbol="BTCUSDT", best_bid_price=100.8, best_ask_price=101.0, best_bid_low=100.8, best_bid_high=100.8, best_ask_low=101.0, best_ask_high=101.0),
        MicroBucket1s(ts=3_000, symbol="BTCUSDT", best_bid_price=102.1, best_ask_price=102.3, best_bid_low=101.8, best_bid_high=102.2, best_ask_low=102.0, best_ask_high=102.4, trade_low=101.9, trade_high=102.3, last_trade_price=102.2),
    ]
    candidate = CandidateEvent(
        ts=1_000,
        venue="bybit",
        symbol="BTCUSDT",
        module="continuation_v1",
        side="long",
        entry=100.5,
        stop=100.0,
        tp=101.0,
        target_r_multiple=1.0,
        timeout_bars=96,
        rule_reasons=["test"],
        veto_reasons=[],
    )

    labels = label_candidates([candidate], bars, micro_buckets=micro, slippage_bps=0.0)
    assert labels
    label = labels[0]
    assert label.entry_source == "micro_quote"
    assert label.path_source == "micro_buckets"
    assert label.outcome == "tp"


def test_backfill_bybit_rest_history_writes_events(tmp_path: Path) -> None:
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = {"retCode": 0, "retMsg": "OK", "result": {"list": []}, "time": 1234567890}
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as client:
            with RawEventWriter(tmp_path / "raw") as writer:
                counts = await backfill_bybit_rest_history(
                    writer=writer,
                    symbol="BTCUSDT",
                    category="linear",
                    start_ms=0,
                    end_ms=1_000,
                    datasets=["kline_15", "funding_history"],
                    client=client,
                )
            assert counts == {"kline_15": 1, "funding_history": 1}

    asyncio.run(_run())
    events = list(iter_raw_events([tmp_path / "raw"]))
    assert len(events) == 2
    assert {event.topic for event in events} == {"rest.kline_15.BTCUSDT", "rest.funding_history.BTCUSDT"}
    assert all(event.metadata.get("backfill") is True for event in events)


def test_backfill_deribit_dvol_history_writes_events(tmp_path: Path) -> None:
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            payload = {"jsonrpc": "2.0", "id": body["id"], "result": {"data": [[0, 70.0, 71.0, 69.0, 70.5]]}}
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as client:
            with RawEventWriter(tmp_path / "raw") as writer:
                count = await backfill_deribit_dvol_history(
                    writer=writer,
                    currency="BTC",
                    resolution="60",
                    start_ms=0,
                    end_ms=1_000,
                    client=client,
                )
            assert count == 1

    asyncio.run(_run())
    events = list(iter_raw_events([tmp_path / "raw"]))
    assert len(events) == 1
    assert events[0].topic == "deribit.volatility_index.BTC.60"
    assert events[0].metadata.get("backfill") is True


def test_bybit_history_specs_support_hourly_pilot_intervals() -> None:
    specs = default_bybit_history_specs(oi_interval="1h", account_ratio_period="1h")
    names = {spec.name for spec in specs}
    assert "open_interest_1h" in names
    assert "account_ratio_1h" in names


def test_backfill_bybit_rest_history_accepts_generic_dataset_aliases(tmp_path: Path) -> None:
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = {"retCode": 0, "retMsg": "OK", "result": {"list": []}, "time": 1234567890}
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as client:
            with RawEventWriter(tmp_path / "raw") as writer:
                counts = await backfill_bybit_rest_history(
                    writer=writer,
                    symbol="BTCUSDT",
                    category="linear",
                    start_ms=0,
                    end_ms=1_000,
                    datasets=["open_interest", "account_ratio"],
                    oi_interval="1h",
                    account_ratio_period="1h",
                    client=client,
                )
            assert counts == {"open_interest_1h": 1, "account_ratio_1h": 1}

    asyncio.run(_run())
    events = list(iter_raw_events([tmp_path / "raw"]))
    assert {event.topic for event in events} == {"rest.open_interest_1h.BTCUSDT", "rest.account_ratio_1h.BTCUSDT"}
