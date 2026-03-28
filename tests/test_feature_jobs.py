import math
from pathlib import Path

from btc_alert_engine.features.bybit_foundation import (
    build_crowding_features,
    build_micro_features,
    build_regime_features,
    build_trend_features,
)
from btc_alert_engine.materialize.bybit_foundation import materialize_bybit_bars
from btc_alert_engine.schemas import MicroBucket1s, RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter


def _write_events(base_dir: Path, events: list[RawEvent]) -> None:
    with RawEventWriter(base_dir) as writer:
        for event in events:
            writer.write(event)


def _make_bar_events(base_ts: int, n: int) -> list[RawEvent]:
    rows_trade = []
    rows_index = []
    rows_premium = []
    for i in range(n):
        start = base_ts + i * 900_000
        cycle = math.sin(i / 20.0) * 2.0
        open_px = 20_000 + i * 2 + cycle
        close_px = open_px + 1.5 + math.sin(i / 10.0)
        high_px = max(open_px, close_px) + 0.8
        low_px = min(open_px, close_px) - 0.8
        rows_trade.append([str(start), f"{open_px:.4f}", f"{high_px:.4f}", f"{low_px:.4f}", f"{close_px:.4f}", "10", "1000"])
        rows_index.append([str(start), f"{open_px - 3:.4f}", f"{high_px - 3:.4f}", f"{low_px - 3:.4f}", f"{close_px - 3:.4f}"])
        premium = 0.0005 + 0.0001 * math.sin(i / 15.0)
        rows_premium.append([str(start), f"{premium:.6f}", f"{premium + 0.0001:.6f}", f"{premium - 0.0001:.6f}", f"{premium:.6f}"])
    event_ts = base_ts + n * 900_000 + 1_000
    return [
        RawEvent(source="bybit_rest", topic="rest.kline_15.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_trade}}),
        RawEvent(source="bybit_rest", topic="rest.index_price_kline_15.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_index}}),
        RawEvent(source="bybit_rest", topic="rest.premium_index_price_kline_15.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_premium}}),
    ]


def test_trend_and_regime_features(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    events = _make_bar_events(base_ts, 900)
    _write_events(tmp_path / "raw", events)
    bars = materialize_bybit_bars([tmp_path / "raw"], symbol="BTCUSDT")

    trend = build_trend_features(bars["trade_15m"], symbol="BTCUSDT")
    regime = build_regime_features(bars["trade_15m"], bars["index_price_15m"], bars["premium_index_15m"], symbol="BTCUSDT")

    assert len(trend) == len(bars["trade_15m"])
    assert len(regime) == len(bars["trade_15m"])
    last_trend = trend[-1]
    last_regime = regime[-1]
    assert last_trend.ema50_4h_gap is not None
    assert last_trend.ema50_4h_gap > 0
    assert last_trend.ret_4h_6 is not None
    assert last_regime.trend_score is not None and last_regime.range_score is not None and last_regime.stress_score is not None
    assert abs((last_regime.trend_score + last_regime.range_score + last_regime.stress_score) - 1.0) < 1e-6


def test_crowding_and_micro_features(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    rest_events = _make_bar_events(base_ts, 400)

    funding_rows = []
    oi_rows = []
    ratio_rows = []
    for i in range(60):
        ts = base_ts + i * 8 * 60 * 60 * 1000
        funding = 0.0001 + i * 0.00001
        funding_rows.append({"symbol": "BTCUSDT", "fundingRate": f"{funding:.8f}", "fundingRateTimestamp": str(ts)})
    for i in range(400):
        ts = base_ts + i * 15 * 60 * 1000
        oi_rows.append({"symbol": "BTCUSDT", "openInterest": f"{1_000_000 + i * 5000:.4f}", "timestamp": str(ts)})
        buy_ratio = 0.52 + 0.0005 * i
        sell_ratio = max(0.48 - 0.0002 * i, 0.05)
        ratio_rows.append({"symbol": "BTCUSDT", "buyRatio": f"{buy_ratio:.6f}", "sellRatio": f"{sell_ratio:.6f}", "timestamp": str(ts)})
    event_ts = base_ts + 400 * 15 * 60 * 1000 + 1_000
    rest_events.extend(
        [
            RawEvent(source="bybit_rest", topic="rest.funding_history.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": funding_rows}}),
            RawEvent(source="bybit_rest", topic="rest.open_interest_5min.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": oi_rows}}),
            RawEvent(source="bybit_rest", topic="rest.account_ratio_5min.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": ratio_rows}}),
        ]
    )
    _write_events(tmp_path / "raw", rest_events)
    bars = materialize_bybit_bars([tmp_path / "raw"], symbol="BTCUSDT")

    micro_buckets = []
    for i in range(1000):
        ts = base_ts + i * 1000
        micro_buckets.append(
            MicroBucket1s(
                ts=ts,
                symbol="BTCUSDT",
                mid_price=20_000 + i * 0.1,
                spread_bps=1.0,
                ofi_proxy=50.0,
                cum_delta=0.3,
                bookimb_l1=0.10,
                bookimb_l5=0.08,
                bookimb_l10=0.07,
                top10_depth_usd=2_000_000,
                depth_decay=0.12,
                vwap_mid_dev=0.0002,
                replenish_notional=5_000,
                cancel_notional=2_500,
                last_trade_price=20_000 + i * 0.1,
                buy_volume=1.5,
                sell_volume=0.8,
                long_liq_notional=100.0,
                short_liq_notional=50.0,
            )
        )

    crowding = build_crowding_features([tmp_path / "raw"], bars["premium_index_15m"], micro_buckets, symbol="BTCUSDT")
    micro = build_micro_features(micro_buckets, symbol="BTCUSDT")

    assert crowding
    assert micro
    last_crowding = crowding[-1]
    last_micro = micro[-1]
    assert last_crowding.crowding_long_score is not None
    assert last_crowding.crowding_long_score > 0
    assert last_crowding.long_short_ratio_1h is not None and last_crowding.long_short_ratio_1h > 1
    assert last_micro.ofi_60s is not None and last_micro.ofi_60s > 0
    assert last_micro.median_bookimb_l10_60s is not None and last_micro.median_bookimb_l10_60s > 0.05
    assert last_micro.gate_pass is True
