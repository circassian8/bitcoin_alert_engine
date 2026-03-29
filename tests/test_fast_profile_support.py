from pathlib import Path

from btc_alert_engine.features.bybit_foundation import build_regime_features, build_trend_features
from btc_alert_engine.materialize.bybit_foundation import materialize_bybit_bars
from btc_alert_engine.schemas import CrowdingFeatureSnapshot, MicroFeatureSnapshot, PriceBar, RawEvent, RegimeFeatureSnapshot, TrendFeatureSnapshot
from btc_alert_engine.storage.raw_ndjson import RawEventWriter
from btc_alert_engine.strategy.bybit_candidates import build_continuation_candidates


def _write_events(base_dir: Path, events: list[RawEvent]) -> None:
    with RawEventWriter(base_dir) as writer:
        for event in events:
            writer.write(event)


def _make_multi_interval_bar_events(base_ts: int, n5: int = 600) -> list[RawEvent]:
    rows_trade_5 = []
    rows_index_5 = []
    rows_premium_5 = []
    rows_trade_15 = []
    rows_index_15 = []
    rows_premium_15 = []
    for i in range(n5):
        start = base_ts + i * 300_000
        open_px = 30_000 + i * 0.8
        close_px = open_px + 0.6
        high_px = close_px + 0.2
        low_px = open_px - 0.2
        rows_trade_5.append([str(start), f"{open_px:.4f}", f"{high_px:.4f}", f"{low_px:.4f}", f"{close_px:.4f}", "12", "1200"])
        rows_index_5.append([str(start), f"{open_px - 2:.4f}", f"{high_px - 2:.4f}", f"{low_px - 2:.4f}", f"{close_px - 2:.4f}"])
        premium = 0.0004 + 0.00005 * ((i % 20) / 20.0)
        rows_premium_5.append([str(start), f"{premium:.6f}", f"{premium + 0.00005:.6f}", f"{premium - 0.00005:.6f}", f"{premium:.6f}"])
        if i % 3 == 0:
            rows_trade_15.append([str(start), f"{open_px:.4f}", f"{high_px + 0.4:.4f}", f"{low_px - 0.4:.4f}", f"{close_px + 0.2:.4f}", "36", "3600"])
            rows_index_15.append([str(start), f"{open_px - 2:.4f}", f"{high_px - 1.6:.4f}", f"{low_px - 2.4:.4f}", f"{close_px - 1.8:.4f}"])
            rows_premium_15.append([str(start), f"{premium:.6f}", f"{premium + 0.00008:.6f}", f"{premium - 0.00008:.6f}", f"{premium:.6f}"])
    event_ts = base_ts + n5 * 300_000 + 1000
    return [
        RawEvent(source="bybit_rest", topic="rest.kline_5.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_trade_5}}),
        RawEvent(source="bybit_rest", topic="rest.index_price_kline_5.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_index_5}}),
        RawEvent(source="bybit_rest", topic="rest.premium_index_price_kline_5.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_premium_5}}),
        RawEvent(source="bybit_rest", topic="rest.kline_15.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_trade_15}}),
        RawEvent(source="bybit_rest", topic="rest.index_price_kline_15.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_index_15}}),
        RawEvent(source="bybit_rest", topic="rest.premium_index_price_kline_15.BTCUSDT", symbol="BTCUSDT", exchange_ts=event_ts, local_received_ts=event_ts, payload={"time": event_ts, "result": {"list": rows_premium_15}}),
    ]


def test_materialize_bybit_bars_supports_multiple_intervals(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    events = _make_multi_interval_bar_events(base_ts, n5=90)
    _write_events(tmp_path / "raw", events)
    bars = materialize_bybit_bars([tmp_path / "raw"], symbol="BTCUSDT")
    assert "trade_5m" in bars
    assert "trade_15m" in bars
    assert "index_price_5m" in bars
    assert "premium_index_5m" in bars
    assert len(bars["trade_5m"]) > len(bars["trade_15m"])


def test_fast_profile_feature_builders_emit_generic_fields(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    events = _make_multi_interval_bar_events(base_ts, n5=900)
    _write_events(tmp_path / "raw", events)
    bars = materialize_bybit_bars([tmp_path / "raw"], symbol="BTCUSDT")

    trend = build_trend_features(bars["trade_5m"], symbol="BTCUSDT", profile="fast")
    regime = build_regime_features(bars["trade_5m"], bars["index_price_5m"], bars["premium_index_5m"], symbol="BTCUSDT", profile="fast")

    assert len(trend) == len(bars["trade_5m"])
    assert len(regime) == len(bars["trade_5m"])
    assert trend[-1].profile_id == "fast"
    assert trend[-1].trigger_interval == "5m"
    assert trend[-1].setup_interval == "15m"
    assert trend[-1].regime_interval == "1h"
    assert trend[-1].ret_trigger_1 is not None
    assert trend[-1].ret_setup_4 is not None
    assert trend[-1].ret_regime_6 is not None
    assert trend[-1].setup_break_age is not None
    assert regime[-1].premium_index_trigger is not None


def _make_fast_bars(n: int = 120) -> list[PriceBar]:
    base_ts = 1_700_000_000_000
    bars: list[PriceBar] = []
    for i in range(n):
        ts = base_ts + (i + 1) * 300_000
        open_px = 100 + i * 0.2
        close_px = open_px + 0.3
        high_px = close_px + 0.1
        low_px = open_px - 0.1
        if i >= 110:
            close_px += 0.8
            high_px += 1.0
        bars.append(PriceBar(ts=ts, symbol="BTCUSDT", interval="5", open=open_px, high=high_px, low=low_px, close=close_px, volume=10, turnover=1000))
    return bars


def test_fast_continuation_candidates_use_fast_module_and_timeout() -> None:
    bars = _make_fast_bars()
    trend = []
    regime = []
    crowding = []
    micro = []
    for bar in bars:
        trend.append(
            TrendFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                profile_id="fast",
                trigger_interval="5m",
                setup_interval="15m",
                regime_interval="1h",
                ret_trigger_1=0.01,
                ret_setup_4=0.03,
                ret_regime_6=0.06,
                ema_fast_regime_gap=0.02,
                ema_fast_regime_slope=0.01,
                ema_slow_regime_slope=0.005,
                adx_regime=22,
                atr_pctile_90d=50,
                setup_break_age=4,
                setup_impulse_atr=1.2,
                setup_pullback_depth_frac=0.35,
                setup_pullback_bars=2,
                dist_to_setup_breakout_level=0.01,
                dist_to_regime_ema=0.015,
            )
        )
        regime.append(
            RegimeFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                profile_id="fast",
                trigger_interval="5m",
                setup_interval="15m",
                regime_interval="1h",
                rv_1d=0.03,
                rv_7d=0.04,
                atr_pctile_90d=50,
                jump_intensity_1d=0.05,
                mark_index_gap=0.0005,
                premium_index_trigger=0.0008,
                premium_z_7d=0.1,
                stress_score=0.1,
                trend_score=0.7,
                range_score=0.2,
            )
        )
        crowding.append(CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", crowding_long_score=0.2, veto_long=False))
        micro.append(MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_60s=100, median_bookimb_l10_60s=0.1, spread_z=0.2, vwap_mid_dev_30s_z=0.1, gate_pass_long=True))

    candidates = build_continuation_candidates(
        bars,
        trend,
        regime,
        crowding,
        micro,
        symbol="BTCUSDT",
        profile="fast",
        sides=["long"],
        require_regime_gate=True,
        require_crowding_veto=True,
        require_micro_gate=True,
    )
    assert candidates
    candidate = candidates[-1]
    assert candidate.module == "continuation_v1_fast"
    assert candidate.timeout_bars == 288
