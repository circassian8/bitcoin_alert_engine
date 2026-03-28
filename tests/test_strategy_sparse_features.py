from btc_alert_engine.schemas import CrowdingFeatureSnapshot, MicroFeatureSnapshot, PriceBar, RegimeFeatureSnapshot, TrendFeatureSnapshot
from btc_alert_engine.strategy.bybit_candidates import build_continuation_candidates, build_stress_reversal_candidates


def _bars(n: int = 80) -> list[PriceBar]:
    base_ts = 1_700_000_000_000
    rows: list[PriceBar] = []
    for i in range(n):
        ts = base_ts + (i + 1) * 900_000
        rows.append(
            PriceBar(
                ts=ts,
                symbol="BTCUSDT",
                interval="15",
                open=100.0 + i * 0.1,
                high=100.8 + i * 0.1,
                low=99.6 + i * 0.1,
                close=100.4 + i * 0.1,
                volume=10.0,
                turnover=1000.0,
            )
        )
    return rows


def test_continuation_candidates_tolerate_nullable_joined_features() -> None:
    bars = _bars()
    trend = [
        TrendFeatureSnapshot(
            ts=bar.ts,
            symbol="BTCUSDT",
            ema50_4h_gap=None,
            ema50_4h_slope=None,
            ema200_4h_slope=None,
            breakout_age_1h=None,
            impulse_atr_1h=None,
            pullback_depth_frac=None,
            dist_to_breakout_level=None,
            dist_to_ema50_4h=None,
        )
        for bar in bars
    ]
    regime = [RegimeFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", stress_score=None, trend_score=None, range_score=None) for bar in bars]
    crowding = [CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", crowding_long_score=None, veto_long=False) for bar in bars]
    micro = [MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_60s=None, median_bookimb_l10_60s=None, gate_pass=False) for bar in bars]

    candidates = build_continuation_candidates(
        bars,
        trend,
        regime,
        crowding,
        micro,
        symbol="BTCUSDT",
        require_regime_gate=True,
        require_crowding_veto=True,
        require_micro_gate=True,
    )

    assert candidates == []


def test_stress_reversal_candidates_tolerate_nullable_joined_features() -> None:
    bars = _bars()
    regime = [RegimeFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", stress_score=None, trend_score=None, range_score=None) for bar in bars]
    crowding = [CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", liq_longs_z_1h=None, veto_long=False) for bar in bars]
    micro = [MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_60s=None, gate_pass=False) for bar in bars]

    candidates = build_stress_reversal_candidates(
        bars,
        regime,
        crowding,
        micro,
        symbol="BTCUSDT",
        require_crowding_veto=True,
        require_micro_gate=True,
    )

    assert candidates == []
