from __future__ import annotations

import pandas as pd
import pytest

from btc_alert_engine.features.indicators import atr
from btc_alert_engine.schemas import CrowdingFeatureSnapshot, MicroFeatureSnapshot, PriceBar, RegimeFeatureSnapshot, TrendFeatureSnapshot
from btc_alert_engine.strategy.bybit_candidates import build_continuation_candidates


def _bars_to_df(bars: list[PriceBar]) -> pd.DataFrame:
    rows = [bar.model_dump(mode="json") for bar in bars]
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)


def _make_fast_long_bars(n: int = 18) -> list[PriceBar]:
    base_ts = 1_700_000_000_000
    bars: list[PriceBar] = []
    close_px = 100.0
    for i in range(n):
        ts = base_ts + (i + 1) * 300_000
        open_px = close_px
        if i < n - 2:
            close_px = open_px + 0.25
            high_px = close_px + 0.20
            low_px = open_px - 0.20
        elif i == n - 2:
            close_px = open_px + 1.10
            high_px = close_px + 0.20
            low_px = open_px - 0.15
        else:
            close_px = open_px + 0.90
            high_px = close_px + 0.15
            low_px = open_px - 0.10
        bars.append(PriceBar(ts=ts, symbol="BTCUSDT", interval="5", open=open_px, high=high_px, low=low_px, close=close_px, volume=10, turnover=1000))
    return bars


def _make_fast_short_bars(n: int = 18) -> list[PriceBar]:
    base_ts = 1_700_100_000_000
    bars: list[PriceBar] = []
    close_px = 120.0
    for i in range(n):
        ts = base_ts + (i + 1) * 300_000
        open_px = close_px
        if i < n - 2:
            close_px = open_px - 0.25
            high_px = open_px + 0.20
            low_px = close_px - 0.20
        elif i == n - 2:
            close_px = open_px - 1.10
            high_px = open_px + 0.15
            low_px = close_px - 0.20
        else:
            close_px = open_px - 0.90
            high_px = open_px + 0.10
            low_px = close_px - 0.15
        bars.append(PriceBar(ts=ts, symbol="BTCUSDT", interval="5", open=open_px, high=high_px, low=low_px, close=close_px, volume=10, turnover=1000))
    return bars


def _long_inputs(bars: list[PriceBar], *, early_invalid: bool = False) -> tuple[list[TrendFeatureSnapshot], list[RegimeFeatureSnapshot], list[CrowdingFeatureSnapshot], list[MicroFeatureSnapshot]]:
    trend: list[TrendFeatureSnapshot] = []
    regime: list[RegimeFeatureSnapshot] = []
    crowding: list[CrowdingFeatureSnapshot] = []
    micro: list[MicroFeatureSnapshot] = []
    episode_id = bars[-3].ts
    for i, bar in enumerate(bars):
        break_age = 99 if early_invalid and i < len(bars) - 2 else 3
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
                ret_regime_6=0.05,
                ema_fast_regime_gap=0.02,
                ema_fast_regime_slope=0.008,
                ema_slow_regime_slope=0.004,
                adx_regime=22,
                atr_pctile_90d=55,
                setup_break_age=break_age,
                setup_impulse_atr=1.3,
                setup_pullback_depth_frac=0.35,
                setup_pullback_bars=2,
                setup_break_episode_id=episode_id,
                setup_breakout_anchor_low=101.0,
                dist_to_setup_breakout_level=0.01,
                dist_to_regime_ema=0.02,
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
                atr_pctile_90d=55,
                jump_intensity_1d=0.05,
                mark_index_gap=0.0005,
                premium_index_trigger=0.0008,
                premium_z_7d=0.1,
                stress_score=0.1,
                trend_score=0.7,
                range_score=0.2,
            )
        )
        crowding.append(CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", crowding_long_score=0.1, crowding_short_score=-0.1, veto_long=False, veto_short=False))
        micro.append(MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_60s=100, median_bookimb_l10_60s=0.1, spread_z=0.2, vwap_mid_dev_30s_z=0.1, gate_pass_long=True, gate_pass_short=False))
    return trend, regime, crowding, micro


def _short_inputs(bars: list[PriceBar]) -> tuple[list[TrendFeatureSnapshot], list[RegimeFeatureSnapshot], list[CrowdingFeatureSnapshot], list[MicroFeatureSnapshot]]:
    trend: list[TrendFeatureSnapshot] = []
    regime: list[RegimeFeatureSnapshot] = []
    crowding: list[CrowdingFeatureSnapshot] = []
    micro: list[MicroFeatureSnapshot] = []
    episode_id = bars[-3].ts
    for bar in bars:
        trend.append(
            TrendFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                profile_id="fast",
                trigger_interval="5m",
                setup_interval="15m",
                regime_interval="1h",
                ret_trigger_1=-0.01,
                ret_setup_4=-0.03,
                ret_regime_6=-0.05,
                ema_fast_regime_gap=-0.02,
                ema_fast_regime_slope=-0.008,
                ema_slow_regime_slope=-0.004,
                adx_regime=22,
                atr_pctile_90d=55,
                setup_breakdown_age=3,
                setup_downside_impulse_atr=1.3,
                setup_bounce_depth_frac=0.35,
                setup_bounce_bars=2,
                setup_breakdown_episode_id=episode_id,
                setup_breakdown_anchor_high=119.5,
                dist_to_setup_breakdown_level=0.01,
                dist_below_regime_ema=0.02,
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
                atr_pctile_90d=55,
                jump_intensity_1d=0.05,
                mark_index_gap=0.0005,
                premium_index_trigger=0.0008,
                premium_z_7d=0.1,
                stress_score=0.1,
                trend_score=0.7,
                range_score=0.2,
            )
        )
        crowding.append(CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", crowding_long_score=-0.1, crowding_short_score=0.1, veto_long=False, veto_short=False))
        micro.append(MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_60s=-100, median_bookimb_l10_60s=-0.1, spread_z=0.2, vwap_mid_dev_30s_z=0.1, gate_pass_long=False, gate_pass_short=True))
    return trend, regime, crowding, micro


def test_fast_continuation_default_stop_is_setup_anchored() -> None:
    bars = _make_fast_long_bars()
    trend, regime, crowding, micro = _long_inputs(bars)
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
    atr_series = atr(_bars_to_df(bars), 14)
    trigger_ts = pd.to_datetime(candidate.ts, unit="ms", utc=True)
    atr_trigger = float(atr_series.loc[trigger_ts])
    expected_stop = 101.0 - 0.15 * atr_trigger
    assert candidate.stop == pytest.approx(expected_stop, rel=1e-6)


def test_fast_continuation_emits_only_first_trigger_per_setup_by_default() -> None:
    bars = _make_fast_long_bars()
    trend, regime, crowding, micro = _long_inputs(bars, early_invalid=True)
    default_candidates = build_continuation_candidates(
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
    multi_candidates = build_continuation_candidates(
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
        generator_params={"one_trigger_per_setup": False, "stop_anchor_mode": "setup_anchor", "stop_buffer_atr_trigger": 0.15},
    )
    assert len(default_candidates) == 1
    assert len(multi_candidates) == 2


def test_fast_continuation_supports_side_specific_short_params() -> None:
    bars = _make_fast_short_bars()
    trend, regime, crowding, micro = _short_inputs(bars)
    candidates = build_continuation_candidates(
        bars,
        trend,
        regime,
        crowding,
        micro,
        symbol="BTCUSDT",
        profile="fast",
        sides=["short"],
        require_regime_gate=True,
        require_crowding_veto=True,
        require_micro_gate=True,
        generator_params={
            "stop_anchor_mode": "setup_anchor",
            "stop_buffer_atr_trigger": 0.10,
            "target_r_multiple": 1.25,
            "short": {"stop_buffer_atr_trigger": 0.20, "target_r_multiple": 1.50},
        },
    )
    assert candidates
    candidate = candidates[-1]
    assert candidate.side == "short"
    assert candidate.target_r_multiple == pytest.approx(1.50)
    assert candidate.stop > candidate.entry
    assert candidate.tp < candidate.entry
