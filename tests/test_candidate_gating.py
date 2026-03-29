from btc_alert_engine.schemas import CrowdingFeatureSnapshot, MicroFeatureSnapshot, PriceBar, RegimeFeatureSnapshot, TrendFeatureSnapshot
from btc_alert_engine.strategy.bybit_candidates import build_continuation_candidates


def _make_trending_bars(n: int = 120) -> list[PriceBar]:
    base_ts = 1_700_000_000_000
    bars = []
    for i in range(n):
        ts = base_ts + (i + 1) * 900_000
        open_px = 100 + i * 0.5
        close_px = open_px + 0.8
        high_px = close_px + 0.2
        low_px = open_px - 0.1
        if i >= 110:
            close_px += 1.2
            high_px += 1.5
        bars.append(PriceBar(ts=ts, symbol="BTCUSDT", interval="15", open=open_px, high=high_px, low=low_px, close=close_px, volume=10, turnover=1000))
    return bars


def test_continuation_candidate_gating_flags() -> None:
    bars = _make_trending_bars()
    trend = []
    regime_bad = []
    regime_good = []
    crowding_veto = []
    crowding_ok = []
    micro_bad = []
    micro_good = []
    micro_false_gate_but_positive_fallback = []
    for bar in bars:
        trend.append(
            TrendFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                ret_15m_1=0.01,
                ret_1h_4=0.04,
                ret_4h_6=0.08,
                ema50_4h_gap=0.03,
                ema50_4h_slope=0.02,
                ema200_4h_slope=0.01,
                adx14_4h=24,
                atr_pctile_90d=55,
                breakout_age_1h=5,
                impulse_atr_1h=1.4,
                pullback_depth_frac=0.35,
                pullback_bars=2,
                dist_to_breakout_level=0.01,
                dist_to_ema50_4h=0.015,
            )
        )
        regime_bad.append(
            RegimeFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                rv_1d=0.02,
                rv_7d=0.03,
                atr_pctile_90d=55,
                jump_intensity_1d=0.05,
                mark_index_gap=0.0005,
                premium_index_15m=0.001,
                premium_z_7d=0.2,
                stress_score=0.75,
                trend_score=0.10,
                range_score=0.70,
            )
        )
        regime_good.append(
            RegimeFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                rv_1d=0.02,
                rv_7d=0.03,
                atr_pctile_90d=55,
                jump_intensity_1d=0.05,
                mark_index_gap=0.0005,
                premium_index_15m=0.001,
                premium_z_7d=0.2,
                stress_score=0.10,
                trend_score=0.70,
                range_score=0.15,
            )
        )
        crowding_veto.append(
            CrowdingFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                crowding_long_score=3.5,
                veto_long=True,
            )
        )
        crowding_ok.append(
            CrowdingFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                crowding_long_score=0.5,
                veto_long=False,
            )
        )
        micro_bad.append(
            MicroFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                ofi_60s=-10,
                median_bookimb_l10_60s=-0.05,
                spread_z=3.0,
                vwap_mid_dev_30s_z=4.0,
                gate_pass_long=False,
                gate_pass_short=False,
            )
        )
        micro_good.append(
            MicroFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                ofi_60s=200,
                median_bookimb_l10_60s=0.08,
                spread_z=0.2,
                vwap_mid_dev_30s_z=0.1,
                gate_pass_long=True,
                gate_pass_short=False,
            )
        )
        micro_false_gate_but_positive_fallback.append(
            MicroFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                ofi_60s=200,
                median_bookimb_l10_60s=0.08,
                spread_z=0.2,
                vwap_mid_dev_30s_z=0.1,
                gate_pass_long=False,
                gate_pass_short=False,
            )
        )

    ungated = build_continuation_candidates(
        bars,
        trend,
        regime_bad,
        crowding_veto,
        micro_bad,
        symbol="BTCUSDT",
        require_regime_gate=False,
        require_crowding_veto=False,
        require_micro_gate=False,
    )
    assert ungated

    regime_gated = build_continuation_candidates(
        bars,
        trend,
        regime_bad,
        crowding_ok,
        micro_good,
        symbol="BTCUSDT",
        require_regime_gate=True,
        require_crowding_veto=False,
        require_micro_gate=False,
    )
    assert not regime_gated

    crowding_gated = build_continuation_candidates(
        bars,
        trend,
        regime_good,
        crowding_veto,
        micro_good,
        symbol="BTCUSDT",
        require_regime_gate=True,
        require_crowding_veto=True,
        require_micro_gate=False,
    )
    assert not crowding_gated

    micro_gated = build_continuation_candidates(
        bars,
        trend,
        regime_good,
        crowding_ok,
        micro_bad,
        symbol="BTCUSDT",
        require_regime_gate=True,
        require_crowding_veto=False,
        require_micro_gate=True,
    )
    assert not micro_gated

    strict_micro_gate = build_continuation_candidates(
        bars,
        trend,
        regime_good,
        crowding_ok,
        micro_false_gate_but_positive_fallback,
        symbol="BTCUSDT",
        require_regime_gate=True,
        require_crowding_veto=False,
        require_micro_gate=True,
    )
    assert not strict_micro_gate

    fully_gated = build_continuation_candidates(
        bars,
        trend,
        regime_good,
        crowding_ok,
        micro_good,
        symbol="BTCUSDT",
        require_regime_gate=True,
        require_crowding_veto=True,
        require_micro_gate=True,
    )
    assert fully_gated


def test_continuation_candidates_tolerate_missing_optional_feature_frames() -> None:
    bars = _make_trending_bars()
    trend = []
    for bar in bars:
        trend.append(
            TrendFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                ret_15m_1=0.01,
                ret_1h_4=0.04,
                ret_4h_6=0.08,
                ema50_4h_gap=0.03,
                ema50_4h_slope=0.02,
                ema200_4h_slope=0.01,
                adx14_4h=24,
                atr_pctile_90d=55,
                breakout_age_1h=5,
                impulse_atr_1h=1.4,
                pullback_depth_frac=0.35,
                pullback_bars=2,
                dist_to_breakout_level=0.01,
                dist_to_ema50_4h=0.015,
            )
        )

    candidates = build_continuation_candidates(
        bars,
        trend,
        [],
        [],
        [],
        symbol="BTCUSDT",
        require_regime_gate=False,
        require_crowding_veto=False,
        require_micro_gate=False,
    )
    assert candidates


def _make_downtrending_bars(n: int = 120) -> list[PriceBar]:
    base_ts = 1_700_000_000_000
    bars = []
    for i in range(n):
        ts = base_ts + (i + 1) * 900_000
        open_px = 200 - i * 0.6
        close_px = open_px - 0.8
        high_px = open_px + 0.1
        low_px = close_px - 0.2
        if i >= 110:
            close_px -= 1.2
            low_px -= 1.5
        bars.append(PriceBar(ts=ts, symbol="BTCUSDT", interval="15", open=open_px, high=high_px, low=low_px, close=close_px, volume=10, turnover=1000))
    return bars


def test_continuation_short_candidates_can_be_mirrored() -> None:
    bars = _make_downtrending_bars()
    trend = []
    regime = []
    crowding = []
    micro = []
    for bar in bars:
        trend.append(
            TrendFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                ret_15m_1=-0.01,
                ret_1h_4=-0.04,
                ret_4h_6=-0.08,
                ema50_4h_gap=-0.03,
                ema50_4h_slope=-0.02,
                ema200_4h_slope=-0.01,
                adx14_4h=24,
                atr_pctile_90d=55,
                breakdown_age_1h=5,
                downside_impulse_atr_1h=1.4,
                bounce_depth_frac=0.35,
                bounce_bars=2,
                dist_to_breakdown_level=0.01,
                dist_below_ema50_4h=0.015,
            )
        )
        regime.append(
            RegimeFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                rv_1d=0.02,
                rv_7d=0.03,
                atr_pctile_90d=55,
                jump_intensity_1d=0.05,
                mark_index_gap=0.0005,
                premium_index_15m=0.001,
                premium_z_7d=0.2,
                stress_score=0.10,
                trend_score=0.70,
                range_score=0.15,
            )
        )
        crowding.append(CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", crowding_short_score=0.5, veto_short=False))
        micro.append(MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_60s=-200, median_bookimb_l10_60s=-0.08, spread_z=0.2, vwap_mid_dev_30s_z=0.1, gate_pass_long=False, gate_pass_short=True))

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
        sides=["short"],
    )
    assert candidates
    assert all(item.side == "short" for item in candidates)
