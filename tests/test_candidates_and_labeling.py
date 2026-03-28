from btc_alert_engine.research.labeling import label_candidates
from btc_alert_engine.schemas import CandidateEvent, CrowdingFeatureSnapshot, MicroBucket1s, MicroFeatureSnapshot, PriceBar, RegimeFeatureSnapshot, TrendFeatureSnapshot
from btc_alert_engine.strategy.bybit_candidates import build_continuation_candidates, build_stress_reversal_candidates


def _make_trending_bars(n: int = 120) -> list[PriceBar]:
    base_ts = 1_700_000_000_000
    bars = []
    for i in range(n):
        ts = base_ts + (i + 1) * 900_000
        open_px = 100 + i * 0.5
        close_px = open_px + 0.8
        high_px = close_px + 0.2
        low_px = open_px - 0.1
        # Add a stronger breakout late in the sample.
        if i >= 110:
            close_px += 1.2
            high_px += 1.5
        bars.append(PriceBar(ts=ts, symbol="BTCUSDT", interval="15", open=open_px, high=high_px, low=low_px, close=close_px, volume=10, turnover=1000))
    return bars


def test_build_continuation_candidates_and_labels() -> None:
    bars = _make_trending_bars()
    trend = []
    regime = []
    crowding = []
    micro = []
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
                stress_score=0.15,
                trend_score=0.70,
                range_score=0.15,
            )
        )
        crowding.append(
            CrowdingFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                funding_8h=0.0002,
                funding_z_7d=0.5,
                premium_index_15m=0.001,
                premium_z_7d=0.2,
                oi_level=1_000_000,
                oi_change_1h=0.01,
                oi_change_4h=0.02,
                long_short_ratio_1h=1.05,
                liq_longs_z_1h=0.1,
                liq_shorts_z_1h=0.1,
                crowding_long_score=0.4,
                crowding_short_score=-0.4,
                veto_long=False,
                veto_short=False,
            )
        )
        micro.append(
            MicroFeatureSnapshot(
                ts=bar.ts,
                symbol="BTCUSDT",
                ofi_10s=50,
                ofi_60s=200,
                ofi_300s=500,
                cum_delta_60s=20,
                spread_bps=1.0,
                spread_z=0.2,
                bookimb_l1=0.10,
                bookimb_l5=0.09,
                bookimb_l10=0.08,
                top10_depth_usd=2_000_000,
                depth_decay=0.12,
                vwap_mid_dev_30s=0.0002,
                vwap_mid_dev_30s_z=0.1,
                replenish_rate_30s=1.5,
                cancel_add_ratio_30s=0.6,
                micro_vol_60s=0.002,
                median_bookimb_l10_60s=0.08,
                gate_pass=True,
            )
        )

    candidates = build_continuation_candidates(bars, trend, regime, crowding, micro, symbol="BTCUSDT")
    assert candidates
    candidate = candidates[-1]
    assert candidate.module == "continuation_v1"
    assert candidate.side == "long"
    assert candidate.target_r_multiple == 2.0
    assert candidate.stop < candidate.entry < candidate.tp

    labels = label_candidates([candidate], bars, slippage_bps=0.0)
    assert labels
    label = labels[0]
    assert label.outcome in {"tp", "timeout", "sl"}
    assert label.executed_entry > 0
    assert label.executed_tp > label.executed_entry
    assert label.target_r_multiple == 2.0


def test_label_candidates_use_quote_entry_and_module_target_multiple() -> None:
    bars = [
        PriceBar(ts=1_000, symbol="BTCUSDT", interval="15", open=100.0, high=101.0, low=99.5, close=100.5, volume=10, turnover=1000),
        PriceBar(ts=901_000, symbol="BTCUSDT", interval="15", open=100.0, high=105.0, low=99.8, close=104.5, volume=10, turnover=1000),
        PriceBar(ts=1_801_000, symbol="BTCUSDT", interval="15", open=104.5, high=106.0, low=103.5, close=105.5, volume=10, turnover=1000),
    ]
    quotes = [MicroBucket1s(ts=2_000, symbol="BTCUSDT", best_bid_price=100.8, best_ask_price=101.0)]
    candidate = CandidateEvent(
        ts=1_000,
        venue="bybit",
        symbol="BTCUSDT",
        module="stress_reversal_v0",
        side="long",
        entry=100.5,
        stop=99.5,
        tp=102.0,
        target_r_multiple=1.5,
        timeout_bars=32,
        rule_reasons=["test"],
        veto_reasons=[],
    )

    labels = label_candidates([candidate], bars, micro_buckets=quotes, slippage_bps=0.0)
    assert labels
    label = labels[0]
    assert label.executed_entry == 101.0
    assert label.target_r_multiple == 1.5
    assert label.executed_tp == 103.25
    assert label.tp_before_sl_within_horizon is True
    assert label.outcome == "tp"


def test_build_stress_reversal_candidates() -> None:
    bars = []
    base_ts = 1_700_000_000_000
    for i in range(60):
        ts = base_ts + (i + 1) * 900_000
        open_px = 100.0
        high_px = 100.5
        low_px = 99.5
        close_px = 100.0
        if i == 40:
            low_px = 95.0
            close_px = 99.9
            high_px = 100.2
        if i == 41:
            open_px = 100.0
            close_px = 101.2
            high_px = 101.5
            low_px = 99.8
        bars.append(PriceBar(ts=ts, symbol="BTCUSDT", interval="15", open=open_px, high=high_px, low=low_px, close=close_px, volume=10, turnover=1000))
    regime = []
    crowding = []
    micro = []
    for bar in bars:
        stress = 0.85 if bar.ts == bars[41].ts else 0.2
        liq_long = 3.2 if bar.ts == bars[41].ts else 0.1
        regime.append(RegimeFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", rv_1d=0.04, rv_7d=0.03, atr_pctile_90d=90, jump_intensity_1d=0.3, mark_index_gap=0.001, premium_index_15m=0.0, premium_z_7d=0.1, stress_score=stress, trend_score=0.1, range_score=0.1))
        crowding.append(CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", funding_8h=0.0, funding_z_7d=0.0, premium_index_15m=0.0, premium_z_7d=0.0, oi_level=1_000_000, oi_change_1h=0.0, oi_change_4h=0.0, long_short_ratio_1h=1.0, liq_longs_z_1h=liq_long, liq_shorts_z_1h=0.1, crowding_long_score=0.0, crowding_short_score=0.0))
        micro.append(MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_10s=10, ofi_60s=10, ofi_300s=10, cum_delta_60s=1, spread_bps=1.0, spread_z=0.0, bookimb_l1=0.1, bookimb_l5=0.1, bookimb_l10=0.1, top10_depth_usd=1_000_000, depth_decay=0.1, vwap_mid_dev_30s=0.0, vwap_mid_dev_30s_z=0.0, replenish_rate_30s=1.0, cancel_add_ratio_30s=1.0, micro_vol_60s=0.01, median_bookimb_l10_60s=0.1, gate_pass=True))

    candidates = build_stress_reversal_candidates(bars, regime, crowding, micro, symbol="BTCUSDT")
    assert candidates
    candidate = candidates[0]
    assert candidate.module == "stress_reversal_v0"
    assert candidate.entry > candidate.stop
    assert candidate.target_r_multiple == 1.5


def test_candidate_event_rejects_inconsistent_target_multiple() -> None:
    from pydantic import ValidationError

    try:
        CandidateEvent(
            ts=1_000,
            venue="bybit",
            symbol="BTCUSDT",
            module="continuation_v1",
            side="long",
            entry=100.0,
            stop=99.0,
            tp=101.2,
            target_r_multiple=2.0,
            timeout_bars=96,
            rule_reasons=["test"],
            veto_reasons=[],
        )
    except ValidationError:
        return
    raise AssertionError("Expected CandidateEvent validation to fail for inconsistent tp and target_r_multiple")
