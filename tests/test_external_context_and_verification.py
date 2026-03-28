from __future__ import annotations

from pathlib import Path

from btc_alert_engine.config import load_research_registry
from btc_alert_engine.features.external_context import (
    build_aggregate_derivs_coinglass_features,
    build_macro_veto_features,
    build_onchain_cryptoquant_features,
    build_onchain_glassnode_features,
    build_options_deribit_features,
    build_options_glassnode_features,
)
from btc_alert_engine.research.experiments import build_experiment_event_frame
from btc_alert_engine.schemas import (
    CoinGlassAggregateFeatureSnapshot,
    CrowdingFeatureSnapshot,
    CryptoQuantOnChainFeatureSnapshot,
    DeribitOptionsFeatureSnapshot,
    GlassnodeOnChainFeatureSnapshot,
    GlassnodeOptionsFeatureSnapshot,
    MacroVetoFeatureSnapshot,
    MicroFeatureSnapshot,
    PriceBar,
    RawEvent,
    RegimeFeatureSnapshot,
    TrendFeatureSnapshot,
)
from btc_alert_engine.storage.partitioned_ndjson import PartitionedNdjsonWriter
from btc_alert_engine.storage.raw_ndjson import RawEventWriter
from btc_alert_engine.verification.project import run_static_contract_checks


def _make_trending_bars(n: int = 160) -> list[PriceBar]:
    base_ts = 1_700_000_000_000
    bars = []
    for i in range(n):
        ts = base_ts + (i + 1) * 900_000
        open_px = 100 + i * 0.45
        close_px = open_px + 0.7
        high_px = close_px + 0.25
        low_px = open_px - 0.15
        if i >= 145:
            close_px += 1.4
            high_px += 1.6
        bars.append(PriceBar(ts=ts, symbol="BTCUSDT", interval="15", open=open_px, high=high_px, low=low_px, close=close_px, volume=10, turnover=1000))
    return bars


def _write_events(base_dir: Path, events: list[RawEvent]) -> None:
    with RawEventWriter(base_dir) as writer:
        for event in events:
            writer.write(event)


def _vendor_events(base_ts: int, hours: int = 40) -> list[RawEvent]:
    last_ts = base_ts + hours * 60 * 60 * 1000
    deribit_rows = [[base_ts + i * 60 * 60 * 1000, 70.0 + i, 71.0 + i, 69.0 + i, 70.5 + i] for i in range(hours)]
    events = [
        RawEvent(
            source="deribit_http",
            topic="deribit.volatility_index.BTC.60",
            symbol="BTC",
            exchange_ts=last_ts,
            local_received_ts=last_ts,
            payload={"result": {"data": deribit_rows}},
        )
    ]

    def metric_event(source: str, topic: str, values: list[tuple[int, float]]) -> RawEvent:
        return RawEvent(
            source=source,
            topic=topic,
            symbol="BTC",
            exchange_ts=values[-1][0],
            local_received_ts=values[-1][0],
            payload={"result": [{"t": ts, "v": val} for ts, val in values]},
        )

    glass_vals = [(base_ts + i * 60 * 60 * 1000, 0.50 + 0.002 * i) for i in range(hours)]
    events.extend(
        [
            metric_event("glassnode_api", "glassnode.options_iv_7d.btc", glass_vals),
            metric_event("glassnode_api", "glassnode.options_iv_30d.btc", [(ts, val + 0.08) for ts, val in glass_vals]),
            metric_event("glassnode_api", "glassnode.options_skew_25d.btc", [(ts, 0.01 + 0.0005 * idx) for idx, (ts, _) in enumerate(glass_vals)]),
            metric_event("glassnode_api", "glassnode.options_put_call_iv_spread.btc", [(ts, 0.02 + 0.0003 * idx) for idx, (ts, _) in enumerate(glass_vals)]),
            metric_event("glassnode_api", "glassnode.exchange_balance.btc", [(ts, 1_000_000 + idx * 1000) for idx, (ts, _) in enumerate(glass_vals)]),
            metric_event("glassnode_api", "glassnode.exchange_netflow.btc", [(ts, 2000 + idx * 15) for idx, (ts, _) in enumerate(glass_vals)]),
            metric_event("glassnode_api", "glassnode.sopr.btc", [(ts, 1.0 + idx * 0.001) for idx, (ts, _) in enumerate(glass_vals)]),
            metric_event("glassnode_api", "glassnode.nrpl.btc", [(ts, 100 + idx) for idx, (ts, _) in enumerate(glass_vals)]),
            metric_event("glassnode_api", "glassnode.etf_netflow.btc", [(ts, 10 + idx * 0.5) for idx, (ts, _) in enumerate(glass_vals)]),
        ]
    )

    cq_vals = [(base_ts + i * 60 * 60 * 1000, 5000 + 10 * i) for i in range(hours)]
    events.extend(
        [
            metric_event("cryptoquant_api", "cryptoquant.reserve.btc", cq_vals),
            metric_event("cryptoquant_api", "cryptoquant.inflow.btc", [(ts, val * 0.5) for ts, val in cq_vals]),
            metric_event("cryptoquant_api", "cryptoquant.outflow.btc", [(ts, val * 0.45) for ts, val in cq_vals]),
            metric_event("cryptoquant_api", "cryptoquant.inflow_spot_exchange.btc", [(ts, val * 0.3) for ts, val in cq_vals]),
            metric_event("cryptoquant_api", "cryptoquant.inflow_derivative_exchange.btc", [(ts, val * 0.2) for ts, val in cq_vals]),
        ]
    )

    cg_vals = [(base_ts + i * 60 * 60 * 1000, 1_000_000 + 1000 * i) for i in range(hours)]
    events.extend(
        [
            metric_event("coinglass_api", "coinglass.btc_agg_oi", cg_vals),
            metric_event("coinglass_api", "coinglass.btc_agg_funding", [(ts, 0.0001 + 0.00001 * idx) for idx, (ts, _) in enumerate(cg_vals)]),
            metric_event("coinglass_api", "coinglass.btc_global_account_ratio", [(ts, 1.1 + 0.001 * idx) for idx, (ts, _) in enumerate(cg_vals)]),
        ]
    )

    events.append(
        RawEvent(
            source="macro_csv",
            topic="macro.fomc",
            symbol="MACRO",
            exchange_ts=base_ts + 28 * 60 * 60 * 1000,
            local_received_ts=base_ts + 28 * 60 * 60 * 1000,
            payload={"event_type": "fomc", "source": "fed"},
        )
    )
    return events


def test_external_context_feature_builders_apply_lags_and_macro_windows(tmp_path: Path) -> None:
    bars = _make_trending_bars(220)
    raw_dir = tmp_path / "raw"
    _write_events(raw_dir, _vendor_events(bars[0].ts - 60 * 60 * 1000, hours=48))

    deribit = build_options_deribit_features([raw_dir], bars, symbol="BTCUSDT")
    macro = build_macro_veto_features([raw_dir], bars, symbol="BTCUSDT")
    glass_opts = build_options_glassnode_features([raw_dir], bars, symbol="BTCUSDT")
    glass_on = build_onchain_glassnode_features([raw_dir], bars, symbol="BTCUSDT")
    cq = build_onchain_cryptoquant_features([raw_dir], bars, symbol="BTCUSDT")
    cg = build_aggregate_derivs_coinglass_features([raw_dir], bars, symbol="BTCUSDT")

    assert deribit and glass_opts and glass_on and cq and cg and macro
    assert deribit[-1].dvol_level is not None
    assert deribit[-1].dvol_change_1h is not None
    assert glass_opts[0].iv_7d is None  # 12h lag blocks the earliest bars
    assert glass_opts[-1].iv_7d is not None and glass_opts[-1].iv_term_slope is not None
    assert glass_on[0].exchange_balance is None  # 24h lag blocks the earliest bars
    assert glass_on[-1].exchange_balance is not None
    assert cq[-1].reserve is not None and cq[-1].inflow_derivative_exchange is not None
    assert cg[-1].btc_agg_oi is not None and cg[-1].btc_global_account_ratio is not None
    assert any(item.veto_active for item in macro)
    assert any(item.veto_event_type == "fomc" for item in macro if item.veto_active)


def test_continuation_candidates_respect_macro_veto() -> None:
    bars = _make_trending_bars()
    trend = []
    regime = []
    crowding = []
    micro = []
    macro = []
    for bar in bars:
        trend.append(TrendFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ret_15m_1=0.01, ret_1h_4=0.04, ret_4h_6=0.08, ema50_4h_gap=0.03, ema50_4h_slope=0.02, ema200_4h_slope=0.01, adx14_4h=24, atr_pctile_90d=55, breakout_age_1h=5, impulse_atr_1h=1.4, pullback_depth_frac=0.35, pullback_bars=2, dist_to_breakout_level=0.01, dist_to_ema50_4h=0.015))
        regime.append(RegimeFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", rv_1d=0.02, rv_7d=0.03, atr_pctile_90d=55, jump_intensity_1d=0.05, mark_index_gap=0.0005, premium_index_15m=0.001, premium_z_7d=0.2, stress_score=0.10, trend_score=0.70, range_score=0.15))
        crowding.append(CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", crowding_long_score=0.5, veto_long=False))
        micro.append(MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_60s=200, median_bookimb_l10_60s=0.08, spread_z=0.2, vwap_mid_dev_30s_z=0.1, gate_pass=True))
        macro.append(MacroVetoFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", mins_to_fomc=15.0, veto_active=True, veto_event_type="fomc"))

    from btc_alert_engine.strategy.bybit_candidates import build_continuation_candidates

    ungated = build_continuation_candidates(
        bars,
        trend,
        regime,
        crowding,
        micro,
        symbol="BTCUSDT",
        macro_features=macro,
        require_macro_veto=False,
    )
    gated = build_continuation_candidates(
        bars,
        trend,
        regime,
        crowding,
        micro,
        symbol="BTCUSDT",
        macro_features=macro,
        require_macro_veto=True,
    )
    assert ungated
    assert not gated


def test_static_contract_checks_are_clean() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checks = run_static_contract_checks(repo_root / "research_registry.yaml", repo_root / "feature_contracts.yaml")
    errors = [item for item in checks if item.level == "error"]
    assert not errors


def test_experiment_builder_accepts_all_optional_blocks(tmp_path: Path) -> None:
    bars = _make_trending_bars()
    derived_dir = tmp_path / "derived"
    with PartitionedNdjsonWriter(derived_dir) as writer:
        for bar in bars:
            writer.write(namespace="bars/bybit/trade_15m", symbol="BTCUSDT", ts_ms=bar.ts, record=bar)
            writer.write(namespace="features/trend_bybit", symbol="BTCUSDT", ts_ms=bar.ts, record=TrendFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ret_15m_1=0.01, ret_1h_4=0.04, ret_4h_6=0.08, ema50_4h_gap=0.03, ema50_4h_slope=0.02, ema200_4h_slope=0.01, adx14_4h=24, atr_pctile_90d=55, breakout_age_1h=5, impulse_atr_1h=1.4, pullback_depth_frac=0.35, pullback_bars=2, dist_to_breakout_level=0.01, dist_to_ema50_4h=0.015))
            writer.write(namespace="features/regime_bybit", symbol="BTCUSDT", ts_ms=bar.ts, record=RegimeFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", rv_1d=0.02, rv_7d=0.03, atr_pctile_90d=55, jump_intensity_1d=0.05, mark_index_gap=0.0005, premium_index_15m=0.001, premium_z_7d=0.2, stress_score=0.10, trend_score=0.70, range_score=0.15))
            writer.write(namespace="features/crowding_bybit", symbol="BTCUSDT", ts_ms=bar.ts, record=CrowdingFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", funding_8h=0.0002, funding_z_7d=0.5, premium_index_15m=0.001, premium_z_7d=0.2, oi_level=1_000_000, oi_change_1h=0.01, oi_change_4h=0.02, long_short_ratio_1h=1.05, liq_longs_z_1h=0.1, liq_shorts_z_1h=0.1, crowding_long_score=0.4, crowding_short_score=-0.4, veto_long=False, veto_short=False))
            writer.write(namespace="features/micro_bybit", symbol="BTCUSDT", ts_ms=bar.ts, record=MicroFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", ofi_10s=50, ofi_60s=200, ofi_300s=500, cum_delta_60s=20, spread_bps=1.0, spread_z=0.2, bookimb_l1=0.10, bookimb_l5=0.09, bookimb_l10=0.08, top10_depth_usd=2_000_000, depth_decay=0.12, vwap_mid_dev_30s=0.0002, vwap_mid_dev_30s_z=0.1, replenish_rate_30s=1.5, cancel_add_ratio_30s=0.6, micro_vol_60s=0.002, median_bookimb_l10_60s=0.08, gate_pass=True))
            writer.write(namespace="features/options_deribit", symbol="BTCUSDT", ts_ms=bar.ts, record=DeribitOptionsFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", dvol_level=72.0, dvol_z_30d=0.3, dvol_change_1h=0.02))
            writer.write(namespace="features/options_glassnode", symbol="BTCUSDT", ts_ms=bar.ts, record=GlassnodeOptionsFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", iv_7d=0.52, iv_30d=0.61, iv_term_slope=0.09, skew_25d=0.01, put_call_iv_spread=0.02))
            writer.write(namespace="features/macro_veto", symbol="BTCUSDT", ts_ms=bar.ts, record=MacroVetoFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", mins_to_fomc=999.0, mins_to_cpi=999.0, mins_to_nfp=999.0, mins_to_pce=999.0, mins_to_gdp=999.0, veto_active=False))
            writer.write(namespace="features/onchain_glassnode", symbol="BTCUSDT", ts_ms=bar.ts, record=GlassnodeOnChainFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", exchange_balance=1_000_000, exchange_netflow=1000, sopr=1.02, nrpl=120.0, etf_netflow=50.0))
            writer.write(namespace="features/onchain_cryptoquant", symbol="BTCUSDT", ts_ms=bar.ts, record=CryptoQuantOnChainFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", reserve=5000.0, inflow=2000.0, outflow=1800.0, inflow_spot_exchange=800.0, inflow_derivative_exchange=600.0))
            writer.write(namespace="features/aggregate_derivs_coinglass", symbol="BTCUSDT", ts_ms=bar.ts, record=CoinGlassAggregateFeatureSnapshot(ts=bar.ts, symbol="BTCUSDT", btc_agg_oi=1_500_000, btc_agg_funding=0.0002, btc_global_account_ratio=1.15))

    repo_root = Path(__file__).resolve().parents[1]
    registry = load_research_registry(repo_root / "research_registry.yaml")
    targets = {exp["id"]: exp for exp in registry.experiments if exp["id"] in {"A8_plus_macro_veto", "A9_plus_glassnode_onchain", "A10_plus_coinglass_overlay", "A11_cryptoquant_instead_of_glassnode"}}

    a8 = build_experiment_event_frame(derived_dir, "BTCUSDT", targets["A8_plus_macro_veto"], skip_missing=False)
    a9 = build_experiment_event_frame(derived_dir, "BTCUSDT", targets["A9_plus_glassnode_onchain"], skip_missing=False)
    a10 = build_experiment_event_frame(derived_dir, "BTCUSDT", targets["A10_plus_coinglass_overlay"], skip_missing=False)
    a11 = build_experiment_event_frame(derived_dir, "BTCUSDT", targets["A11_cryptoquant_instead_of_glassnode"], skip_missing=False)

    for dataset in [a8, a9, a10, a11]:
        assert dataset.skip_reason is None
        assert not dataset.frame.empty

    assert "options_deribit__dvol_level" in a8.frame.columns
    assert "onchain_glassnode__exchange_balance" in a9.frame.columns
    assert "aggregate_derivs_coinglass__btc_agg_oi" in a10.frame.columns
    assert "onchain_cryptoquant__reserve" in a11.frame.columns
