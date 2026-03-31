from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from btc_alert_engine.research.experiments import ExperimentDataset, _to_label_frame, build_experiment_event_frame
from btc_alert_engine.research.walkforward import (
    _apply_portfolio_policy,
    _prune_feature_columns,
    evaluate_predictions,
    threshold_for_budget,
    SplitSpec,
)
from btc_alert_engine.schemas import CandidateLabel, PriceBar, RegimeFeatureSnapshot, TrendFeatureSnapshot
from btc_alert_engine.storage.partitioned_ndjson import PartitionedNdjsonWriter
from btc_alert_engine.verification.project import run_static_contract_checks


def test_prune_feature_columns_drops_all_null_and_constant() -> None:
    frame = pd.DataFrame(
        {
            "all_null": [np.nan, np.nan, np.nan],
            "constant": [1.0, 1.0, 1.0],
            "useful": [1.0, 2.0, 3.0],
        }
    )
    kept, dropped = _prune_feature_columns(frame, ["all_null", "constant", "useful"])
    assert kept == ["useful"]
    assert dropped == ["all_null", "constant"]


def test_threshold_for_budget_uses_split_duration_not_candidate_span() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "ts": (ts.view("int64") // 1_000_000).astype(int),
            "ts_dt": ts,
            "entry_ts": (ts.view("int64") // 1_000_000).astype(int),
            "exit_ts": ((ts + pd.Timedelta(minutes=30)).view("int64") // 1_000_000).astype(int),
        }
    )
    probabilities = np.array([0.9, 0.8, 0.7])
    threshold = threshold_for_budget(
        frame,
        probabilities,
        budget_per_week=1,
        window_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        window_end=pd.Timestamp("2026-01-15T00:00:00Z"),
        portfolio_mode="one_position_at_a_time",
        selection_policy="first_signal",
        cooldown_minutes=0,
    )
    # Two-week window at 1 alert/week should target the top 2 alerts, not collapse to 1
    assert threshold == 0.8


def test_evaluate_predictions_reports_event_and_portfolio_metrics() -> None:
    ts = pd.to_datetime(
        [
            "2026-02-01T00:00:00Z",
            "2026-02-01T00:10:00Z",
            "2026-02-01T02:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "ts": (ts.view("int64") // 1_000_000).astype(int),
            "ts_dt": ts,
            "entry_ts": (ts.view("int64") // 1_000_000).astype(int),
            "exit_ts": [
                int((ts[0] + pd.Timedelta(minutes=90)).timestamp() * 1000),
                int((ts[1] + pd.Timedelta(minutes=40)).timestamp() * 1000),
                int((ts[2] + pd.Timedelta(minutes=30)).timestamp() * 1000),
            ],
            "y": [1, 0, 1],
            "realized_r": [2.0, -1.0, 2.0],
            "outcome": ["tp", "sl", "tp"],
            "holding_minutes": [90.0, 40.0, 30.0],
        }
    )
    split = SplitSpec(
        split_kind="final",
        fold_id="final_frozen",
        train_start=pd.Timestamp("2025-01-01T00:00:00Z"),
        train_end=pd.Timestamp("2025-06-01T00:00:00Z"),
        calibrate_start=pd.Timestamp("2025-06-01T00:00:00Z"),
        calibrate_end=pd.Timestamp("2025-07-01T00:00:00Z"),
        test_start=pd.Timestamp("2026-02-01T00:00:00Z"),
        test_end=pd.Timestamp("2026-02-08T00:00:00Z"),
    )
    result = evaluate_predictions(
        frame,
        np.array([0.9, 0.85, 0.8]),
        threshold=0.8,
        budget_per_week=3,
        experiment_id="A0",
        generator="continuation_v1",
        model_key="baseline",
        split=split,
        portfolio_mode="one_position_at_a_time",
        selection_policy="first_signal",
        cooldown_minutes=0,
    )
    assert result.event_n_alerts == 3
    assert result.n_alerts == 2
    assert result.overlap_filtered_alerts == 1
    assert result.event_max_concurrent_positions == 2
    assert result.event_expectancy_r_per_alert == 1.0
    assert result.expectancy_r_per_alert == 2.0


def test_to_label_frame_deducts_round_trip_fees_from_realized_r() -> None:
    label = CandidateLabel(
        candidate_id="short_tp",
        ts=int(pd.Timestamp("2026-02-01T00:00:00Z").timestamp() * 1000),
        module="continuation_v1_fast",
        symbol="BTCUSDT",
        side="short",
        signal_entry=100.0,
        signal_tp=98.75,
        entry_ts=int(pd.Timestamp("2026-02-01T00:00:00Z").timestamp() * 1000),
        executed_entry=100.0,
        stop=101.0,
        executed_tp=98.75,
        exit_ts=int(pd.Timestamp("2026-02-01T00:05:00Z").timestamp() * 1000),
        executed_exit=98.75,
        target_r_multiple=1.25,
        timeout_bars=288,
        tp_before_sl_within_horizon=True,
        tp1_before_sl_within_12h=True,
        mfe_r_24h=1.25,
        mae_r_24h=0.0,
        net_r_24h_timeout=1.25,
        minutes_to_tp_or_sl=5,
        outcome="tp",
    )

    frame = _to_label_frame([label], fee_bps_per_side=5.0)

    expected_fee_r = ((100.0 + 98.75) * 0.0005) / 1.0
    assert frame.loc[0, "gross_realized_r"] == pytest.approx(1.25)
    assert frame.loc[0, "fee_r"] == pytest.approx(expected_fee_r)
    assert frame.loc[0, "realized_r"] == pytest.approx(1.25 - expected_fee_r)


def test_apply_portfolio_policy_highest_probability_keeps_later_nonoverlapping_trades() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "ts": [0, 10, 30, 50],
            "entry_ts": [0, 10, 30, 50],
            "exit_ts": [100, 20, 40, 60],
            "p": [0.1, 0.9, 0.8, 0.7],
        }
    )
    selected = _apply_portfolio_policy(
        frame,
        mode="one_position_at_a_time",
        selection_policy="highest_probability",
        cooldown_minutes=0,
    )
    assert list(selected["candidate_id"]) == ["b", "c", "d"]


def test_evaluate_predictions_event_level_mode_preserves_all_selected_alerts() -> None:
    ts = pd.to_datetime(
        [
            "2026-02-01T00:00:00Z",
            "2026-02-01T00:10:00Z",
            "2026-02-01T02:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "ts": (ts.view("int64") // 1_000_000).astype(int),
            "ts_dt": ts,
            "entry_ts": (ts.view("int64") // 1_000_000).astype(int),
            "exit_ts": [
                int((ts[0] + pd.Timedelta(minutes=90)).timestamp() * 1000),
                int((ts[1] + pd.Timedelta(minutes=40)).timestamp() * 1000),
                int((ts[2] + pd.Timedelta(minutes=30)).timestamp() * 1000),
            ],
            "y": [1, 0, 1],
            "realized_r": [2.0, -1.0, 2.0],
            "outcome": ["tp", "sl", "tp"],
            "holding_minutes": [90.0, 40.0, 30.0],
        }
    )
    split = SplitSpec(
        split_kind="final",
        fold_id="final_frozen",
        train_start=pd.Timestamp("2025-01-01T00:00:00Z"),
        train_end=pd.Timestamp("2025-06-01T00:00:00Z"),
        calibrate_start=pd.Timestamp("2025-06-01T00:00:00Z"),
        calibrate_end=pd.Timestamp("2025-07-01T00:00:00Z"),
        test_start=pd.Timestamp("2026-02-01T00:00:00Z"),
        test_end=pd.Timestamp("2026-02-08T00:00:00Z"),
    )
    result = evaluate_predictions(
        frame,
        np.array([0.9, 0.85, 0.8]),
        threshold=0.8,
        budget_per_week=3,
        experiment_id="A0",
        generator="continuation_v1",
        model_key="baseline",
        split=split,
        portfolio_mode="event_level",
        selection_policy="first_signal",
        cooldown_minutes=0,
    )
    assert result.event_n_alerts == 3
    assert result.n_alerts == 3
    assert result.overlap_filtered_alerts == 0
    assert result.event_expectancy_r_per_alert == result.expectancy_r_per_alert == 1.0


def test_static_contract_checks_warn_on_short_purge_and_embargo(tmp_path: Path) -> None:
    registry = {
        "program": {"id": "wf_checks", "version": 1},
        "execution": {"venue": "bybit", "symbol": "BTCUSDT", "category": "linear", "mode": "alert", "market_truth": "bybit_only"},
        "goal": {"primary_metric": "expectancy_r_per_alert", "production_budget_per_week": 3},
        "timeframes": {"micro": "1s", "trigger": "15m", "setup": "1h", "regime": "4h"},
        "data_sources": {},
        "pit_policy": {},
        "candidate_generators": {},
        "labeling": {"timeout_15m_bars_by_module": {"continuation_v1": 96}},
        "models": {},
        "experiments": [{"id": "A0", "generator": "continuation_v1", "blocks": ["trend_bybit"]}],
        "validation": {
            "warmup_only": {"start": "2025-01-01", "end": "2025-01-31"},
            "outer_walk_forward": {"train_days": 30, "calibrate_days": 10, "test_days": 10, "purge_hours": 0, "embargo_hours": 0},
            "final_frozen_test": {
                "freeze_date": "2025-06-01",
                "calibrate_window": {"start": "2025-06-02", "end": "2025-06-12"},
                "untouched_test": {"start": "2025-06-13", "end": "2025-06-23"},
            },
        },
        "execution_costs": {},
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    checks = run_static_contract_checks(registry_path, repo_root / "feature_contracts.yaml")
    codes = {item.code for item in checks}
    assert "purge_shorter_than_label_horizon" in codes
    assert "embargo_shorter_than_label_horizon" in codes


def test_build_experiment_event_frame_isolates_nonrequested_candidate_features(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    bars = []
    with PartitionedNdjsonWriter(tmp_path / "derived") as writer:
        for i in range(160):
            ts = base_ts + (i + 1) * 900_000
            bar = PriceBar(
                ts=ts,
                symbol="BTCUSDT",
                interval="15",
                open=100 + i * 0.4,
                high=101 + i * 0.4,
                low=99.7 + i * 0.4,
                close=100.8 + i * 0.4,
                volume=10,
                turnover=1000,
            )
            bars.append(bar)
            writer.write(namespace="bars/bybit/trade_15m", symbol="BTCUSDT", ts_ms=bar.ts, record=bar)
            writer.write(
                namespace="features/trend_bybit",
                symbol="BTCUSDT",
                ts_ms=bar.ts,
                record=TrendFeatureSnapshot(
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
                ),
            )
            writer.write(
                namespace="features/regime_bybit",
                symbol="BTCUSDT",
                ts_ms=bar.ts,
                record=RegimeFeatureSnapshot(
                    ts=bar.ts,
                    symbol="BTCUSDT",
                    rv_1d=0.02,
                    rv_7d=0.03,
                    atr_pctile_90d=55,
                    jump_intensity_1d=0.05,
                    mark_index_gap=0.0005,
                    premium_index_15m=0.001,
                    premium_z_7d=0.2,
                    stress_score=0.1,
                    trend_score=0.7,
                    range_score=0.15,
                ),
            )
    dataset = build_experiment_event_frame(
        tmp_path / "derived",
        "BTCUSDT",
        {"id": "S0_native_continuation", "generator": "continuation_v1", "blocks": ["trend_bybit"]},
        skip_missing=False,
    )
    assert dataset.skip_reason is None
    assert "candidate__atr15" in dataset.frame.columns
    assert "candidate__trend_score" not in dataset.frame.columns
    assert "candidate__stress_score" not in dataset.frame.columns
    assert "candidate__crowding_long_score" not in dataset.frame.columns
    assert "candidate__ofi_60s" not in dataset.frame.columns
    assert not any(col.startswith("regime_bybit__") for col in dataset.frame.columns)


def test_build_experiment_event_frame_applies_stop_width_multiplier(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    with PartitionedNdjsonWriter(tmp_path / "derived") as writer:
        for i in range(160):
            ts = base_ts + (i + 1) * 900_000
            bar = PriceBar(
                ts=ts,
                symbol="BTCUSDT",
                interval="15",
                open=100 + i * 0.4,
                high=101 + i * 0.4,
                low=99.7 + i * 0.4,
                close=100.8 + i * 0.4,
                volume=10,
                turnover=1000,
            )
            writer.write(namespace="bars/bybit/trade_15m", symbol="BTCUSDT", ts_ms=bar.ts, record=bar)
            writer.write(
                namespace="features/trend_bybit",
                symbol="BTCUSDT",
                ts_ms=bar.ts,
                record=TrendFeatureSnapshot(
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
                ),
            )

    baseline = build_experiment_event_frame(
        tmp_path / "derived",
        "BTCUSDT",
        {"id": "S0_native_continuation", "generator": "continuation_v1", "blocks": ["trend_bybit"]},
        skip_missing=False,
    )
    wider = build_experiment_event_frame(
        tmp_path / "derived",
        "BTCUSDT",
        {
            "id": "S0_native_continuation_stop150",
            "generator": "continuation_v1",
            "blocks": ["trend_bybit"],
            "generator_params": {"stop_width_multiplier": 1.5},
        },
        skip_missing=False,
    )

    assert baseline.skip_reason is None
    assert wider.skip_reason is None
    baseline_first = baseline.frame.iloc[0]
    wider_first = wider.frame.iloc[0]
    base_risk = baseline_first["candidate__signal_risk"]
    wider_risk = wider_first["candidate__signal_risk"]
    assert wider_risk == pytest.approx(base_risk * 1.5)
    assert wider_first["stop_x"] < baseline_first["stop_x"]


def test_build_experiment_event_frame_applies_target_r_multiple(tmp_path: Path) -> None:
    base_ts = 1_700_000_000_000
    with PartitionedNdjsonWriter(tmp_path / "derived") as writer:
        for i in range(160):
            ts = base_ts + (i + 1) * 900_000
            bar = PriceBar(
                ts=ts,
                symbol="BTCUSDT",
                interval="15",
                open=100 + i * 0.4,
                high=101 + i * 0.4,
                low=99.7 + i * 0.4,
                close=100.8 + i * 0.4,
                volume=10,
                turnover=1000,
            )
            writer.write(namespace="bars/bybit/trade_15m", symbol="BTCUSDT", ts_ms=bar.ts, record=bar)
            writer.write(
                namespace="features/trend_bybit",
                symbol="BTCUSDT",
                ts_ms=bar.ts,
                record=TrendFeatureSnapshot(
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
                ),
            )

    baseline = build_experiment_event_frame(
        tmp_path / "derived",
        "BTCUSDT",
        {"id": "S0_native_continuation", "generator": "continuation_v1", "blocks": ["trend_bybit"]},
        skip_missing=False,
    )
    tighter_target = build_experiment_event_frame(
        tmp_path / "derived",
        "BTCUSDT",
        {
            "id": "S0_native_continuation_tp125",
            "generator": "continuation_v1",
            "blocks": ["trend_bybit"],
            "generator_params": {"target_r_multiple": 1.25},
        },
        skip_missing=False,
    )

    assert baseline.skip_reason is None
    assert tighter_target.skip_reason is None
    baseline_first = baseline.frame.iloc[0]
    target_first = tighter_target.frame.iloc[0]
    base_risk = baseline_first["candidate__signal_risk"]
    assert target_first["target_r_multiple_x"] == pytest.approx(1.25)
    assert target_first["tp"] == pytest.approx(target_first["entry"] + 1.25 * base_risk)
    assert target_first["tp"] < baseline_first["tp"]
