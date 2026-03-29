from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from btc_alert_engine.research.experiments import ExperimentDataset
from btc_alert_engine.research.walkforward import _build_estimator, run_walkforward_experiments


def _synthetic_event_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2021-01-01", periods=180, freq="D", tz="UTC")
    trend_edge = rng.normal(0, 1, len(ts))
    regime_score = 0.8 * trend_edge + rng.normal(0, 0.25, len(ts))
    micro_flow = 0.4 * trend_edge + rng.normal(0, 0.5, len(ts))
    latent = 0.25 * trend_edge + 0.95 * regime_score + 0.15 * micro_flow + rng.normal(0, 0.45, len(ts))
    y = (latent > 0.0).astype(int)
    target_r_multiple = np.where(np.arange(len(ts)) % 5 == 0, 1.5, 2.0)
    realized = np.where(y == 1, target_r_multiple, -1.0)
    timeout_idx = np.arange(len(ts)) % 11 == 0
    realized = realized.astype(float)
    realized[timeout_idx] = np.where(y[timeout_idx] == 1, 0.6, -0.3)
    outcomes = np.where(y == 1, "tp", "sl").astype(object)
    outcomes[timeout_idx] = "timeout"

    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(len(ts))],
            "ts": (ts.view("int64") // 1_000_000).astype(int),
            "ts_dt": ts,
            "module": "continuation_v1",
            "symbol": "BTCUSDT",
            "side": "long",
            "entry": 20000 + np.arange(len(ts)),
            "stop": 19900 + np.arange(len(ts)),
            "tp": 20200 + np.arange(len(ts)),
            "timeout_bars": 96,
            "signal_entry": 20000 + np.arange(len(ts)),
            "executed_entry": 20001 + np.arange(len(ts)),
            "executed_tp": 20201 + np.arange(len(ts)),
            "target_r_multiple": target_r_multiple,
            "tp_before_sl_within_horizon": y.astype(bool),
            "tp1_before_sl_within_12h": (latent > -0.3).astype(bool),
            "mfe_r_24h": np.where(y == 1, 2.3, 0.6),
            "mae_r_24h": np.where(y == 1, 0.4, 1.1),
            "net_r_24h_timeout": realized,
            "minutes_to_tp_or_sl": np.where(outcomes == "timeout", np.nan, 240),
            "outcome": outcomes,
            "y": y,
            "realized_r": realized,
            "holding_minutes": np.where(outcomes == "timeout", 1440, 240),
            "trend_bybit__ret_4h_6": trend_edge,
            "trend_bybit__ema50_4h_gap": 0.02 + 0.01 * trend_edge,
            "regime_bybit__trend_score": regime_score,
            "regime_bybit__stress_score": 1.0 - (regime_score - regime_score.min()) / (regime_score.max() - regime_score.min() + 1e-6),
            "micro_bybit_score__ofi_60s": micro_flow,
            "candidate__atr15": np.abs(trend_edge) + 0.5,
        }
    )
    return frame


def _write_registry(path: Path) -> None:
    payload = {
        "program": {"id": "wf_test", "version": 1},
        "execution": {"venue": "bybit", "symbol": "BTCUSDT", "category": "linear", "mode": "alert", "market_truth": "bybit_only"},
        "goal": {"primary_metric": "expectancy_r_per_alert", "secondary_metric": "worst_20_trade_drawdown_r", "alert_budgets_per_week": [1, 3], "production_budget_per_week": 3},
        "timeframes": {"micro": "1s", "trigger": "15m", "setup": "1h", "regime": "4h"},
        "data_sources": {},
        "pit_policy": {},
        "candidate_generators": {},
        "labeling": {},
        "models": {"baseline": {"type": "logistic_regression"}, "calibration": {"order": ["isotonic", "sigmoid"]}},
        "experiments": [
            {"id": "A0_native_continuation", "generator": "continuation_v1", "blocks": ["trend_bybit"]},
            {"id": "A1_native_plus_soft_regime", "generator": "continuation_v1", "blocks": ["trend_bybit", "regime_bybit"]},
            {"id": "A4_native_plus_micro_score", "generator": "continuation_v1", "blocks": ["trend_bybit", "regime_bybit", "micro_bybit_score"]},
        ],
        "validation": {
            "warmup_only": {"start": "2021-01-01", "end": "2021-01-10"},
            "outer_walk_forward": {"train_days": 40, "calibrate_days": 15, "test_days": 15, "purge_hours": 0, "embargo_hours": 0},
            "final_frozen_test": {
                "freeze_date": "2021-04-30",
                "calibrate_window": {"start": "2021-05-01", "end": "2021-05-15"},
                "untouched_test": {"start": "2021-05-16", "end": "2021-06-29"},
            },
        },
        "execution_costs": {},
        "promotion_rules": {"primary_budget_per_week": 3, "improve_parent_expectancy_by_at_least_r": 0.01, "near_equal_expectancy_tolerance_r": 0.02, "drawdown_improvement_required_percent": 5, "positive_sign_required_outer_folds_percent": 50},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_run_walkforward_experiments_with_synthetic_dataset(tmp_path: Path, monkeypatch) -> None:
    frame = _synthetic_event_frame()
    registry_path = tmp_path / "registry.yaml"
    _write_registry(registry_path)

    def fake_builder(derived_dir: Path, symbol: str, experiment: dict, *, slippage_bps: float = 1.0, skip_missing: bool = True, **_: object):
        blocks = list(experiment["blocks"])
        keep_cols = {
            "candidate_id",
            "ts",
            "ts_dt",
            "module",
            "symbol",
            "side",
            "entry",
            "stop",
            "tp",
            "timeout_bars",
            "signal_entry",
            "executed_entry",
            "executed_tp",
            "target_r_multiple",
            "tp_before_sl_within_horizon",
            "tp1_before_sl_within_12h",
            "mfe_r_24h",
            "mae_r_24h",
            "net_r_24h_timeout",
            "minutes_to_tp_or_sl",
            "outcome",
            "y",
            "realized_r",
            "holding_minutes",
            "candidate__atr15",
        }
        for block in blocks:
            prefix = f"{block}__"
            keep_cols.update({col for col in frame.columns if col.startswith(prefix)})
        subset = frame[[col for col in frame.columns if col in keep_cols]].copy()
        return ExperimentDataset(
            experiment_id=str(experiment["id"]),
            generator=str(experiment["generator"]),
            blocks=blocks,
            frame=subset,
            candidates=[],
            labels=pd.DataFrame(),
            skip_reason=None,
        )

    import btc_alert_engine.research.walkforward as walkforward

    monkeypatch.setattr(walkforward, "build_experiment_event_frame", fake_builder)

    reports_dir = run_walkforward_experiments(
        data_dir=tmp_path,
        registry_path=registry_path,
        symbol="BTCUSDT",
        output_dir=tmp_path / "reports",
        model_keys=["baseline"],
        skip_missing=True,
    )

    assert (reports_dir / "manifest.json").exists()
    assert (reports_dir / "fold_metrics.csv").exists()
    assert (reports_dir / "summary_metrics.csv").exists()
    assert (reports_dir / "promotion_decisions.csv").exists()
    assert (reports_dir / "promotion_report.md").exists()

    summary = pd.read_csv(reports_dir / "summary_metrics.csv")
    assert {"A0_native_continuation", "A1_native_plus_soft_regime", "A4_native_plus_micro_score"}.issubset(set(summary["experiment_id"]))
    assert {"outer", "final"}.issubset(set(summary["split_kind"]))
    final_budget3 = summary[(summary["split_kind"] == "final") & (summary["budget_per_week"] == 3)]
    assert not final_budget3.empty


def test_baseline_estimator_uses_liblinear_solver() -> None:
    estimator = _build_estimator("baseline")
    assert estimator.named_steps["model"].solver == "liblinear"
