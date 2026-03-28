from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from btc_alert_engine.research.experiments import ExperimentDataset
from btc_alert_engine.research.walkforward import run_walkforward_experiments


def _write_registry(path: Path) -> None:
    payload = {
        "program": {"id": "wf_empty_test", "version": 1},
        "execution": {"venue": "bybit", "symbol": "BTCUSDT", "category": "linear", "mode": "alert", "market_truth": "bybit_only"},
        "goal": {"primary_metric": "expectancy_r_per_alert", "secondary_metric": "worst_20_trade_drawdown_r", "alert_budgets_per_week": [1, 3], "production_budget_per_week": 3},
        "timeframes": {"micro": "1s", "trigger": "15m", "setup": "1h", "regime": "4h"},
        "data_sources": {},
        "pit_policy": {},
        "candidate_generators": {},
        "labeling": {},
        "models": {"baseline": {"type": "logistic_regression"}, "calibration": {"order": ["isotonic", "sigmoid"]}},
        "experiments": [{"id": "A0_native_continuation", "generator": "continuation_v1", "blocks": ["trend_bybit"]}],
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
        "promotion_rules": {"primary_budget_per_week": 3},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_run_walkforward_experiments_handles_no_evaluable_results(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.yaml"
    _write_registry(registry_path)

    def fake_builder(*args, **kwargs):
        ts = pd.date_range("2021-01-01", periods=10, freq="D", tz="UTC")
        frame = pd.DataFrame(
            {
                "candidate_id": [f"c{i}" for i in range(len(ts))],
                "ts": (ts.view("int64") // 1_000_000).astype(int),
                "ts_dt": ts,
                "module": "continuation_v1",
                "symbol": "BTCUSDT",
                "side": "long",
                "y": [0, 1] * 5,
                "realized_r": [-1.0, 2.0] * 5,
                "outcome": ["sl", "tp"] * 5,
                "holding_minutes": [240.0] * len(ts),
                "trend_bybit__ema50_4h_gap": [0.1] * len(ts),
            }
        )
        return ExperimentDataset(
            experiment_id="A0_native_continuation",
            generator="continuation_v1",
            blocks=["trend_bybit"],
            frame=frame,
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
    assert summary.empty
