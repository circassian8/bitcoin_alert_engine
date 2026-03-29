#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from btc_alert_engine.config import default_reports_root, load_research_registry
from btc_alert_engine.research.execution import build_raw_execution_tape
from btc_alert_engine.research.experiments import build_experiment_event_frame, feature_columns_for_experiment
from btc_alert_engine.research.walkforward import (
    _apply_portfolio_policy,
    _portfolio_settings,
    make_final_split,
    threshold_for_budget,
    train_calibrated_model,
    _slice_split,
)


def _first_notional(registry: object, override: float | None) -> float | None:
    if override is not None:
        return float(override)
    notionals = registry.execution_costs.get("notionals_usd", [])
    if notionals:
        return float(notionals[0])
    return None


def _enrich(frame: pd.DataFrame, *, notional_usd: float | None) -> pd.DataFrame:
    out = frame.copy()
    out["ts_dt"] = pd.to_datetime(out["ts_dt"], utc=True)
    out["trade_index"] = range(1, len(out) + 1)
    out["cumulative_r"] = out["realized_r"].cumsum()
    out["equity_peak_r"] = out["cumulative_r"].cummax()
    out["drawdown_r"] = out["equity_peak_r"] - out["cumulative_r"]
    if {"executed_entry", "stop_y"}.issubset(out.columns):
        out["signal_risk"] = out["executed_entry"] - out["stop_y"]
        out["risk_pct"] = out["signal_risk"] / out["executed_entry"]
        if notional_usd is not None:
            out["notional_usd"] = float(notional_usd)
            out["risk_usd"] = out["risk_pct"] * float(notional_usd)
            out["pnl_usd"] = out["realized_r"] * out["risk_usd"]
            out["cumulative_pnl_usd"] = out["pnl_usd"].cumsum()
            out["equity_peak_pnl_usd"] = out["cumulative_pnl_usd"].cummax()
            out["drawdown_pnl_usd"] = out["equity_peak_pnl_usd"] - out["cumulative_pnl_usd"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final-split event and portfolio alert CSVs for a walk-forward experiment.")
    parser.add_argument("--data-dir", default="./data-pilot-long")
    parser.add_argument("--registry", default="./research_registry_broad_test.yaml")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--model-key", default="baseline")
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--notional-usd", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    registry = load_research_registry(args.registry)
    experiment = next((exp for exp in registry.experiments if exp["id"] == args.experiment_id), None)
    if experiment is None:
        raise SystemExit(f"unknown experiment id: {args.experiment_id}")

    derived_dir = data_root / "derived"
    raw_dir = data_root / "raw"
    raw_paths = [raw_dir] if raw_dir.exists() else None
    raw_tape = None
    if raw_paths:
        raw_tape, _ = build_raw_execution_tape(raw_paths, symbol=args.symbol, tolerate_gaps=True)

    dataset = build_experiment_event_frame(
        derived_dir,
        args.symbol,
        experiment,
        raw_paths=raw_paths,
        raw_tape=raw_tape,
        slippage_bps=args.slippage_bps,
        latency_ms=int(registry.labeling.get("latency_ms", 0)),
        skip_missing=True,
    )
    if dataset.skip_reason or dataset.frame.empty:
        raise SystemExit(f"experiment unavailable: {dataset.skip_reason or 'empty_frame'}")

    feature_columns = feature_columns_for_experiment(dataset.frame, dataset.blocks)
    final_split = make_final_split(registry.validation)
    train, calibrate, test = _slice_split(dataset.frame, final_split)
    model = train_calibrated_model(
        train,
        calibrate,
        feature_columns,
        model_key=args.model_key,
        calibration_order=registry.models.get("calibration", {}).get("order", ["isotonic", "sigmoid"]),
    )

    portfolio_mode, selection_policy, cooldown_minutes = _portfolio_settings(registry.goal)
    p_cal = model.predict_proba(calibrate)
    p_test = model.predict_proba(test)
    threshold = threshold_for_budget(
        calibrate,
        p_cal,
        budget_per_week=args.budget,
        window_start=final_split.calibrate_start,
        window_end=final_split.calibrate_end,
        portfolio_mode=portfolio_mode,
        selection_policy=selection_policy,
        cooldown_minutes=cooldown_minutes,
    )

    event_selected = test.copy()
    event_selected["p"] = p_test
    event_selected = event_selected[event_selected["p"] >= threshold].sort_values("ts").reset_index(drop=True)
    portfolio_selected = _apply_portfolio_policy(
        event_selected,
        mode=portfolio_mode,
        selection_policy=selection_policy,
        cooldown_minutes=cooldown_minutes,
    ).reset_index(drop=True)

    notional_usd = _first_notional(registry, args.notional_usd)
    common = {
        "split_kind": "final",
        "budget_per_week": args.budget,
        "threshold": threshold,
        "model_key": args.model_key,
        "portfolio_mode": portfolio_mode,
        "selection_policy": selection_policy,
        "cooldown_minutes": cooldown_minutes,
    }
    output_dir = Path(args.output_dir) if args.output_dir else default_reports_root(data_root) / "walkforward_broad_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    event_export = _enrich(event_selected.assign(selection_scope="event", **common), notional_usd=notional_usd)
    portfolio_export = _enrich(portfolio_selected.assign(selection_scope="portfolio", **common), notional_usd=notional_usd)

    event_path = output_dir / f"{args.experiment_id}_event_final_alerts.csv"
    portfolio_path = output_dir / f"{args.experiment_id}_portfolio_final_alerts.csv"
    event_export.to_csv(event_path, index=False)
    portfolio_export.to_csv(portfolio_path, index=False)

    print(event_path)
    print(portfolio_path)


if __name__ == "__main__":
    main()
