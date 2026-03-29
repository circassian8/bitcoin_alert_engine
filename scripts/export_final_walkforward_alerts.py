#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from btc_alert_engine.config import default_reports_root, load_research_registry
from btc_alert_engine.research.execution import build_raw_execution_tape
from btc_alert_engine.research.experiments import build_experiment_event_frame, feature_columns_for_experiment
from btc_alert_engine.research.walkforward import (
    _apply_portfolio_policy,
    _portfolio_settings,
    _slice_split,
    make_final_split,
    threshold_for_budget,
    train_calibrated_model,
)


def _primary_budget(registry: object) -> int:
    promotion_rules = getattr(registry, "promotion_rules", {}) or {}
    if hasattr(promotion_rules, "get") and "primary_budget_per_week" in promotion_rules:
        return int(promotion_rules["primary_budget_per_week"])
    goal = getattr(registry, "goal", {}) or {}
    if hasattr(goal, "get"):
        return int(goal.get("production_budget_per_week", 3))
    return 3


def _first_notional(registry: object, override: float | None) -> float | None:
    if override is not None:
        return float(override)
    execution_costs = getattr(registry, "execution_costs", {}) or {}
    if hasattr(execution_costs, "get"):
        notionals = execution_costs.get("notionals_usd", [])
        if notionals:
            return float(notionals[0])
    return None


def _load_selected_models(reports_dir: Path | None) -> dict[str, str | None]:
    if reports_dir is None:
        return {}
    manifest_path = reports_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected: dict[str, str | None] = {}
    for item in payload.get("experiments", []):
        experiment_id = str(item.get("experiment_id"))
        selected[experiment_id] = item.get("selected_model")
    return selected


def _write_alert_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _side_summary(frame: pd.DataFrame, *, experiment_id: str, selection_kind: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        side_frame = frame[frame["side"] == side].copy()
        rows.append(
            {
                "experiment_id": experiment_id,
                "selection_kind": selection_kind,
                "side": side,
                "n_alerts": int(len(side_frame)),
                "precision_tp_before_sl": None if side_frame.empty else float(side_frame["y"].mean()),
                "expectancy_r_per_alert": None if side_frame.empty else float(side_frame["realized_r"].mean()),
                "hit_rate": None if side_frame.empty else float((side_frame["outcome"] == "tp").mean()),
                "mean_probability": None if side_frame.empty else float(side_frame["p"].mean()),
                "median_holding_minutes": None if side_frame.empty else float(side_frame["holding_minutes"].median()),
            }
        )
    return rows


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
    parser = argparse.ArgumentParser(description="Export final-split alerts for a walk-forward report directory.")
    parser.add_argument("--data-dir", default="./data-pilot-long")
    parser.add_argument("--registry", default="./research_registry_broad_test.yaml")
    parser.add_argument("--reports-dir", default=None, help="Walk-forward report directory containing manifest.json.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--experiment-id", default=None, help="Optional single experiment id to export.")
    parser.add_argument("--experiments", nargs="+", default=None, help="Optional experiment ids to export.")
    parser.add_argument("--budget", type=int, default=None, help="Budget to export. Defaults to the registry primary budget.")
    parser.add_argument("--model-key", default=None, help="Optional model override. Defaults to the selected model in manifest.json.")
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--notional-usd", type=float, default=None)
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to reports-dir when provided.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_dir)
    registry = load_research_registry(args.registry)
    reports_dir = Path(args.reports_dir) if args.reports_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else (reports_dir or default_reports_root(data_root) / "walkforward")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_ids: set[str] = set()
    if args.experiment_id:
        requested_ids.add(str(args.experiment_id))
    if args.experiments:
        requested_ids.update(str(item) for item in args.experiments)

    selected_models = _load_selected_models(reports_dir)
    budget = args.budget if args.budget is not None else _primary_budget(registry)
    calibration_order = registry.models.get("calibration", {}).get("order", ["isotonic", "sigmoid"])
    portfolio_mode, selection_policy, cooldown_minutes = _portfolio_settings(registry.goal)
    final_split = make_final_split(registry.validation)
    latency_ms = int(registry.labeling.get("latency_ms", 0))
    notional_usd = _first_notional(registry, args.notional_usd)

    derived_dir = data_root / "derived"
    raw_dir = data_root / "raw"
    raw_paths = [raw_dir] if raw_dir.exists() else None
    raw_tape = None
    if raw_paths:
        raw_tape, _ = build_raw_execution_tape(raw_paths, symbol=args.symbol, tolerate_gaps=True)

    selected_experiments = [
        experiment
        for experiment in registry.experiments
        if not requested_ids or str(experiment["id"]) in requested_ids
    ]
    if not selected_experiments:
        raise SystemExit("no matching experiments")

    summary_rows: list[dict[str, object]] = []
    for experiment in selected_experiments:
        experiment_id = str(experiment["id"])
        model_key = args.model_key or selected_models.get(experiment_id)
        if not model_key:
            print(f"skip {experiment_id}: no selected model")
            continue

        dataset = build_experiment_event_frame(
            derived_dir,
            args.symbol,
            experiment,
            raw_paths=raw_paths,
            raw_tape=raw_tape,
            slippage_bps=args.slippage_bps,
            latency_ms=latency_ms,
            skip_missing=True,
        )
        if dataset.skip_reason or dataset.frame.empty:
            print(f"skip {experiment_id}: {dataset.skip_reason or 'empty_frame'}")
            continue

        feature_columns = feature_columns_for_experiment(dataset.frame, dataset.blocks)
        train, calibrate, test = _slice_split(dataset.frame, final_split)
        if len(train) < 20 or len(calibrate) < 5 or len(test) < 5:
            print(f"skip {experiment_id}: insufficient_final_split_rows")
            continue

        model = train_calibrated_model(
            train,
            calibrate,
            feature_columns,
            model_key=model_key,
            calibration_order=calibration_order,
        )
        p_cal = model.predict_proba(calibrate)
        p_test = model.predict_proba(test)
        threshold = threshold_for_budget(
            calibrate,
            p_cal,
            budget_per_week=budget,
            window_start=final_split.calibrate_start,
            window_end=final_split.calibrate_end,
            portfolio_mode=portfolio_mode,
            selection_policy=selection_policy,
            cooldown_minutes=cooldown_minutes,
        )

        final_frame = test.copy()
        final_frame["p"] = p_test
        final_frame["split_kind"] = "final"
        final_frame["budget_per_week"] = budget
        final_frame["threshold"] = threshold
        final_frame["model_key"] = model_key
        final_frame["portfolio_mode"] = portfolio_mode
        final_frame["selection_policy"] = selection_policy
        final_frame["cooldown_minutes"] = cooldown_minutes

        event_selected = final_frame[final_frame["p"] >= threshold].sort_values("ts").reset_index(drop=True)
        portfolio_selected = _apply_portfolio_policy(
            event_selected,
            mode=portfolio_mode,
            selection_policy=selection_policy,
            cooldown_minutes=cooldown_minutes,
        ).reset_index(drop=True)

        export_sets = {
            "event": _enrich(event_selected.assign(selection_scope="event"), notional_usd=notional_usd),
            "portfolio": _enrich(portfolio_selected.assign(selection_scope="portfolio"), notional_usd=notional_usd),
        }
        for selection_kind, frame in export_sets.items():
            canonical = output_dir / f"{experiment_id}_final_{selection_kind}_alerts.csv"
            compat = output_dir / f"{experiment_id}_{selection_kind}_final_alerts.csv"
            _write_alert_csv(canonical, frame)
            _write_alert_csv(compat, frame)
            if selection_kind == "event":
                _write_alert_csv(output_dir / f"{experiment_id}_final_alerts.csv", frame)
            for side in ("long", "short"):
                side_frame = frame[frame["side"] == side].reset_index(drop=True)
                _write_alert_csv(output_dir / f"{experiment_id}_final_{selection_kind}_alerts_{side}.csv", side_frame)
            summary_rows.extend(_side_summary(frame, experiment_id=experiment_id, selection_kind=selection_kind))

        print(
            "exported",
            experiment_id,
            f"event={len(event_selected)}",
            f"portfolio={len(portfolio_selected)}",
            f"threshold={threshold:.6f}",
            f"model={model_key}",
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_path = output_dir / "final_alerts_side_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(summary_path)


if __name__ == "__main__":
    main()
