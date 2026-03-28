#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from btc_alert_engine.config import load_research_registry
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
    if "primary_budget_per_week" in promotion_rules:
        return int(promotion_rules["primary_budget_per_week"])
    goal = getattr(registry, "goal")
    return int(goal.get("production_budget_per_week", 3))


def _load_selected_models(reports_dir: Path) -> dict[str, str | None]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final-split alerts for a walk-forward report directory.")
    parser.add_argument("--data-dir", required=True, help="Data directory used for the walk-forward run.")
    parser.add_argument("--registry", required=True, help="Registry used for the walk-forward run.")
    parser.add_argument("--reports-dir", required=True, help="Walk-forward report directory containing manifest.json.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--experiments", nargs="+", default=None, help="Optional experiment ids to export.")
    parser.add_argument("--budget-per-week", type=int, default=None, help="Budget to export. Defaults to the registry primary budget.")
    parser.add_argument("--model-key", default=None, help="Optional model override. Defaults to the selected model in manifest.json.")
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    registry = load_research_registry(args.registry)
    reports_dir = Path(args.reports_dir)
    derived_dir = data_dir / "derived"
    raw_dir = data_dir / "raw"
    raw_paths = [raw_dir] if raw_dir.exists() else None
    raw_tape = None
    if raw_paths:
        raw_tape, _ = build_raw_execution_tape(raw_paths, symbol=args.symbol, tolerate_gaps=True)

    selected_models = _load_selected_models(reports_dir)
    budget_per_week = args.budget_per_week if args.budget_per_week is not None else _primary_budget(registry)
    calibration_order = registry.models.get("calibration", {}).get("order", ["isotonic", "sigmoid"])
    selection_policy, cooldown_minutes = _portfolio_settings(registry.goal)
    final_split = make_final_split(registry.validation)
    latency_ms = int(registry.labeling.get("latency_ms", 0))

    summary_rows: list[dict[str, object]] = []
    selected_experiments = [
        experiment
        for experiment in registry.experiments
        if args.experiments is None or str(experiment["id"]) in set(args.experiments)
    ]

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
            budget_per_week=budget_per_week,
            window_start=final_split.calibrate_start,
            window_end=final_split.calibrate_end,
            selection_policy=selection_policy,
            cooldown_minutes=cooldown_minutes,
        )

        final_frame = test.copy()
        final_frame["p"] = p_test
        final_frame["split_kind"] = "final"
        final_frame["budget_per_week"] = budget_per_week
        final_frame["threshold"] = threshold
        final_frame["model_key"] = model_key

        event_selected = final_frame[final_frame["p"] >= threshold].sort_values("ts").reset_index(drop=True)
        portfolio_selected = _apply_portfolio_policy(
            event_selected,
            selection_policy=selection_policy,
            cooldown_minutes=cooldown_minutes,
        ).reset_index(drop=True)

        export_sets = {
            "event": event_selected,
            "portfolio": portfolio_selected,
        }
        for selection_kind, frame in export_sets.items():
            csv_path = reports_dir / f"{experiment_id}_final_{selection_kind}_alerts.csv"
            _write_alert_csv(csv_path, frame)

            # Backward-compatible alias for the historical event-level export name.
            if selection_kind == "event":
                _write_alert_csv(reports_dir / f"{experiment_id}_final_alerts.csv", frame)

            for side in ("long", "short"):
                side_frame = frame[frame["side"] == side].reset_index(drop=True)
                _write_alert_csv(reports_dir / f"{experiment_id}_final_{selection_kind}_alerts_{side}.csv", side_frame)
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
    summary_path = reports_dir / "final_alerts_side_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(summary_path)


if __name__ == "__main__":
    main()
