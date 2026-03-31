#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REVISED_PREFIX = "F"
BASELINE_ID = "S2_current_fast_crowding_feature"


def _load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty summary metrics: {path}")
    return frame


def _pick_best(frame: pd.DataFrame, *, budget: int) -> pd.Series:
    subset = frame[(frame["split_kind"] == "final") & (frame["budget_per_week"] == budget)].copy()
    subset = subset[subset["experiment_id"].astype(str).str.startswith(REVISED_PREFIX)]
    if subset.empty:
        raise ValueError(f"no revised fast variants found for budget {budget}")
    subset = subset.sort_values(
        ["expectancy_r_per_alert", "precision_tp_before_sl", "n_alerts"],
        ascending=[False, False, False],
    )
    return subset.iloc[0]


def _policy_label(summary_path: Path) -> str:
    name = summary_path.parent.name
    if "highest_probability" in name:
        return "highest_probability"
    if "first_signal" in name:
        return "first_signal"
    return name


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _export_and_render(
    *,
    python_bin: str,
    data_dir: Path,
    registry: Path,
    reports_dir: Path,
    output_dir: Path,
    experiment_id: str,
    title_prefix: str,
) -> tuple[Path, Path]:
    _run(
        [
            python_bin,
            "scripts/export_final_walkforward_alerts.py",
            "--data-dir",
            str(data_dir),
            "--registry",
            str(registry),
            "--reports-dir",
            str(reports_dir),
            "--experiment-id",
            experiment_id,
            "--output-dir",
            str(output_dir),
        ]
    )
    event_csv = output_dir / f"{experiment_id}_final_event_alerts.csv"
    portfolio_csv = output_dir / f"{experiment_id}_final_portfolio_alerts.csv"
    if event_csv.exists():
        _run([python_bin, "scripts/render_alert_dashboard.py", str(event_csv), "--title", f"{title_prefix} event alerts"]) 
    if portfolio_csv.exists():
        _run([python_bin, "scripts/render_alert_dashboard.py", str(portfolio_csv), "--title", f"{title_prefix} portfolio alerts"]) 
    if event_csv.exists() and portfolio_csv.exists():
        _run(
            [
                python_bin,
                "scripts/render_alert_comparison.py",
                str(event_csv),
                str(portfolio_csv),
                "--output",
                str(output_dir / f"{experiment_id}_event_vs_portfolio.html"),
                "--title",
                f"{title_prefix} event vs portfolio",
            ]
        )
    return event_csv, portfolio_csv


def _load_side_summary(path: Path, *, experiment_id: str, selection_kind: str = "portfolio") -> pd.DataFrame:
    frame = pd.read_csv(path)
    mask = (frame["experiment_id"] == experiment_id) & (frame["selection_kind"] == selection_kind)
    return frame.loc[mask].copy()


def _event_vs_portfolio_line(summary: pd.Series) -> str:
    return (
        f"portfolio {summary['n_alerts']:.0f} alerts @ {summary['alerts_per_week']:.2f}/week, "
        f"{summary['precision_tp_before_sl']:.1%} precision, {summary['expectancy_r_per_alert']:.3f}R/alert; "
        f"event {summary['event_n_alerts']:.0f} alerts @ {summary['event_alerts_per_week']:.2f}/week, "
        f"{summary['event_precision_tp_before_sl']:.1%} precision, {summary['event_expectancy_r_per_alert']:.3f}R/alert; "
        f"overlap filtered {summary['overlap_filtered_alerts']:.0f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess revised fast walk-forward outputs, export alerts, and render comparisons.")
    parser.add_argument("--data-dir", default="./data-pilot-fast")
    parser.add_argument("--broad-first", default="./reports/data-pilot-fast/walkforward_broad_test_fast_revised_first_signal")
    parser.add_argument("--broad-highest", default="./reports/data-pilot-fast/walkforward_broad_test_fast_revised_highest_probability")
    parser.add_argument("--registry-first", default="./research_registry_broad_test_fast_revised_first_signal.yaml")
    parser.add_argument("--registry-highest", default="./research_registry_broad_test_fast_revised_highest_probability.yaml")
    parser.add_argument("--output-dir", default="./reports/data-pilot-fast/fast_revised_postprocess")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    broad_first = Path(args.broad_first)
    broad_highest = Path(args.broad_highest)
    summaries = {
        "first_signal": _load_summary(broad_first / "summary_metrics.csv"),
        "highest_probability": _load_summary(broad_highest / "summary_metrics.csv"),
    }
    registries = {
        "first_signal": Path(args.registry_first),
        "highest_probability": Path(args.registry_highest),
    }
    reports_dirs = {
        "first_signal": broad_first,
        "highest_probability": broad_highest,
    }

    best_budget1: tuple[str, pd.Series] | None = None
    best_budget3: tuple[str, pd.Series] | None = None
    for policy, frame in summaries.items():
        row1 = _pick_best(frame, budget=1)
        row3 = _pick_best(frame, budget=3)
        if best_budget1 is None or float(row1["expectancy_r_per_alert"]) > float(best_budget1[1]["expectancy_r_per_alert"]):
            best_budget1 = (policy, row1)
        if best_budget3 is None or float(row3["expectancy_r_per_alert"]) > float(best_budget3[1]["expectancy_r_per_alert"]):
            best_budget3 = (policy, row3)

    assert best_budget1 is not None and best_budget3 is not None
    best_policy, best_row = best_budget3
    best_experiment = str(best_row["experiment_id"])

    export_dir = output_dir / f"exports_{best_policy}"
    export_dir.mkdir(parents=True, exist_ok=True)
    best_event_csv, best_portfolio_csv = _export_and_render(
        python_bin=args.python_bin,
        data_dir=data_dir,
        registry=registries[best_policy],
        reports_dir=reports_dirs[best_policy],
        output_dir=export_dir,
        experiment_id=best_experiment,
        title_prefix=f"{best_experiment} ({best_policy})",
    )
    base_event_csv, base_portfolio_csv = _export_and_render(
        python_bin=args.python_bin,
        data_dir=data_dir,
        registry=registries[best_policy],
        reports_dir=reports_dirs[best_policy],
        output_dir=export_dir,
        experiment_id=BASELINE_ID,
        title_prefix=f"{BASELINE_ID} ({best_policy})",
    )

    if best_portfolio_csv.exists() and base_portfolio_csv.exists():
        _run(
            [
                args.python_bin,
                "scripts/render_variant_comparison.py",
                f"best={best_portfolio_csv}",
                f"baseline={base_portfolio_csv}",
                "--output",
                str(output_dir / f"{best_experiment}_vs_{BASELINE_ID}_{best_policy}.html"),
                "--title",
                f"Best revised fast variant vs current fast S2 ({best_policy})",
            ]
        )

    side_summary_path = export_dir / "final_alerts_side_summary.csv"
    side_summary = _load_side_summary(side_summary_path, experiment_id=best_experiment) if side_summary_path.exists() else pd.DataFrame()

    lines = [
        "# Revised fast continuation postprocess report",
        "",
        f"Best budget 3 variant: **{best_experiment}** via **{best_policy}**.",
        f"Best budget 1 variant: **{best_budget1[1]['experiment_id']}** via **{best_budget1[0]}**.",
        "",
        "## Best budget 3 metrics",
        "",
        _event_vs_portfolio_line(best_row),
        "",
        "## Best budget 1 metrics",
        "",
        _event_vs_portfolio_line(best_budget1[1]),
        "",
        "## Long vs short contribution (portfolio path)",
        "",
    ]
    if side_summary.empty:
        lines.append("No side summary was available.")
    else:
        lines.append("| side | alerts | precision | expectancy | mean probability | median hold min |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in side_summary.iterrows():
            lines.append(
                f"| {row['side']} | {int(row['n_alerts'])} | {float(row['precision_tp_before_sl']):.1%} | {float(row['expectancy_r_per_alert']):.3f}R | {float(row['mean_probability']):.3f} | {float(row['median_holding_minutes']):.0f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- **Raw signal quality** improved if event-level precision/expectancy rises.",
            "- **Overlap reduction** improved if portfolio metrics rise mostly because overlap_filtered_alerts increases while event metrics stay similar.",
            "- **Portfolio selection** improved if highest_probability beats first_signal on the same variant with similar event metrics.",
            "- **Long vs short fixes** improved if one side contributes materially more expectancy than before.",
            "",
            f"Exports written to `{export_dir}`.",
            f"Comparison dashboard: `{output_dir / f'{best_experiment}_vs_{BASELINE_ID}_{best_policy}.html'}`",
        ]
    )
    report_path = output_dir / "fast_revised_postprocess_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
