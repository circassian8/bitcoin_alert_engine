#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import pandas as pd


VARIANT_COLORS = ["#1f4b99", "#2f855a", "#b7791f", "#c53030", "#6b46c1"]
OUTCOME_ORDER = ["tp", "sl", "timeout"]


def _load(csv_path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"no rows found in {csv_path}")
    frame["ts_dt"] = pd.to_datetime(frame["ts_dt"], utc=True)
    frame = frame.sort_values("ts_dt").reset_index(drop=True)
    frame["variant"] = label
    frame["trade_index"] = range(1, len(frame) + 1)
    frame["cumulative_r"] = frame["realized_r"].cumsum()
    frame["drawdown_r"] = frame["cumulative_r"].cummax() - frame["cumulative_r"]
    if "pnl_usd" in frame.columns:
        frame["cumulative_pnl_usd"] = frame["pnl_usd"].cumsum()
    frame["month"] = frame["ts_dt"].dt.strftime("%Y-%m")
    return frame


def _summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in frames:
        label = str(frame["variant"].iloc[0])
        precision = float((frame["outcome"] == "tp").mean())
        expectancy = float(frame["realized_r"].mean())
        worst_dd = float(frame["drawdown_r"].max())
        avg_hold = float(frame["holding_minutes"].median())
        rows.extend(
            [
                {"variant": label, "metric": "Trades", "label": str(len(frame)), "order": 0},
                {"variant": label, "metric": "Precision", "label": f"{precision:.1%}", "order": 1},
                {"variant": label, "metric": "Expectancy", "label": f"{expectancy:.3f}R", "order": 2},
                {"variant": label, "metric": "Worst DD", "label": f"{worst_dd:.3f}R", "order": 3},
                {"variant": label, "metric": "Median Hold", "label": f"{avg_hold:.0f} min", "order": 4},
            ]
        )
        if "pnl_usd" in frame.columns:
            rows.extend(
                [
                    {"variant": label, "metric": "Total PnL", "label": f"${frame['pnl_usd'].sum():,.2f}", "order": 5},
                    {"variant": label, "metric": "Avg Risk", "label": f"${frame['risk_usd'].mean():,.2f}", "order": 6},
                ]
            )
    return pd.DataFrame(rows)


def build_dashboard(frames: list[pd.DataFrame], *, title: str) -> alt.Chart:
    alt.data_transformers.disable_max_rows()
    combined = pd.concat(frames, ignore_index=True)
    summary = _summary(frames)
    variants = list(dict.fromkeys(combined["variant"]))
    colors = VARIANT_COLORS[: len(variants)]

    cards = (
        alt.Chart(summary)
        .mark_rect(cornerRadius=10, stroke="#d9e2ec", strokeWidth=1)
        .encode(
            x=alt.X("order:O", axis=None, sort=list(summary["order"].drop_duplicates())),
            y=alt.Y("variant:N", axis=alt.Axis(title=None)),
            color=alt.value("#f7fafc"),
        )
        .properties(width=135, height=110, title=title)
    )
    card_metric = (
        alt.Chart(summary)
        .mark_text(dy=-10, fontSize=12, fontWeight="bold", color="#4a5568")
        .encode(
            x=alt.X("order:O", axis=None, sort=list(summary["order"].drop_duplicates())),
            y="variant:N",
            text="metric:N",
        )
    )
    card_value = (
        alt.Chart(summary)
        .mark_text(dy=12, fontSize=17, fontWeight=700, color="#1a202c")
        .encode(
            x=alt.X("order:O", axis=None, sort=list(summary["order"].drop_duplicates())),
            y="variant:N",
            text="label:N",
        )
    )

    equity = (
        alt.Chart(combined)
        .mark_line(point=True)
        .encode(
            x=alt.X("ts_dt:T", title="Trade time"),
            y=alt.Y("cumulative_r:Q", title="Cumulative R"),
            color=alt.Color("variant:N", scale=alt.Scale(domain=variants, range=colors), title="Variant"),
            tooltip=[
                alt.Tooltip("variant:N", title="Variant"),
                alt.Tooltip("ts_dt:T", title="Time"),
                alt.Tooltip("candidate_id:N", title="Candidate"),
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("realized_r:Q", title="Realized R", format=".3f"),
                alt.Tooltip("cumulative_r:Q", title="Cum R", format=".3f"),
                alt.Tooltip("p:Q", title="Probability", format=".3f"),
            ],
        )
        .properties(height=280, title="Portfolio Equity Comparison")
    )

    drawdown = (
        alt.Chart(combined)
        .mark_line(strokeDash=[6, 4])
        .encode(
            x=alt.X("ts_dt:T", title="Trade time"),
            y=alt.Y("drawdown_r:Q", title="Drawdown R"),
            color=alt.Color("variant:N", scale=alt.Scale(domain=variants, range=colors), title="Variant"),
            tooltip=[
                alt.Tooltip("variant:N", title="Variant"),
                alt.Tooltip("ts_dt:T", title="Time"),
                alt.Tooltip("drawdown_r:Q", title="Drawdown R", format=".3f"),
            ],
        )
        .properties(height=180, title="Drawdown Comparison")
    )

    monthly = (
        combined.groupby(["month", "variant"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    monthly_chart = (
        alt.Chart(monthly)
        .mark_bar()
        .encode(
            x=alt.X("month:N", title="Month", sort=list(monthly["month"].drop_duplicates())),
            y=alt.Y("count:Q", title="Trades"),
            color=alt.Color("variant:N", scale=alt.Scale(domain=variants, range=colors), title="Variant"),
            xOffset="variant:N",
            tooltip=["month:N", "variant:N", "count:Q"],
        )
        .properties(height=220, title="Monthly Trade Count")
    )

    outcome_counts = (
        combined.groupby(["variant", "outcome"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    outcome_chart = (
        alt.Chart(outcome_counts)
        .mark_bar()
        .encode(
            x=alt.X("variant:N", title="Variant", sort=variants),
            y=alt.Y("count:Q", title="Trades"),
            color=alt.Color("outcome:N", title="Outcome", sort=OUTCOME_ORDER),
            tooltip=["variant:N", "outcome:N", "count:Q"],
        )
        .properties(height=220, title="Outcome Mix")
    )

    charts: list[alt.Chart] = [alt.layer(cards, card_metric, card_value), equity, drawdown, monthly_chart, outcome_chart]
    if "cumulative_pnl_usd" in combined.columns:
        pnl_chart = (
            alt.Chart(combined)
            .mark_line(point=True)
            .encode(
                x=alt.X("ts_dt:T", title="Trade time"),
                y=alt.Y("cumulative_pnl_usd:Q", title="Cumulative PnL (USD)"),
                color=alt.Color("variant:N", scale=alt.Scale(domain=variants, range=colors), title="Variant"),
                tooltip=[
                    alt.Tooltip("variant:N", title="Variant"),
                    alt.Tooltip("ts_dt:T", title="Time"),
                    alt.Tooltip("pnl_usd:Q", title="Trade PnL", format=",.2f"),
                    alt.Tooltip("cumulative_pnl_usd:Q", title="Cum PnL", format=",.2f"),
                ],
            )
            .properties(height=240, title="Cumulative PnL Comparison")
        )
        charts.append(pnl_chart)

    dashboard = alt.vconcat(*charts, spacing=18).resolve_scale(color="independent")
    return dashboard.configure_view(stroke=None).configure_axis(
        labelColor="#2d3748",
        titleColor="#1a202c",
        gridColor="#e2e8f0",
    ).configure_title(fontSize=18, anchor="start", color="#1a202c").configure_legend(
        labelColor="#2d3748",
        titleColor="#1a202c",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML comparison dashboard for multiple portfolio alert variants.")
    parser.add_argument("inputs", nargs="+", help="Inputs in label=path.csv form.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default="Portfolio Variant Comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    for item in args.inputs:
        if "=" not in item:
            raise SystemExit(f"expected label=path.csv, got: {item}")
        label, raw_path = item.split("=", 1)
        frames.append(_load(Path(raw_path), label))
    output_path = Path(args.output) if args.output else Path("variant_comparison.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_dashboard(frames, title=args.title).save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
