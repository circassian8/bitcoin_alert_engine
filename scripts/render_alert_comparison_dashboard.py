#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import pandas as pd


OUTCOME_COLORS = {
    "Take Profit": "#2f855a",
    "Stop Loss": "#c53030",
    "Timeout": "#d69e2e",
}

GROUP_COLORS = {
    "Long": "#1f4b99",
    "Short": "#dd6b20",
}


def group_scale(frame: pd.DataFrame) -> alt.Scale:
    groups = list(frame["group"].drop_duplicates())
    palette = list(GROUP_COLORS.values())
    colors = [palette[idx % len(palette)] for idx in range(len(groups))]
    return alt.Scale(domain=groups, range=colors)


def load_alerts(csv_path: Path, *, group_label: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"no rows found in {csv_path}")
    frame["ts_dt"] = pd.to_datetime(frame["ts_dt"], utc=True)
    frame = frame.sort_values("ts_dt").reset_index(drop=True)
    frame["trade_index"] = range(1, len(frame) + 1)
    frame["cumulative_r"] = frame["realized_r"].cumsum()
    frame["equity_peak_r"] = frame["cumulative_r"].cummax()
    frame["drawdown_r"] = frame["equity_peak_r"] - frame["cumulative_r"]
    frame["month"] = frame["ts_dt"].dt.strftime("%Y-%m")
    frame["outcome_label"] = frame["outcome"].map({"tp": "Take Profit", "sl": "Stop Loss", "timeout": "Timeout"}).fillna(frame["outcome"])
    frame["group"] = group_label
    return frame


def build_metric_compare(frame: pd.DataFrame) -> alt.Chart:
    rows: list[dict[str, object]] = []
    for group, group_frame in frame.groupby("group", sort=False):
        rows.extend(
            [
                {"group": group, "metric": "Alerts", "value": float(len(group_frame)), "label": str(len(group_frame))},
                {
                    "group": group,
                    "metric": "Precision",
                    "value": float((group_frame["outcome"] == "tp").mean()),
                    "label": f"{(group_frame['outcome'] == 'tp').mean():.1%}",
                },
                {
                    "group": group,
                    "metric": "Expectancy",
                    "value": float(group_frame["realized_r"].mean()),
                    "label": f"{group_frame['realized_r'].mean():.3f}R",
                },
                {
                    "group": group,
                    "metric": "Mean Prob",
                    "value": float(group_frame["p"].mean()),
                    "label": f"{group_frame['p'].mean():.3f}",
                },
                {
                    "group": group,
                    "metric": "Median Hold",
                    "value": float(group_frame["holding_minutes"].median()),
                    "label": f"{group_frame['holding_minutes'].median():.0f} min",
                },
            ]
        )
    metric_frame = pd.DataFrame(rows)
    scale = group_scale(metric_frame)
    base = alt.Chart(metric_frame).encode(
        y=alt.Y("group:N", title=None),
        color=alt.Color("group:N", scale=scale, title="Side"),
    )
    bars = base.mark_bar(size=20).encode(
        x=alt.X("value:Q", title=None),
    )
    text = base.mark_text(align="left", dx=6, color="#1a202c").encode(
        x="value:Q",
        text="label:N",
    )
    return (
        alt.layer(bars, text)
        .facet(row=alt.Row("metric:N", title=None, header=alt.Header(labelAngle=0, labelFontSize=12, labelColor="#1a202c")))
        .resolve_scale(x="independent")
        .properties(title="Side Summary")
    )


def build_equity_compare(frame: pd.DataFrame) -> alt.Chart:
    scale = group_scale(frame)
    base = alt.Chart(frame).encode(
        x=alt.X("ts_dt:T", title="Trade time"),
        y=alt.Y("cumulative_r:Q", title="Cumulative realized R"),
        color=alt.Color("group:N", scale=scale, title="Side"),
        tooltip=[
            alt.Tooltip("group:N", title="Side"),
            alt.Tooltip("ts_dt:T", title="Time"),
            alt.Tooltip("outcome_label:N", title="Outcome"),
            alt.Tooltip("realized_r:Q", title="Realized R", format=".3f"),
            alt.Tooltip("cumulative_r:Q", title="Cumulative R", format=".3f"),
            alt.Tooltip("p:Q", title="Probability", format=".3f"),
            alt.Tooltip("holding_minutes:Q", title="Hold min", format=".0f"),
        ],
    )
    line = base.mark_line(strokeWidth=2).encode(detail="group:N")
    points = base.mark_circle(size=70, opacity=0.85).encode(shape=alt.Shape("outcome_label:N", title="Outcome"))
    return alt.layer(line, points).properties(height=280, title="Equity Curve")


def build_drawdown_compare(frame: pd.DataFrame) -> alt.Chart:
    scale = group_scale(frame)
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("ts_dt:T", title="Trade time"),
            y=alt.Y("drawdown_r:Q", title="Drawdown R"),
            color=alt.Color("group:N", scale=scale, title="Side"),
            tooltip=[
                alt.Tooltip("group:N", title="Side"),
                alt.Tooltip("ts_dt:T", title="Time"),
                alt.Tooltip("drawdown_r:Q", title="Drawdown R", format=".3f"),
            ],
        )
        .properties(height=140, title="Drawdown")
    )


def build_outcome_mix(frame: pd.DataFrame) -> alt.Chart:
    outcome_frame = frame.groupby(["group", "outcome_label"], dropna=False).size().reset_index(name="count")
    return (
        alt.Chart(outcome_frame)
        .mark_bar()
        .encode(
            x=alt.X("group:N", title="Side"),
            xOffset=alt.XOffset("outcome_label:N"),
            y=alt.Y("count:Q", title="Trade count"),
            color=alt.Color(
                "outcome_label:N",
                title="Outcome",
                scale=alt.Scale(domain=list(OUTCOME_COLORS), range=list(OUTCOME_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("group:N", title="Side"),
                alt.Tooltip("outcome_label:N", title="Outcome"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=220, title="Outcome Mix")
    )


def build_monthly_pnl(frame: pd.DataFrame) -> alt.Chart:
    monthly = frame.groupby(["month", "group"], dropna=False)["realized_r"].sum().reset_index(name="realized_r_sum")
    month_order = list(monthly["month"].drop_duplicates())
    scale = group_scale(monthly)
    return (
        alt.Chart(monthly)
        .mark_bar()
        .encode(
            x=alt.X("month:N", title="Month", sort=month_order),
            xOffset=alt.XOffset("group:N"),
            y=alt.Y("realized_r_sum:Q", title="Monthly realized R"),
            color=alt.Color("group:N", scale=scale, title="Side"),
            tooltip=[
                alt.Tooltip("month:N", title="Month"),
                alt.Tooltip("group:N", title="Side"),
                alt.Tooltip("realized_r_sum:Q", title="Realized R", format=".3f"),
            ],
        )
        .properties(height=220, title="Monthly Realized R")
    )


def build_probability_compare(frame: pd.DataFrame) -> alt.Chart:
    scale = group_scale(frame)
    return (
        alt.Chart(frame)
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X("p:Q", title="Model probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("realized_r:Q", title="Realized R"),
            color=alt.Color("group:N", scale=scale, title="Side"),
            shape=alt.Shape("outcome_label:N", title="Outcome"),
            tooltip=[
                alt.Tooltip("group:N", title="Side"),
                alt.Tooltip("ts_dt:T", title="Time"),
                alt.Tooltip("outcome_label:N", title="Outcome"),
                alt.Tooltip("p:Q", title="Probability", format=".3f"),
                alt.Tooltip("realized_r:Q", title="Realized R", format=".3f"),
            ],
        )
        .properties(height=220, title="Probability vs Outcome")
    )


def build_dashboard(frame: pd.DataFrame, *, title: str) -> alt.Chart:
    alt.data_transformers.disable_max_rows()
    metrics = build_metric_compare(frame)
    top = build_equity_compare(frame) & build_drawdown_compare(frame)
    bottom = build_outcome_mix(frame) | build_monthly_pnl(frame) | build_probability_compare(frame)
    return (
        alt.vconcat(metrics, top, bottom, spacing=18)
        .properties(title=title)
        .configure_view(stroke=None)
        .configure_axis(labelColor="#2d3748", titleColor="#1a202c", gridColor="#e2e8f0")
        .configure_title(fontSize=18, anchor="start", color="#1a202c")
        .configure_legend(labelColor="#2d3748", titleColor="#1a202c")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a comparison dashboard for two alert CSVs.")
    parser.add_argument("left_csv", help="First alert CSV.")
    parser.add_argument("right_csv", help="Second alert CSV.")
    parser.add_argument("--left-label", default="Long", help="Legend label for the first CSV.")
    parser.add_argument("--right-label", default="Short", help="Legend label for the second CSV.")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults next to the long CSV.")
    parser.add_argument("--title", default="Long vs Short Alert Comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_path = Path(args.left_csv)
    right_path = Path(args.right_csv)
    output_path = Path(args.output) if args.output else left_path.with_name(left_path.stem + "_comparison.html")

    left_frame = load_alerts(left_path, group_label=args.left_label)
    right_frame = load_alerts(right_path, group_label=args.right_label)
    dashboard = build_dashboard(pd.concat([left_frame, right_frame], ignore_index=True), title=args.title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
