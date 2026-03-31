#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import altair as alt
import pandas as pd


OUTCOME_COLORS = {
    "tp": "#2f855a",
    "sl": "#c53030",
    "timeout": "#d69e2e",
}
CARD_WIDTH = 148


def _format_metric(value: float | int | None, *, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if kind == "int":
        return f"{int(value)}"
    if kind == "pct":
        return f"{float(value):.1%}"
    if kind == "r":
        return f"{float(value):.3f}R"
    if kind == "num":
        return f"{float(value):.2f}"
    return str(value)


def load_alerts(csv_path: Path) -> pd.DataFrame:
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
    return frame


def build_summary_cards(frame: pd.DataFrame, title: str) -> alt.Chart:
    precision = (frame["outcome"] == "tp").mean()
    timeout_rate = (frame["outcome"] == "timeout").mean()
    rows = [
        {"metric": "Alerts", "value": _format_metric(len(frame), kind="int"), "order": 0},
        {"metric": "Precision", "value": _format_metric(precision, kind="pct"), "order": 1},
        {"metric": "Expectancy", "value": _format_metric(frame["realized_r"].mean(), kind="r"), "order": 2},
        {"metric": "Worst DD", "value": _format_metric(frame["drawdown_r"].max(), kind="r"), "order": 3},
        {"metric": "Avg Hold", "value": _format_metric(frame["holding_minutes"].median(), kind="num") + " min", "order": 4},
        {"metric": "Timeout Rate", "value": _format_metric(timeout_rate, kind="pct"), "order": 5},
    ]
    if "pnl_usd" in frame.columns:
        rows.extend(
            [
                {"metric": "Total PnL", "value": f"${frame['pnl_usd'].sum():,.2f}", "order": 6},
                {"metric": "Avg PnL", "value": f"${frame['pnl_usd'].mean():,.2f}", "order": 7},
            ]
        )
    cards = pd.DataFrame(rows)
    chart_width = max(len(cards) * CARD_WIDTH, 720)
    base = alt.Chart(cards).encode(
        x=alt.X("order:O", axis=None, sort=list(cards["order"]), scale=alt.Scale(paddingInner=0.08, paddingOuter=0.04)),
    )
    rect = base.mark_rect(cornerRadius=10, stroke="#d9e2ec", strokeWidth=1).encode(
        color=alt.value("#f7fafc"),
    )
    metric = base.mark_text(
        dy=-10,
        fontSize=13,
        fontWeight="bold",
        color="#4a5568",
    ).encode(text="metric:N")
    value = base.mark_text(
        dy=14,
        fontSize=18,
        fontWeight=700,
        color="#1a202c",
    ).encode(text="value:N")
    return (
        alt.layer(rect, metric, value)
        .properties(width=chart_width, height=74, title=title)
    )


def build_equity_chart(frame: pd.DataFrame) -> alt.Chart:
    tooltips = [
        alt.Tooltip("ts_dt:T", title="Time"),
        alt.Tooltip("candidate_id:N", title="Candidate"),
        alt.Tooltip("outcome_label:N", title="Outcome"),
        alt.Tooltip("realized_r:Q", title="Realized R", format=".3f"),
        alt.Tooltip("cumulative_r:Q", title="Cumulative R", format=".3f"),
        alt.Tooltip("p:Q", title="Probability", format=".3f"),
        alt.Tooltip("holding_minutes:Q", title="Hold min", format=".0f"),
        alt.Tooltip("executed_entry:Q", title="Entry", format=".2f"),
        alt.Tooltip("executed_exit:Q", title="Exit", format=".2f"),
    ]
    if "pnl_usd" in frame.columns:
        tooltips.extend(
            [
                alt.Tooltip("pnl_usd:Q", title="Trade PnL", format=",.2f"),
                alt.Tooltip("cumulative_pnl_usd:Q", title="Cum PnL", format=",.2f"),
            ]
        )
    base = alt.Chart(frame).encode(
        x=alt.X("ts_dt:T", title="Trade time"),
        tooltip=tooltips,
    )
    line = base.mark_line(color="#1f4b99", strokeWidth=2).encode(
        y=alt.Y("cumulative_r:Q", title="Cumulative realized R"),
    )
    points = base.mark_circle(size=70).encode(
        y="cumulative_r:Q",
        color=alt.Color(
            "outcome_label:N",
            title="Outcome",
            scale=alt.Scale(
                domain=["Take Profit", "Stop Loss", "Timeout"],
                range=[OUTCOME_COLORS["tp"], OUTCOME_COLORS["sl"], OUTCOME_COLORS["timeout"]],
            ),
        ),
    )
    return alt.layer(line, points).properties(height=280, title="Equity Curve")


def build_drawdown_chart(frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_area(color="#f56565", opacity=0.4)
        .encode(
            x=alt.X("ts_dt:T", title="Trade time"),
            y=alt.Y("drawdown_r:Q", title="Drawdown R"),
            tooltip=[
                alt.Tooltip("ts_dt:T", title="Time"),
                alt.Tooltip("drawdown_r:Q", title="Drawdown R", format=".3f"),
                alt.Tooltip("cumulative_r:Q", title="Cumulative R", format=".3f"),
            ],
        )
        .properties(height=130, title="Drawdown")
    )


def build_monthly_chart(frame: pd.DataFrame) -> alt.Chart:
    monthly = (
        frame.groupby(["month", "outcome_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    return (
        alt.Chart(monthly)
        .mark_bar()
        .encode(
            x=alt.X("month:N", title="Month", sort=list(monthly["month"].drop_duplicates())),
            y=alt.Y("count:Q", title="Alert count"),
            color=alt.Color(
                "outcome_label:N",
                title="Outcome",
                scale=alt.Scale(
                    domain=["Take Profit", "Stop Loss", "Timeout"],
                    range=[OUTCOME_COLORS["tp"], OUTCOME_COLORS["sl"], OUTCOME_COLORS["timeout"]],
                ),
            ),
            tooltip=[
                alt.Tooltip("month:N", title="Month"),
                alt.Tooltip("outcome_label:N", title="Outcome"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=220, title="Monthly Outcome Mix")
    )


def build_probability_chart(frame: pd.DataFrame) -> alt.Chart:
    tooltips = [
        alt.Tooltip("ts_dt:T", title="Time"),
        alt.Tooltip("candidate_id:N", title="Candidate"),
        alt.Tooltip("p:Q", title="Probability", format=".3f"),
        alt.Tooltip("realized_r:Q", title="Realized R", format=".3f"),
        alt.Tooltip("holding_minutes:Q", title="Hold min", format=".0f"),
    ]
    if "pnl_usd" in frame.columns:
        tooltips.append(alt.Tooltip("pnl_usd:Q", title="Trade PnL", format=",.2f"))
    return (
        alt.Chart(frame)
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("p:Q", title="Model probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("realized_r:Q", title="Realized R"),
            color=alt.Color(
                "outcome_label:N",
                title="Outcome",
                scale=alt.Scale(
                    domain=["Take Profit", "Stop Loss", "Timeout"],
                    range=[OUTCOME_COLORS["tp"], OUTCOME_COLORS["sl"], OUTCOME_COLORS["timeout"]],
                ),
            ),
            tooltip=tooltips,
        )
        .properties(height=220, title="Probability vs Outcome")
    )


def build_holding_chart(frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("holding_minutes:Q", bin=alt.Bin(maxbins=18), title="Holding time (minutes)"),
            y=alt.Y("count():Q", title="Trades"),
            color=alt.Color(
                "outcome_label:N",
                title="Outcome",
                scale=alt.Scale(
                    domain=["Take Profit", "Stop Loss", "Timeout"],
                    range=[OUTCOME_COLORS["tp"], OUTCOME_COLORS["sl"], OUTCOME_COLORS["timeout"]],
                ),
            ),
            tooltip=[
                alt.Tooltip("count():Q", title="Trades"),
                alt.Tooltip("outcome_label:N", title="Outcome"),
            ],
        )
        .properties(height=220, title="Holding Time Distribution")
    )


def build_dashboard(frame: pd.DataFrame, *, title: str) -> alt.Chart:
    alt.data_transformers.disable_max_rows()
    cards = build_summary_cards(frame, title)
    top = build_equity_chart(frame) & build_drawdown_chart(frame)
    bottom = build_monthly_chart(frame) | build_probability_chart(frame) | build_holding_chart(frame)
    charts: list[alt.Chart] = [cards, top, bottom]
    if "cumulative_pnl_usd" in frame.columns:
        pnl_chart = (
            alt.Chart(frame)
            .mark_line(point=True, color="#805ad5")
            .encode(
                x=alt.X("ts_dt:T", title="Trade time"),
                y=alt.Y("cumulative_pnl_usd:Q", title="Cumulative PnL (USD)"),
                tooltip=[
                    alt.Tooltip("ts_dt:T", title="Time"),
                    alt.Tooltip("pnl_usd:Q", title="Trade PnL", format=",.2f"),
                    alt.Tooltip("cumulative_pnl_usd:Q", title="Cum PnL", format=",.2f"),
                ],
            )
            .properties(height=220, title="Cumulative PnL")
        )
        charts.append(pnl_chart)
    dashboard = alt.vconcat(*charts, spacing=18).resolve_scale(color="shared")
    return dashboard.configure_view(stroke=None).configure_axis(
        labelColor="#2d3748",
        titleColor="#1a202c",
        gridColor="#e2e8f0",
    ).configure_title(
        fontSize=18,
        anchor="start",
        color="#1a202c",
    ).configure_legend(
        labelColor="#2d3748",
        titleColor="#1a202c",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an interactive HTML dashboard for alert CSV output.")
    parser.add_argument("input_csv", help="Path to the alert CSV to visualize.")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults next to the CSV.")
    parser.add_argument("--title", default=None, help="Optional dashboard title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")
    frame = load_alerts(input_path)
    title = args.title or input_path.stem.replace("_", " ")
    dashboard = build_dashboard(frame, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
