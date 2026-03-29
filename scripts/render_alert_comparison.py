#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import pandas as pd


SCOPE_COLORS = {
    "event": "#1f4b99",
    "portfolio": "#2f855a",
}


def _load(csv_path: Path, scope: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"no rows found in {csv_path}")
    frame["ts_dt"] = pd.to_datetime(frame["ts_dt"], utc=True)
    frame = frame.sort_values("ts_dt").reset_index(drop=True)
    frame["scope"] = scope
    frame["cumulative_r"] = frame["realized_r"].cumsum()
    if "pnl_usd" in frame.columns:
        frame["cumulative_pnl_usd"] = frame["pnl_usd"].cumsum()
    return frame


def _summary_frame(event_frame: pd.DataFrame, portfolio_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, frame in [("event", event_frame), ("portfolio", portfolio_frame)]:
        precision = float((frame["outcome"] == "tp").mean()) if len(frame) else None
        expectancy = float(frame["realized_r"].mean()) if len(frame) else None
        avg_hold = float(frame["holding_minutes"].median()) if len(frame) else None
        rows.extend(
            [
                {"scope": scope, "metric": "Alerts", "value": int(len(frame)), "label": str(int(len(frame))), "order": 0},
                {"scope": scope, "metric": "Precision", "value": precision, "label": "n/a" if precision is None else f"{precision:.1%}", "order": 1},
                {"scope": scope, "metric": "Expectancy", "value": expectancy, "label": "n/a" if expectancy is None else f"{expectancy:.3f}R", "order": 2},
                {"scope": scope, "metric": "Median Hold", "value": avg_hold, "label": "n/a" if avg_hold is None else f"{avg_hold:.0f} min", "order": 3},
            ]
        )
        if "pnl_usd" in frame.columns:
            total_pnl = float(frame["pnl_usd"].sum())
            rows.append({"scope": scope, "metric": "Total PnL", "value": total_pnl, "label": f"${total_pnl:,.2f}", "order": 4})
    return pd.DataFrame(rows)


def build_dashboard(event_frame: pd.DataFrame, portfolio_frame: pd.DataFrame, *, title: str) -> alt.Chart:
    alt.data_transformers.disable_max_rows()
    combined = pd.concat([event_frame, portfolio_frame], ignore_index=True)
    summary = _summary_frame(event_frame, portfolio_frame)

    cards = (
        alt.Chart(summary)
        .mark_rect(cornerRadius=10, stroke="#d9e2ec", strokeWidth=1)
        .encode(
            x=alt.X("order:O", axis=None, sort=list(summary["order"].drop_duplicates())),
            y=alt.Y("scope:N", axis=alt.Axis(title=None)),
            color=alt.value("#f7fafc"),
        )
        .properties(width=130, height=90, title=title)
    )
    card_metric = alt.Chart(summary).mark_text(dy=-10, fontSize=13, fontWeight="bold", color="#4a5568").encode(
        x=alt.X("order:O", axis=None, sort=list(summary["order"].drop_duplicates())),
        y="scope:N",
        text="metric:N",
    )
    card_value = alt.Chart(summary).mark_text(dy=12, fontSize=18, fontWeight=700, color="#1a202c").encode(
        x=alt.X("order:O", axis=None, sort=list(summary["order"].drop_duplicates())),
        y="scope:N",
        text="label:N",
    )

    line_r = (
        alt.Chart(combined)
        .mark_line(point=True)
        .encode(
            x=alt.X("ts_dt:T", title="Trade time"),
            y=alt.Y("cumulative_r:Q", title="Cumulative R"),
            color=alt.Color("scope:N", scale=alt.Scale(domain=list(SCOPE_COLORS), range=list(SCOPE_COLORS.values())), title="Path"),
            tooltip=[
                alt.Tooltip("scope:N", title="Path"),
                alt.Tooltip("ts_dt:T", title="Time"),
                alt.Tooltip("candidate_id:N", title="Candidate"),
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("realized_r:Q", title="Realized R", format=".3f"),
                alt.Tooltip("cumulative_r:Q", title="Cumulative R", format=".3f"),
            ],
        )
        .properties(height=260, title="Event vs Portfolio Equity")
    )

    monthly = (
        combined.groupby([combined["ts_dt"].dt.strftime("%Y-%m"), "scope"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"ts_dt": "month"})
    )
    bar = (
        alt.Chart(monthly)
        .mark_bar()
        .encode(
            x=alt.X("month:N", title="Month", sort=list(monthly["month"].drop_duplicates())),
            y=alt.Y("count:Q", title="Trades"),
            color=alt.Color("scope:N", scale=alt.Scale(domain=list(SCOPE_COLORS), range=list(SCOPE_COLORS.values())), title="Path"),
            xOffset="scope:N",
            tooltip=["month:N", "scope:N", "count:Q"],
        )
        .properties(height=220, title="Monthly Trade Count")
    )

    charts: list[alt.Chart] = [alt.layer(cards, card_metric, card_value), line_r, bar]
    if "cumulative_pnl_usd" in combined.columns:
        line_usd = (
            alt.Chart(combined)
            .mark_line(point=True)
            .encode(
                x=alt.X("ts_dt:T", title="Trade time"),
                y=alt.Y("cumulative_pnl_usd:Q", title="Cumulative PnL (USD)"),
                color=alt.Color("scope:N", scale=alt.Scale(domain=list(SCOPE_COLORS), range=list(SCOPE_COLORS.values())), title="Path"),
                tooltip=[
                    alt.Tooltip("scope:N", title="Path"),
                    alt.Tooltip("ts_dt:T", title="Time"),
                    alt.Tooltip("pnl_usd:Q", title="Trade PnL", format=",.2f"),
                    alt.Tooltip("cumulative_pnl_usd:Q", title="Cumulative PnL", format=",.2f"),
                ],
            )
            .properties(height=220, title="Event vs Portfolio PnL")
        )
        charts.append(line_usd)
    dashboard = alt.vconcat(*charts, spacing=18).resolve_scale(color="shared")
    return dashboard.configure_view(stroke=None).configure_axis(
        labelColor="#2d3748",
        titleColor="#1a202c",
        gridColor="#e2e8f0",
    ).configure_title(fontSize=18, anchor="start", color="#1a202c").configure_legend(
        labelColor="#2d3748",
        titleColor="#1a202c",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML comparison dashboard for event vs portfolio alerts.")
    parser.add_argument("event_csv")
    parser.add_argument("portfolio_csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default="Alert Path Comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_path = Path(args.event_csv)
    portfolio_path = Path(args.portfolio_csv)
    output_path = Path(args.output) if args.output else portfolio_path.with_name(portfolio_path.stem.replace("_portfolio_", "_comparison_") + ".html")
    event_frame = _load(event_path, "event")
    portfolio_frame = _load(portfolio_path, "portfolio")
    dashboard = build_dashboard(event_frame, portfolio_frame, title=args.title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
