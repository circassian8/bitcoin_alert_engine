from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from btc_alert_engine.schemas import PriceBar, RawEvent

_INTERVAL_MS = {
    "1": 60_000,
    "3": 3 * 60_000,
    "5": 5 * 60_000,
    "15": 15 * 60_000,
    "30": 30 * 60_000,
    "60": 60 * 60_000,
    "120": 120 * 60_000,
    "240": 240 * 60_000,
    "360": 360 * 60_000,
    "720": 720 * 60_000,
    "D": 24 * 60 * 60_000,
    "W": 7 * 24 * 60 * 60_000,
    "M": 30 * 24 * 60 * 60_000,
}


def _list_rows(payload: dict[str, Any]) -> list[Any]:
    result = payload.get("result", {})
    rows = result.get("list")
    if rows is None:
        return []
    return list(rows)


def _topic_interval(topic: str) -> str:
    bits = topic.split(".")
    middle = bits[1]
    # e.g. rest.kline_15.BTCUSDT, rest.index_price_kline_15.BTCUSDT
    return middle.rsplit("_", 1)[1]


def _topic_kind(topic: str) -> str:
    bits = topic.split(".")
    return bits[1]


def parse_kline_event(event: RawEvent) -> list[PriceBar]:
    topic = event.topic
    if not topic.startswith("rest.") or "kline" not in topic:
        return []
    interval = _topic_interval(topic)
    interval_ms = _INTERVAL_MS[interval]
    source_kind = _topic_kind(topic)
    cutoff = event.exchange_ts if event.exchange_ts is not None else event.local_received_ts
    bars: list[PriceBar] = []
    for row in _list_rows(event.payload):
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        start = int(row[0])
        ts = start + interval_ms
        if ts > cutoff:
            # Ignore open/unsettled bars.
            continue
        volume = float(row[5]) if len(row) > 5 else 0.0
        turnover = float(row[6]) if len(row) > 6 else 0.0
        bars.append(
            PriceBar(
                ts=ts,
                symbol=event.symbol,
                interval=interval,
                open=row[1],
                high=row[2],
                low=row[3],
                close=row[4],
                volume=volume,
                turnover=turnover,
                source_topic=source_kind,
            )
        )
    return bars


def parse_funding_history_event(event: RawEvent) -> list[dict[str, float | int | str]]:
    if not event.topic.startswith("rest.funding_history."):
        return []
    results: list[dict[str, float | int | str]] = []
    for row in _list_rows(event.payload):
        if not isinstance(row, dict):
            continue
        ts = int(row["fundingRateTimestamp"])
        symbol = str(row.get("symbol") or event.symbol)
        results.append({
            "ts": ts,
            "symbol": symbol,
            "funding_rate": float(row["fundingRate"]),
        })
    return results


def parse_open_interest_event(event: RawEvent) -> list[dict[str, float | int | str]]:
    if not event.topic.startswith("rest.open_interest_"):
        return []
    results: list[dict[str, float | int | str]] = []
    for row in _list_rows(event.payload):
        if not isinstance(row, dict):
            continue
        ts = int(row["timestamp"])
        symbol = str(row.get("symbol") or event.symbol)
        results.append({
            "ts": ts,
            "symbol": symbol,
            "open_interest": float(row["openInterest"]),
        })
    return results


def parse_account_ratio_event(event: RawEvent) -> list[dict[str, float | int | str]]:
    if not event.topic.startswith("rest.account_ratio_"):
        return []
    results: list[dict[str, float | int | str]] = []
    for row in _list_rows(event.payload):
        if not isinstance(row, dict):
            continue
        ts = int(row["timestamp"])
        buy_ratio = float(row.get("buyRatio") or row.get("longAccount") or 0.0)
        sell_ratio = float(row.get("sellRatio") or row.get("shortAccount") or 0.0)
        symbol = str(row.get("symbol") or event.symbol)
        results.append({
            "ts": ts,
            "symbol": symbol,
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
        })
    return results


def latest_by_timestamp(rows: Iterable[PriceBar]) -> list[PriceBar]:
    latest: dict[int, PriceBar] = {}
    for row in rows:
        latest[row.ts] = row
    return [latest[ts] for ts in sorted(latest)]
