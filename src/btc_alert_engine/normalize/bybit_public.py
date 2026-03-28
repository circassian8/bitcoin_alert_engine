from __future__ import annotations

from decimal import Decimal
from typing import Any

from btc_alert_engine.schemas import BookLevel, BybitOrderBookMessage, NormalizedLiquidation, NormalizedTrade, RawEvent


class UnsupportedBybitMessage(ValueError):
    pass


def parse_orderbook_payload(payload: dict[str, Any]) -> BybitOrderBookMessage:
    topic = str(payload["topic"])
    data = payload["data"]
    return BybitOrderBookMessage(
        topic=topic,
        message_type=payload["type"],
        ts=int(payload["ts"]),
        symbol=str(data["s"]),
        update_id=int(data["u"]),
        seq=int(data["seq"]) if data.get("seq") is not None else None,
        cts=int(payload["cts"]) if payload.get("cts") is not None else None,
        bids=[BookLevel(price=px, size=qty) for px, qty in data.get("b", [])],
        asks=[BookLevel(price=px, size=qty) for px, qty in data.get("a", [])],
    )


def parse_trade_payload(payload: dict[str, Any]) -> list[NormalizedTrade]:
    results: list[NormalizedTrade] = []
    for item in payload.get("data", []):
        results.append(
            NormalizedTrade(
                ts=int(item["T"]),
                symbol=str(item["s"]),
                price=item["p"],
                size=item["v"],
                side=item["S"],
                trade_id=str(item["i"]),
                seq=int(item["seq"]) if item.get("seq") is not None else None,
                tick_direction=item.get("L"),
                block_trade=bool(item.get("BT", False)),
                rpi_trade=bool(item.get("RPI", False)),
            )
        )
    return results


def parse_liquidation_payload(payload: dict[str, Any]) -> list[NormalizedLiquidation]:
    results: list[NormalizedLiquidation] = []
    for item in payload.get("data", []):
        results.append(
            NormalizedLiquidation(
                ts=int(item["T"]),
                symbol=str(item["s"]),
                side=item["S"],
                size=item["v"],
                bankruptcy_price=item["p"],
            )
        )
    return results


def normalize_raw_bybit_event(event: RawEvent) -> list[object]:
    payload = event.payload
    topic = event.topic
    if topic.startswith("orderbook."):
        return [parse_orderbook_payload(payload)]
    if topic.startswith("publicTrade."):
        return parse_trade_payload(payload)
    if topic.startswith("allLiquidation."):
        return parse_liquidation_payload(payload)
    raise UnsupportedBybitMessage(f"Unsupported Bybit raw topic: {topic}")


def compute_book_imbalance(top_bids: list[BookLevel], top_asks: list[BookLevel], depth: int = 10) -> Decimal:
    bid_vol = sum(level.size for level in top_bids[:depth])
    ask_vol = sum(level.size for level in top_asks[:depth])
    denom = bid_vol + ask_vol
    if denom == 0:
        return Decimal("0")
    return (bid_vol - ask_vol) / denom
