from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import httpx

from btc_alert_engine.collectors.bybit_rest import BYBIT_REST_BASE
from btc_alert_engine.schemas import RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter

HistoryParamsBuilder = Callable[[str, str, int, int], dict[str, Any]]


@dataclass(slots=True)
class BybitHistorySpec:
    name: str
    path: str
    step_ms: int
    params_builder: HistoryParamsBuilder


_DAY_MS = 24 * 60 * 60 * 1000
_OI_INTERVAL_MS = {
    "5min": 5 * 60 * 1000,
    "15min": 15 * 60 * 1000,
    "30min": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}
_RATIO_INTERVAL_MS = dict(_OI_INTERVAL_MS)


def default_bybit_history_specs(*, oi_interval: str = "5min", account_ratio_period: str = "5min") -> list[BybitHistorySpec]:
    if oi_interval not in _OI_INTERVAL_MS:
        raise ValueError(f"Unsupported oi_interval: {oi_interval}")
    if account_ratio_period not in _RATIO_INTERVAL_MS:
        raise ValueError(f"Unsupported account_ratio_period: {account_ratio_period}")
    oi_points_per_page = 200
    ratio_points_per_page = 500
    return [
        BybitHistorySpec(
            name="kline_15",
            path="/v5/market/kline",
            step_ms=7 * _DAY_MS,
            params_builder=lambda category, symbol, start, end: {
                "category": category,
                "symbol": symbol,
                "interval": "15",
                "start": start,
                "end": end,
                "limit": 1000,
            },
        ),
        BybitHistorySpec(
            name="index_price_kline_15",
            path="/v5/market/index-price-kline",
            step_ms=7 * _DAY_MS,
            params_builder=lambda category, symbol, start, end: {
                "category": category,
                "symbol": symbol,
                "interval": "15",
                "start": start,
                "end": end,
                "limit": 1000,
            },
        ),
        BybitHistorySpec(
            name="premium_index_price_kline_15",
            path="/v5/market/premium-index-price-kline",
            step_ms=7 * _DAY_MS,
            params_builder=lambda category, symbol, start, end: {
                "category": category,
                "symbol": symbol,
                "interval": "15",
                "start": start,
                "end": end,
                "limit": 1000,
            },
        ),
        BybitHistorySpec(
            name="funding_history",
            path="/v5/market/funding/history",
            step_ms=30 * _DAY_MS,
            params_builder=lambda category, symbol, start, end: {
                "category": category,
                "symbol": symbol,
                "startTime": start,
                "endTime": end,
                "limit": 200,
            },
        ),
        BybitHistorySpec(
            name=f"open_interest_{oi_interval}",
            path="/v5/market/open-interest",
            step_ms=int(_OI_INTERVAL_MS[oi_interval] * (oi_points_per_page - 1)),
            params_builder=lambda category, symbol, start, end: {
                "category": category,
                "symbol": symbol,
                "intervalTime": oi_interval,
                "startTime": start,
                "endTime": end,
                "limit": oi_points_per_page,
            },
        ),
        BybitHistorySpec(
            name=f"account_ratio_{account_ratio_period}",
            path="/v5/market/account-ratio",
            step_ms=int(_RATIO_INTERVAL_MS[account_ratio_period] * (ratio_points_per_page - 1)),
            params_builder=lambda category, symbol, start, end: {
                "category": category,
                "symbol": symbol,
                "period": account_ratio_period,
                "startTime": start,
                "endTime": end,
                "limit": ratio_points_per_page,
            },
        ),
    ]


def _iter_windows(start_ms: int, end_ms: int, *, step_ms: int) -> Iterable[tuple[int, int]]:
    cursor = int(start_ms)
    end_ms = int(end_ms)
    while cursor <= end_ms:
        window_end = min(cursor + step_ms - 1, end_ms)
        yield cursor, window_end
        cursor = window_end + 1


def _dataset_requested(requested: list[str] | None, spec_name: str) -> bool:
    if requested is None:
        return True
    if spec_name in requested:
        return True
    if spec_name.startswith("open_interest_") and "open_interest" in requested:
        return True
    if spec_name.startswith("account_ratio_") and "account_ratio" in requested:
        return True
    return False


async def backfill_bybit_rest_history(
    *,
    writer: RawEventWriter,
    symbol: str,
    category: str,
    start_ms: int,
    end_ms: int,
    datasets: list[str] | None = None,
    oi_interval: str = "5min",
    account_ratio_period: str = "5min",
    testnet: bool = False,
    timeout_s: float = 20.0,
    sleep_s: float = 0.0,
    client: httpx.AsyncClient | None = None,
) -> dict[str, int]:
    specs = default_bybit_history_specs(oi_interval=oi_interval, account_ratio_period=account_ratio_period)
    selected = [spec for spec in specs if _dataset_requested(datasets, spec.name)]
    counts = {spec.name: 0 for spec in selected}
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(base_url=BYBIT_REST_BASE[testnet], timeout=timeout_s)
    try:
        for spec in selected:
            for window_start, window_end in _iter_windows(start_ms, end_ms, step_ms=spec.step_ms):
                params = spec.params_builder(category, symbol, window_start, window_end)
                response = await client.get(spec.path, params=params)
                response.raise_for_status()
                payload = response.json()
                if payload.get("retCode", 0) != 0:
                    raise RuntimeError(f"Bybit backfill error for {spec.name}: {payload}")
                event = RawEvent(
                    source="bybit_rest",
                    topic=f"rest.{spec.name}.{symbol}",
                    symbol=symbol,
                    exchange_ts=int(window_end),
                    local_received_ts=int(time.time() * 1000),
                    payload=payload,
                    metadata={"path": spec.path, "params": params, "backfill": True},
                )
                writer.write(event)
                counts[spec.name] += 1
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
    finally:
        if owns_client:
            await client.aclose()
    return counts
