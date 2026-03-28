from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any

import httpx

from btc_alert_engine.collectors.deribit_dvol import DERIBIT_HTTP_BASE
from btc_alert_engine.schemas import RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter

_DAY_MS = 24 * 60 * 60 * 1000


def _iter_windows(start_ms: int, end_ms: int, *, step_ms: int) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    cursor = int(start_ms)
    end_ms = int(end_ms)
    while cursor <= end_ms:
        window_end = min(cursor + step_ms - 1, end_ms)
        windows.append((cursor, window_end))
        cursor = window_end + 1
    return windows


async def backfill_deribit_dvol_history(
    *,
    writer: RawEventWriter,
    currency: str,
    resolution: str,
    start_ms: int,
    end_ms: int,
    testnet: bool = False,
    timeout_s: float = 20.0,
    sleep_s: float = 0.0,
    step_ms: int = 30 * _DAY_MS,
    client: httpx.AsyncClient | None = None,
) -> int:
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(base_url=DERIBIT_HTTP_BASE[testnet], timeout=timeout_s)
    request_ids = itertools.count(1)
    count = 0
    try:
        for window_start, window_end in _iter_windows(start_ms, end_ms, step_ms=step_ms):
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": next(request_ids),
                "method": "public/get_volatility_index_data",
                "params": {
                    "currency": currency,
                    "start_timestamp": window_start,
                    "end_timestamp": window_end,
                    "resolution": resolution,
                },
            }
            response = await client.post("/public/get_volatility_index_data", json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get("error"):
                raise RuntimeError(f"Deribit DVOL backfill error: {result}")
            event = RawEvent(
                source="deribit_http",
                topic=f"deribit.volatility_index.{currency}.{resolution}",
                symbol=currency,
                exchange_ts=int(window_end),
                local_received_ts=int(time.time() * 1000),
                payload=result,
                metadata={"backfill": True, "params": payload["params"]},
            )
            writer.write(event)
            count += 1
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
    finally:
        if owns_client:
            await client.aclose()
    return count
