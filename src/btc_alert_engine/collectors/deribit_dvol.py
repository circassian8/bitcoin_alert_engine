from __future__ import annotations

import asyncio
import itertools
import logging
import time
from typing import Any

import httpx

from btc_alert_engine.collectors.base import BaseCollector
from btc_alert_engine.schemas import RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter

DERIBIT_HTTP_BASE = {
    False: "https://www.deribit.com/api/v2",
    True: "https://test.deribit.com/api/v2",
}


class DeribitVolatilityIndexPoller(BaseCollector):
    """Polls Deribit volatility index candles over JSON-RPC HTTP.

    The poller requests a rolling window and stores the raw JSON-RPC result in the raw event store.
    """

    def __init__(
        self,
        *,
        currency: str,
        resolution: str,
        interval_s: float,
        testnet: bool,
        writer: RawEventWriter,
        logger: logging.Logger | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        super().__init__(writer=writer, logger=logger)
        self.currency = currency
        self.resolution = resolution
        self.interval_s = interval_s
        self.base_url = DERIBIT_HTTP_BASE[testnet]
        self.timeout_s = timeout_s
        self._stop = asyncio.Event()
        self._request_ids = itertools.count(1)

    async def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
            while not self._stop.is_set():
                now = int(time.time() * 1000)
                start = now - (24 * 60 * 60 * 1000)
                rpc_payload: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": next(self._request_ids),
                    "method": "public/get_volatility_index_data",
                    "params": {
                        "currency": self.currency,
                        "start_timestamp": start,
                        "end_timestamp": now,
                        "resolution": self.resolution,
                    },
                }
                try:
                    response = await client.post("/public/get_volatility_index_data", json=rpc_payload)
                    response.raise_for_status()
                    payload = response.json()
                    event = RawEvent(
                        source="deribit_http",
                        topic=f"deribit.volatility_index.{self.currency}.{self.resolution}",
                        symbol=self.currency,
                        exchange_ts=now,
                        local_received_ts=int(time.time() * 1000),
                        payload=payload,
                    )
                    self.writer.write(event)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - network path
                    self.logger.exception("Deribit DVOL poll error", exc_info=exc)
                await asyncio.sleep(self.interval_s)
