from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from btc_alert_engine.collectors.base import BaseCollector
from btc_alert_engine.schemas import RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter

BYBIT_REST_BASE = {
    False: "https://api.bybit.com",
    True: "https://api-testnet.bybit.com",
}

ParamsBuilder = Callable[[str, str], dict[str, Any]]


@dataclass(slots=True)
class PollSpec:
    name: str
    path: str
    params_builder: ParamsBuilder
    interval_s: float


def _now_ms() -> int:
    return int(time.time() * 1000)


def _recent_window(hours: int = 24) -> tuple[int, int]:
    end = _now_ms()
    start = end - (hours * 60 * 60 * 1000)
    return start, end


def default_specs() -> list[PollSpec]:
    return [
        PollSpec(
            name="funding_history",
            path="/v5/market/funding/history",
            params_builder=lambda category, symbol: {"category": category, "symbol": symbol, "limit": 200},
            interval_s=300,
        ),
        PollSpec(
            name="open_interest_5min",
            path="/v5/market/open-interest",
            params_builder=lambda category, symbol: {
                "category": category,
                "symbol": symbol,
                "intervalTime": "5min",
                "limit": 200,
            },
            interval_s=300,
        ),
        PollSpec(
            name="account_ratio_5min",
            path="/v5/market/account-ratio",
            params_builder=lambda category, symbol: {
                "category": category,
                "symbol": symbol,
                "period": "5min",
                "limit": 200,
            },
            interval_s=300,
        ),
        PollSpec(
            name="tickers",
            path="/v5/market/tickers",
            params_builder=lambda category, symbol: {"category": category, "symbol": symbol},
            interval_s=15,
        ),
        PollSpec(
            name="instruments_info",
            path="/v5/market/instruments-info",
            params_builder=lambda category, symbol: {"category": category, "symbol": symbol},
            interval_s=3600,
        ),
        PollSpec(
            name="kline_15",
            path="/v5/market/kline",
            params_builder=lambda category, symbol: {
                "category": category,
                "symbol": symbol,
                "interval": "15",
                "limit": 200,
            },
            interval_s=60,
        ),
        PollSpec(
            name="index_price_kline_15",
            path="/v5/market/index-price-kline",
            params_builder=lambda category, symbol: {
                "category": category,
                "symbol": symbol,
                "interval": "15",
                "limit": 200,
            },
            interval_s=60,
        ),
        PollSpec(
            name="premium_index_price_kline_15",
            path="/v5/market/premium-index-price-kline",
            params_builder=lambda category, symbol: {
                "category": category,
                "symbol": symbol,
                "interval": "15",
                "limit": 200,
            },
            interval_s=60,
        ),
    ]


class BybitMarketRestPoller(BaseCollector):
    def __init__(
        self,
        *,
        symbol: str,
        category: str,
        testnet: bool,
        writer: RawEventWriter,
        logger: logging.Logger | None = None,
        specs: list[PollSpec] | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        super().__init__(writer=writer, logger=logger)
        self.symbol = symbol
        self.category = category
        self.base_url = BYBIT_REST_BASE[testnet]
        self.specs = specs or default_specs()
        self.timeout_s = timeout_s
        self._stop = asyncio.Event()

    async def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
            tasks = [asyncio.create_task(self._run_spec(client, spec)) for spec in self.specs]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()

    async def _run_spec(self, client: httpx.AsyncClient, spec: PollSpec) -> None:
        while not self._stop.is_set():
            started = _now_ms()
            try:
                params = spec.params_builder(self.category, self.symbol)
                response = await client.get(spec.path, params=params)
                response.raise_for_status()
                payload = response.json()
                event = RawEvent(
                    source="bybit_rest",
                    topic=f"rest.{spec.name}.{self.symbol}",
                    symbol=self.symbol,
                    exchange_ts=int(payload.get("time")) if payload.get("time") is not None else started,
                    local_received_ts=_now_ms(),
                    payload=payload,
                    metadata={"path": spec.path, "params": params},
                )
                self.writer.write(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network path
                self.logger.exception("REST poller error for %s", spec.name, exc_info=exc)
            await asyncio.sleep(spec.interval_s)
