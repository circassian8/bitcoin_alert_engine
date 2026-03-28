from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import websockets

from btc_alert_engine.schemas import RawEvent
from btc_alert_engine.storage.raw_ndjson import RawEventWriter


@dataclass(slots=True)
class CollectorContext:
    symbol: str
    source: str


class BaseCollector(abc.ABC):
    def __init__(self, writer: RawEventWriter, logger: logging.Logger | None = None) -> None:
        self.writer = writer
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    async def run(self) -> None:
        raise NotImplementedError


class BaseWebSocketCollector(BaseCollector):
    def __init__(
        self,
        *,
        url: str,
        topics: list[str],
        symbol: str,
        source: str,
        writer: RawEventWriter,
        logger: logging.Logger | None = None,
        heartbeat_interval_s: float = 20.0,
    ) -> None:
        super().__init__(writer=writer, logger=logger)
        self.url = url
        self.topics = topics
        self.context = CollectorContext(symbol=symbol, source=source)
        self.heartbeat_interval_s = heartbeat_interval_s
        self._stop = asyncio.Event()

    async def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.logger.info("connecting", extra={"url": self.url, "topics": self.topics})
                async with websockets.connect(
                    self.url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    max_size=None,
                ) as ws:
                    await self._subscribe(ws)
                    backoff = 1.0
                    consumer = asyncio.create_task(self._consume(ws))
                    pinger = asyncio.create_task(self._heartbeat(ws))
                    done, pending = await asyncio.wait(
                        {consumer, pinger},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network path
                self.logger.exception("collector error; reconnecting", exc_info=exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _subscribe(self, ws: websockets.ClientConnection) -> None:
        payload = {"op": "subscribe", "args": self.topics}
        await ws.send(json.dumps(payload))

    async def _heartbeat(self, ws: websockets.ClientConnection) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.heartbeat_interval_s)
            await ws.send(json.dumps({"op": "ping"}))

    async def _consume(self, ws: websockets.ClientConnection) -> None:
        while not self._stop.is_set():
            message = await ws.recv()
            local_ts = int(time.time() * 1000)
            payload = json.loads(message)
            event = self.build_raw_event(payload=payload, local_received_ts=local_ts)
            if event is not None:
                self.writer.write(event)

    def build_raw_event(self, *, payload: dict[str, Any], local_received_ts: int) -> RawEvent | None:
        topic = str(payload.get("topic") or payload.get("op") or payload.get("type") or "control")
        symbol = self._infer_symbol(topic=topic, payload=payload)
        exchange_ts = payload.get("ts") or payload.get("creationTime")
        connection_id = payload.get("conn_id") or payload.get("connId")
        return RawEvent(
            source=self.context.source,
            topic=topic,
            symbol=symbol,
            exchange_ts=int(exchange_ts) if exchange_ts is not None else None,
            local_received_ts=local_received_ts,
            payload=payload,
            connection_id=connection_id,
        )

    def _infer_symbol(self, *, topic: str, payload: dict[str, Any]) -> str:
        if "." in topic:
            return topic.split(".")[-1]
        data = payload.get("data")
        if isinstance(data, dict) and "s" in data:
            return str(data["s"])
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return str(first.get("s") or self.context.symbol)
        return self.context.symbol
