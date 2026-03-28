from __future__ import annotations

import logging

from btc_alert_engine.collectors.base import BaseWebSocketCollector
from btc_alert_engine.storage.raw_ndjson import RawEventWriter

BYBIT_PUBLIC_URLS = {
    (False, "linear"): "wss://stream.bybit.com/v5/public/linear",
    (True, "linear"): "wss://stream-testnet.bybit.com/v5/public/linear",
    (False, "inverse"): "wss://stream.bybit.com/v5/public/inverse",
    (True, "inverse"): "wss://stream-testnet.bybit.com/v5/public/inverse",
    (False, "spot"): "wss://stream.bybit.com/v5/public/spot",
    (True, "spot"): "wss://stream-testnet.bybit.com/v5/public/spot",
}


class BybitPublicTopicCollector(BaseWebSocketCollector):
    def __init__(
        self,
        *,
        category: str,
        testnet: bool,
        topics: list[str],
        symbol: str,
        writer: RawEventWriter,
        logger: logging.Logger | None = None,
    ) -> None:
        url = BYBIT_PUBLIC_URLS[(testnet, category)]
        super().__init__(
            url=url,
            topics=topics,
            symbol=symbol,
            source="bybit_ws",
            writer=writer,
            logger=logger,
        )


class BybitOrderBookCollector(BybitPublicTopicCollector):
    def __init__(
        self,
        *,
        symbol: str,
        depth: int = 200,
        category: str = "linear",
        testnet: bool = False,
        writer: RawEventWriter,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            category=category,
            testnet=testnet,
            topics=[f"orderbook.{depth}.{symbol}"],
            symbol=symbol,
            writer=writer,
            logger=logger,
        )


class BybitTradeCollector(BybitPublicTopicCollector):
    def __init__(
        self,
        *,
        symbol: str,
        category: str = "linear",
        testnet: bool = False,
        writer: RawEventWriter,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            category=category,
            testnet=testnet,
            topics=[f"publicTrade.{symbol}"],
            symbol=symbol,
            writer=writer,
            logger=logger,
        )


class BybitLiquidationCollector(BybitPublicTopicCollector):
    def __init__(
        self,
        *,
        symbol: str,
        category: str = "linear",
        testnet: bool = False,
        writer: RawEventWriter,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            category=category,
            testnet=testnet,
            topics=[f"allLiquidation.{symbol}"],
            symbol=symbol,
            writer=writer,
            logger=logger,
        )
