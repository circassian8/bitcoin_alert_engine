from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable

from btc_alert_engine.normalize.bybit_public import parse_liquidation_payload, parse_orderbook_payload, parse_trade_payload
from btc_alert_engine.normalize.bybit_rest import latest_by_timestamp, parse_kline_event
from btc_alert_engine.normalize.orderbook import BybitOrderBookBuilder
from btc_alert_engine.schemas import MicroBucket1s, PriceBar, RawEvent
from btc_alert_engine.storage.raw_ndjson import iter_raw_events_sorted


def _event_ts(event: RawEvent) -> int:
    return event.exchange_ts if event.exchange_ts is not None else event.local_received_ts


@dataclass(slots=True)
class _MicroBucketAccumulator:
    start_ms: int
    symbol: str
    trade_prices: list[float] = field(default_factory=list)
    first_trade_price: float | None = None
    trade_high: float | None = None
    trade_low: float | None = None
    trade_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    long_liq_size: float = 0.0
    short_liq_size: float = 0.0
    long_liq_notional: float = 0.0
    short_liq_notional: float = 0.0
    last_trade_price: float | None = None
    ofi_proxy: float = 0.0
    replenish_notional: float = 0.0
    cancel_notional: float = 0.0
    quote_updates: int = 0
    _book_samples: int = 0
    _mid_sum: float = 0.0
    _spread_sum: float = 0.0
    _spread_bps_sum: float = 0.0
    _bookimb1_sum: float = 0.0
    _bookimb5_sum: float = 0.0
    _bookimb10_sum: float = 0.0
    _top10_depth_sum: float = 0.0
    _depth_decay_sum: float = 0.0
    _last_best_bid_price: float | None = None
    _last_best_bid_size: float | None = None
    _last_best_ask_price: float | None = None
    _last_best_ask_size: float | None = None
    _best_bid_low: float | None = None
    _best_bid_high: float | None = None
    _best_ask_low: float | None = None
    _best_ask_high: float | None = None

    def ingest_trade(self, trade_ts: int, side: str, price: Decimal, size: Decimal) -> None:
        price_f = float(price)
        size_f = float(size)
        notional = price_f * size_f
        self.trade_count += 1
        self.trade_prices.append(price_f)
        if self.first_trade_price is None:
            self.first_trade_price = price_f
        self.trade_high = price_f if self.trade_high is None else max(self.trade_high, price_f)
        self.trade_low = price_f if self.trade_low is None else min(self.trade_low, price_f)
        self.last_trade_price = price_f
        if side == "Buy":
            self.buy_volume += size_f
            self.buy_notional += notional
        else:
            self.sell_volume += size_f
            self.sell_notional += notional

    def ingest_liquidation(self, side: str, price: Decimal, size: Decimal) -> None:
        price_f = float(price)
        size_f = float(size)
        notional = price_f * size_f
        # On Bybit's liquidation stream, sell-side liquidations are forced sells,
        # which we treat as long liquidations. Buy-side liquidations are treated as shorts.
        if side == "Sell":
            self.long_liq_size += size_f
            self.long_liq_notional += notional
        else:
            self.short_liq_size += size_f
            self.short_liq_notional += notional

    def ingest_orderbook(self, builder: BybitOrderBookBuilder, *, payload_type: str, bids: list[tuple[Decimal, Decimal]], asks: list[tuple[Decimal, Decimal]], raw_message: object) -> None:
        if payload_type == "delta":
            self.quote_updates += 1
            for price, new_size in bids:
                old_size = builder.bids.get(price, Decimal("0"))
                diff = new_size - old_size
                notional = float(abs(diff) * price)
                if diff > 0:
                    self.replenish_notional += notional
                elif diff < 0:
                    self.cancel_notional += notional
                self.ofi_proxy += float(diff * price)
            for price, new_size in asks:
                old_size = builder.asks.get(price, Decimal("0"))
                diff = new_size - old_size
                notional = float(abs(diff) * price)
                if diff > 0:
                    self.replenish_notional += notional
                elif diff < 0:
                    self.cancel_notional += notional
                self.ofi_proxy += float(-diff * price)
        builder.apply(raw_message)  # type: ignore[arg-type]
        self.capture_book(builder)

    def capture_book(self, builder: BybitOrderBookBuilder) -> None:
        top = builder.top_of_book(ts=self.start_ms)
        if not top.has_book:
            return
        bid = float(top.best_bid_price)
        ask = float(top.best_ask_price)
        bid_size = float(top.best_bid_size)
        ask_size = float(top.best_ask_size)
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = (spread / mid) * 10_000.0 if mid else 0.0
        bookimb1 = float(builder.book_imbalance(depth=1))
        bookimb5 = float(builder.book_imbalance(depth=5))
        bookimb10 = float(builder.book_imbalance(depth=10))
        top10_depth = float(builder.depth_usd(limit=10))
        depth_decay = float(builder.depth_decay(limit=10))

        self._book_samples += 1
        self._mid_sum += mid
        self._spread_sum += spread
        self._spread_bps_sum += spread_bps
        self._bookimb1_sum += bookimb1
        self._bookimb5_sum += bookimb5
        self._bookimb10_sum += bookimb10
        self._top10_depth_sum += top10_depth
        self._depth_decay_sum += depth_decay
        self._last_best_bid_price = bid
        self._last_best_bid_size = bid_size
        self._last_best_ask_price = ask
        self._last_best_ask_size = ask_size
        self._best_bid_low = bid if self._best_bid_low is None else min(self._best_bid_low, bid)
        self._best_bid_high = bid if self._best_bid_high is None else max(self._best_bid_high, bid)
        self._best_ask_low = ask if self._best_ask_low is None else min(self._best_ask_low, ask)
        self._best_ask_high = ask if self._best_ask_high is None else max(self._best_ask_high, ask)

    def to_bucket(self, builder: BybitOrderBookBuilder) -> MicroBucket1s:
        if self._book_samples == 0:
            self.capture_book(builder)
        total_volume = self.buy_volume + self.sell_volume
        total_notional = self.buy_notional + self.sell_notional
        avg_mid = self._mid_sum / self._book_samples if self._book_samples else None
        spread = self._spread_sum / self._book_samples if self._book_samples else None
        spread_bps = self._spread_bps_sum / self._book_samples if self._book_samples else None
        vwap = (total_notional / total_volume) if total_volume > 0 else None
        vwap_mid_dev = ((vwap - avg_mid) / avg_mid) if (vwap is not None and avg_mid not in {None, 0.0}) else 0.0
        micro_vol = 0.0
        if len(self.trade_prices) >= 2:
            log_returns = [math.log(self.trade_prices[i] / self.trade_prices[i - 1]) for i in range(1, len(self.trade_prices)) if self.trade_prices[i - 1] > 0]
            if log_returns:
                micro_vol = statistics.pstdev(log_returns)
        replenish_rate = self.replenish_notional / self.cancel_notional if self.cancel_notional > 0 else 0.0
        cancel_add_ratio = self.cancel_notional / self.replenish_notional if self.replenish_notional > 0 else 0.0
        return MicroBucket1s(
            ts=self.start_ms,
            symbol=self.symbol,
            best_bid_price=self._last_best_bid_price,
            best_bid_size=self._last_best_bid_size,
            best_ask_price=self._last_best_ask_price,
            best_ask_size=self._last_best_ask_size,
            best_bid_low=self._best_bid_low,
            best_bid_high=self._best_bid_high,
            best_ask_low=self._best_ask_low,
            best_ask_high=self._best_ask_high,
            mid_price=avg_mid,
            spread=spread,
            spread_bps=spread_bps,
            trade_count=self.trade_count,
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            buy_notional=self.buy_notional,
            sell_notional=self.sell_notional,
            vwap=vwap,
            vwap_mid_dev=vwap_mid_dev,
            first_trade_price=self.first_trade_price,
            last_trade_price=self.last_trade_price,
            trade_low=self.trade_low,
            trade_high=self.trade_high,
            cum_delta=self.buy_volume - self.sell_volume,
            ofi_proxy=self.ofi_proxy,
            bookimb_l1=(self._bookimb1_sum / self._book_samples) if self._book_samples else None,
            bookimb_l5=(self._bookimb5_sum / self._book_samples) if self._book_samples else None,
            bookimb_l10=(self._bookimb10_sum / self._book_samples) if self._book_samples else None,
            top10_depth_usd=(self._top10_depth_sum / self._book_samples) if self._book_samples else None,
            depth_decay=(self._depth_decay_sum / self._book_samples) if self._book_samples else None,
            replenish_notional=self.replenish_notional,
            cancel_notional=self.cancel_notional,
            replenish_rate=replenish_rate,
            cancel_add_ratio=cancel_add_ratio,
            micro_vol=micro_vol,
            long_liq_size=self.long_liq_size,
            short_liq_size=self.short_liq_size,
            long_liq_notional=self.long_liq_notional,
            short_liq_notional=self.short_liq_notional,
            quote_updates=self.quote_updates,
        )


def materialize_bybit_bars(paths: Iterable[str], *, symbol: str) -> dict[str, list[PriceBar]]:
    by_kind: dict[str, dict[int, PriceBar]] = {
        "trade_15m": {},
        "index_price_15m": {},
        "premium_index_15m": {},
    }
    for event in iter_raw_events_sorted(paths):
        if event.source != "bybit_rest" or event.symbol != symbol:
            continue
        if event.topic.startswith("rest.kline_"):
            kind = "trade_15m"
        elif event.topic.startswith("rest.index_price_kline_"):
            kind = "index_price_15m"
        elif event.topic.startswith("rest.premium_index_price_kline_"):
            kind = "premium_index_15m"
        else:
            continue
        for bar in parse_kline_event(event):
            by_kind[kind][bar.ts] = bar
    return {kind: latest_by_timestamp(rows.values()) for kind, rows in by_kind.items() if rows}


def materialize_micro_buckets(paths: Iterable[str], *, symbol: str) -> list[MicroBucket1s]:
    builder = BybitOrderBookBuilder(symbol=symbol)
    events = [
        event
        for event in iter_raw_events_sorted(paths)
        if event.source == "bybit_ws"
        and event.symbol == symbol
        and (
            event.topic.startswith("orderbook.")
            or event.topic.startswith("publicTrade.")
            or event.topic.startswith("allLiquidation.")
        )
    ]
    if not events:
        return []

    current_bucket_ts = (_event_ts(events[0]) // 1000) * 1000
    current = _MicroBucketAccumulator(start_ms=current_bucket_ts, symbol=symbol)
    buckets: list[MicroBucket1s] = []

    for event in events:
        ts_bucket = (_event_ts(event) // 1000) * 1000
        while ts_bucket > current_bucket_ts:
            buckets.append(current.to_bucket(builder))
            current_bucket_ts += 1000
            current = _MicroBucketAccumulator(start_ms=current_bucket_ts, symbol=symbol)

        if event.topic.startswith("orderbook."):
            message = parse_orderbook_payload(event.payload)
            current.ingest_orderbook(
                builder,
                payload_type=message.message_type,
                bids=[(level.price, level.size) for level in message.bids],
                asks=[(level.price, level.size) for level in message.asks],
                raw_message=message,
            )
        elif event.topic.startswith("publicTrade."):
            for trade in parse_trade_payload(event.payload):
                if trade.symbol != symbol:
                    continue
                current.ingest_trade(trade.ts, trade.side, trade.price, trade.size)
        elif event.topic.startswith("allLiquidation."):
            for liq in parse_liquidation_payload(event.payload):
                if liq.symbol != symbol:
                    continue
                current.ingest_liquidation(liq.side, liq.bankruptcy_price, liq.size)

    buckets.append(current.to_bucket(builder))
    return buckets
