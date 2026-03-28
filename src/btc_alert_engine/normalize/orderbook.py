from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from btc_alert_engine.normalize.bybit_public import compute_book_imbalance
from btc_alert_engine.schemas import BookLevel, BybitOrderBookMessage, TopOfBookState


class OrderBookSequenceError(ValueError):
    """Base class for order-book sequencing failures."""


class OrderBookOutOfOrderError(OrderBookSequenceError):
    """Raised when an update moves backwards or repeats a sequence/update id."""


class OrderBookGapError(OrderBookSequenceError):
    """Raised when an order-book delta stream skips a sequence/update id."""


class OrderBookStateError(OrderBookSequenceError):
    """Raised when a delta arrives before the initial snapshot."""


@dataclass(slots=True)
class BybitOrderBookBuilder:
    symbol: str
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    last_update_id: int | None = None
    last_seq: int | None = None
    last_cts: int | None = None

    def apply(self, message: BybitOrderBookMessage) -> None:
        if message.symbol != self.symbol:
            raise ValueError(f"Builder symbol {self.symbol} cannot consume {message.symbol}")
        if message.message_type == "snapshot" or message.update_id == 1:
            self._reset_from_snapshot(message)
            return
        self._validate_delta_sequence(message)
        for level in message.bids:
            self._apply_side(self.bids, level.price, level.size)
        for level in message.asks:
            self._apply_side(self.asks, level.price, level.size)
        self.last_update_id = message.update_id
        self.last_seq = message.seq
        self.last_cts = message.cts

    def _validate_delta_sequence(self, message: BybitOrderBookMessage) -> None:
        if self.last_update_id is None:
            raise OrderBookStateError("Delta received before snapshot")
        if message.update_id <= self.last_update_id:
            raise OrderBookOutOfOrderError(
                f"Update id moved backwards or repeated: last={self.last_update_id}, got={message.update_id}"
            )
        expected_update_id = self.last_update_id + 1
        if message.update_id != expected_update_id:
            raise OrderBookGapError(
                f"Order-book update gap detected: expected update_id {expected_update_id}, got {message.update_id}"
            )

        if self.last_seq is not None and message.seq is not None:
            if message.seq <= self.last_seq:
                raise OrderBookOutOfOrderError(f"Sequence moved backwards or repeated: last={self.last_seq}, got={message.seq}")
            expected_seq = self.last_seq + 1
            if message.seq != expected_seq:
                raise OrderBookGapError(f"Order-book sequence gap detected: expected seq {expected_seq}, got {message.seq}")

    def _reset_from_snapshot(self, message: BybitOrderBookMessage) -> None:
        self.bids = {level.price: level.size for level in message.bids if level.size != 0}
        self.asks = {level.price: level.size for level in message.asks if level.size != 0}
        self.last_update_id = message.update_id
        self.last_seq = message.seq
        self.last_cts = message.cts

    @staticmethod
    def _apply_side(side: dict[Decimal, Decimal], price: Decimal, size: Decimal) -> None:
        if size == 0:
            side.pop(price, None)
        else:
            side[price] = size

    def best_bid(self) -> tuple[Decimal, Decimal] | None:
        if not self.bids:
            return None
        price = max(self.bids)
        return price, self.bids[price]

    def best_ask(self) -> tuple[Decimal, Decimal] | None:
        if not self.asks:
            return None
        price = min(self.asks)
        return price, self.asks[price]

    def top_levels(self, *, side: str, limit: int | None = None) -> list[BookLevel]:
        if side not in {"bid", "ask"}:
            raise ValueError(side)
        if side == "bid":
            items = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)
        else:
            items = sorted(self.asks.items(), key=lambda kv: kv[0])
        if limit is not None:
            items = items[:limit]
        return [BookLevel(price=price, size=size) for price, size in items]

    def depth_usd(self, *, limit: int = 10, side: str | None = None) -> Decimal:
        total = Decimal("0")
        sides = [side] if side else ["bid", "ask"]
        for side_name in sides:
            for level in self.top_levels(side=side_name, limit=limit):
                total += level.price * level.size
        return total

    def depth_decay(self, *, limit: int = 10) -> Decimal:
        top1 = self.depth_usd(limit=1)
        topn = self.depth_usd(limit=limit)
        if topn == 0:
            return Decimal("0")
        return top1 / topn

    def book_imbalance(self, *, depth: int = 10) -> Decimal:
        bids = self.top_levels(side="bid", limit=depth)
        asks = self.top_levels(side="ask", limit=depth)
        return compute_book_imbalance(bids, asks, depth=depth)

    def top_of_book(self, ts: int) -> TopOfBookState:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        spread = None
        if best_bid and best_ask:
            spread = best_ask[0] - best_bid[0]
        return TopOfBookState(
            ts=ts,
            symbol=self.symbol,
            best_bid_price=best_bid[0] if best_bid else None,
            best_bid_size=best_bid[1] if best_bid else None,
            best_ask_price=best_ask[0] if best_ask else None,
            best_ask_size=best_ask[1] if best_ask else None,
            spread=spread,
            update_id=self.last_update_id,
            seq=self.last_seq,
        )
