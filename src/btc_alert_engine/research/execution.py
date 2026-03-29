from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from btc_alert_engine.normalize.bybit_public import parse_orderbook_payload, parse_trade_payload
from btc_alert_engine.normalize.orderbook import (
    BybitOrderBookBuilder,
    OrderBookGapError,
    OrderBookOutOfOrderError,
    OrderBookStateError,
)
from btc_alert_engine.profiles import profile_for_generator
from btc_alert_engine.schemas import CandidateEvent, MicroBucket1s, PriceBar
from btc_alert_engine.storage.raw_ndjson import iter_raw_events_sorted

_BUCKET_MS = 1_000
_DEFAULT_BAR_MS = 900_000
_TP1_WINDOW_MINUTES = 12 * 60


@dataclass(slots=True)
class EntryDecision:
    entry_ts: int
    executed_entry: float
    entry_source: Literal["raw_quote", "micro_quote", "bar_open"]
    raw_index: int | None = None


@dataclass(slots=True)
class PathResult:
    outcome: Literal["tp", "sl", "timeout"]
    tp_before_sl: bool
    tp1_before_sl: bool
    mfe_r: float
    mae_r: float
    net_r_timeout: float
    minutes_to_tp_or_sl: int | None
    exit_ts: int | None
    executed_exit: float | None
    path_source: Literal["raw_events", "micro_buckets", "trade_bars"]


@dataclass(slots=True)
class QuoteRecoveryStats:
    gap_count: int = 0
    out_of_order_count: int = 0
    state_error_count: int = 0


def bars_to_frame(bars: Iterable[PriceBar]) -> pd.DataFrame:
    rows = [bar.model_dump(mode="json") if isinstance(bar, PriceBar) else bar for bar in bars]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)


def micro_to_frame(micro_buckets: Iterable[MicroBucket1s]) -> pd.DataFrame:
    rows = [bucket.model_dump(mode="json") if isinstance(bucket, MicroBucket1s) else bucket for bucket in micro_buckets]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    return df.reset_index(drop=True)


def build_raw_execution_tape(
    paths: Iterable[str | Path],
    *,
    symbol: str,
    tolerate_gaps: bool = True,
) -> tuple[pd.DataFrame, QuoteRecoveryStats]:
    builder = BybitOrderBookBuilder(symbol=symbol)
    rows: list[dict[str, float | int | str | None]] = []
    ordinal = 0
    waiting_for_snapshot = False
    stats = QuoteRecoveryStats()

    for event in iter_raw_events_sorted(paths):
        if event.symbol != symbol or event.source != "bybit_ws":
            continue
        if event.topic.startswith("orderbook."):
            message = parse_orderbook_payload(event.payload)
            try:
                if waiting_for_snapshot and message.message_type != "snapshot" and message.update_id != 1:
                    continue
                builder.apply(message)
                waiting_for_snapshot = False
            except OrderBookGapError:
                stats.gap_count += 1
                if not tolerate_gaps:
                    raise
                waiting_for_snapshot = True
                continue
            except OrderBookOutOfOrderError:
                stats.out_of_order_count += 1
                if not tolerate_gaps:
                    raise
                continue
            except OrderBookStateError:
                stats.state_error_count += 1
                if not tolerate_gaps:
                    raise
                waiting_for_snapshot = True
                continue
            top = builder.top_of_book(ts=message.cts or message.ts)
            if top.has_book:
                rows.append(
                    {
                        "ts": int(top.ts),
                        "ordinal": ordinal,
                        "kind": "quote",
                        "bid": float(top.best_bid_price) if top.best_bid_price is not None else None,
                        "ask": float(top.best_ask_price) if top.best_ask_price is not None else None,
                        "trade_price": None,
                    }
                )
                ordinal += 1
        elif event.topic.startswith("publicTrade."):
            for trade in parse_trade_payload(event.payload):
                if trade.symbol != symbol:
                    continue
                rows.append(
                    {
                        "ts": int(trade.ts),
                        "ordinal": ordinal,
                        "kind": "trade",
                        "bid": None,
                        "ask": None,
                        "trade_price": float(trade.price),
                    }
                )
                ordinal += 1
    if not rows:
        return pd.DataFrame(columns=["ts", "ordinal", "kind", "bid", "ask", "trade_price"]), stats
    frame = pd.DataFrame(rows)
    return frame.sort_values(["ts", "ordinal"]).reset_index(drop=True), stats


def _fill_price_from_quote(side: str, *, bid: float | None, ask: float | None, slippage_bps: float) -> float | None:
    if side == "long":
        if ask is None or math.isnan(ask):
            return None
        return float(ask) * (1.0 + slippage_bps / 10_000.0)
    if bid is None or math.isnan(bid):
        return None
    return float(bid) * (1.0 - slippage_bps / 10_000.0)


def resolve_entry(
    candidate: CandidateEvent,
    trade_bars: pd.DataFrame,
    *,
    raw_tape: pd.DataFrame | None = None,
    micro_frame: pd.DataFrame | None = None,
    slippage_bps: float = 1.0,
    latency_ms: int = 0,
) -> EntryDecision | None:
    signal_ts = int(candidate.ts)
    eligible_ts = signal_ts + int(latency_ms)

    if raw_tape is not None and not raw_tape.empty:
        quotes = raw_tape[(raw_tape["kind"] == "quote") & (raw_tape["ts"] > eligible_ts)]
        for idx, row in quotes.iterrows():
            fill = _fill_price_from_quote(candidate.side, bid=row.get("bid"), ask=row.get("ask"), slippage_bps=slippage_bps)
            if fill is not None:
                return EntryDecision(entry_ts=int(row["ts"]), executed_entry=fill, entry_source="raw_quote", raw_index=int(idx))

    if micro_frame is not None and not micro_frame.empty:
        future = micro_frame[micro_frame["ts"] > eligible_ts]
        for _, row in future.iterrows():
            fill = _fill_price_from_quote(
                candidate.side,
                bid=float(row["best_bid_price"]) if pd.notna(row.get("best_bid_price")) else None,
                ask=float(row["best_ask_price"]) if pd.notna(row.get("best_ask_price")) else None,
                slippage_bps=slippage_bps,
            )
            if fill is not None:
                return EntryDecision(entry_ts=int(row["ts"]), executed_entry=fill, entry_source="micro_quote")

    future_bars = trade_bars.loc[trade_bars.index > pd.to_datetime(signal_ts, unit="ms", utc=True)]
    if future_bars.empty:
        return None
    first_bar = future_bars.iloc[0]
    first_ts = int(future_bars.index[0].timestamp() * 1000)
    open_px = float(first_bar["open"])
    if candidate.side == "long":
        fill = open_px * (1.0 + slippage_bps / 10_000.0)
    else:
        fill = open_px * (1.0 - slippage_bps / 10_000.0)
    return EntryDecision(entry_ts=first_ts, executed_entry=fill, entry_source="bar_open")


def _mark_price_for_side(side: str, *, bid: float | None = None, ask: float | None = None, trade_price: float | None = None) -> float | None:
    if side == "long":
        if bid is not None and not math.isnan(bid):
            return float(bid)
    else:
        if ask is not None and not math.isnan(ask):
            return float(ask)
    if trade_price is not None and not math.isnan(trade_price):
        return float(trade_price)
    return None


def _trigger_minutes_for_candidate(candidate: CandidateEvent) -> int:
    try:
        return profile_for_generator(candidate.module).trigger_minutes
    except Exception:
        return _DEFAULT_BAR_MS // 60_000


def _trigger_ms_for_candidate(candidate: CandidateEvent) -> int:
    return _trigger_minutes_for_candidate(candidate) * 60_000


def _tp1_window_bars(candidate: CandidateEvent) -> int:
    trigger_minutes = _trigger_minutes_for_candidate(candidate)
    return max(_TP1_WINDOW_MINUTES // trigger_minutes, 1)


def _update_excursions(side: str, observed_price: float | None, *, entry: float, risk: float, mfe_r: float, mae_r: float) -> tuple[float, float]:
    if observed_price is None:
        return mfe_r, mae_r
    if side == "long":
        mfe_r = max(mfe_r, (observed_price - entry) / risk)
        mae_r = max(mae_r, (entry - observed_price) / risk)
    else:
        mfe_r = max(mfe_r, (entry - observed_price) / risk)
        mae_r = max(mae_r, (observed_price - entry) / risk)
    return mfe_r, mae_r


def _barrier_hit(side: str, observed_price: float | None, *, stop: float, tp: float, tp1: float) -> tuple[bool, bool, bool]:
    if observed_price is None:
        return False, False, False
    if side == "long":
        return observed_price <= stop, observed_price >= tp, observed_price >= tp1
    return observed_price >= stop, observed_price <= tp, observed_price <= tp1


def simulate_with_raw_tape(
    candidate: CandidateEvent,
    *,
    entry: EntryDecision,
    raw_tape: pd.DataFrame,
    executed_tp: float,
    executed_tp1: float,
    stop: float,
    risk: float,
) -> PathResult | None:
    if raw_tape.empty:
        return None
    horizon_end_ts = int(candidate.ts) + candidate.timeout_bars * _trigger_ms_for_candidate(candidate)
    start_after = int(entry.raw_index) if entry.raw_index is not None else -1
    future = raw_tape[(raw_tape.index > start_after) & (raw_tape["ts"] > entry.entry_ts) & (raw_tape["ts"] <= horizon_end_ts)]
    if future.empty:
        return None

    mfe_r = 0.0
    mae_r = 0.0
    tp1_before_sl = False
    last_mark = entry.executed_entry
    last_ts = entry.entry_ts
    tp1_deadline = entry.entry_ts + (12 * 60 * 60 * 1000)

    for _, obs in future.iterrows():
        observed = _mark_price_for_side(
            candidate.side,
            bid=float(obs["bid"]) if pd.notna(obs.get("bid")) else None,
            ask=float(obs["ask"]) if pd.notna(obs.get("ask")) else None,
            trade_price=float(obs["trade_price"]) if pd.notna(obs.get("trade_price")) else None,
        )
        if observed is None:
            continue
        last_mark = observed
        last_ts = int(obs["ts"])
        mfe_r, mae_r = _update_excursions(candidate.side, observed, entry=entry.executed_entry, risk=risk, mfe_r=mfe_r, mae_r=mae_r)
        stop_hit, tp_hit, tp1_hit = _barrier_hit(candidate.side, observed, stop=stop, tp=executed_tp, tp1=executed_tp1)
        if tp1_hit and int(obs["ts"]) <= tp1_deadline:
            tp1_before_sl = True
        if stop_hit:
            minutes = max(1, math.ceil((int(obs["ts"]) - entry.entry_ts) / 60_000.0))
            return PathResult(
                outcome="sl",
                tp_before_sl=False,
                tp1_before_sl=tp1_before_sl,
                mfe_r=mfe_r,
                mae_r=mae_r,
                net_r_timeout=(last_mark - entry.executed_entry) / risk if candidate.side == "long" else (entry.executed_entry - last_mark) / risk,
                minutes_to_tp_or_sl=minutes,
                exit_ts=int(obs["ts"]),
                executed_exit=stop,
                path_source="raw_events",
            )
        if tp_hit:
            minutes = max(1, math.ceil((int(obs["ts"]) - entry.entry_ts) / 60_000.0))
            return PathResult(
                outcome="tp",
                tp_before_sl=True,
                tp1_before_sl=tp1_before_sl or int(obs["ts"]) <= tp1_deadline,
                mfe_r=mfe_r,
                mae_r=mae_r,
                net_r_timeout=(last_mark - entry.executed_entry) / risk if candidate.side == "long" else (entry.executed_entry - last_mark) / risk,
                minutes_to_tp_or_sl=minutes,
                exit_ts=int(obs["ts"]),
                executed_exit=executed_tp,
                path_source="raw_events",
            )

    net_r_timeout = (last_mark - entry.executed_entry) / risk if candidate.side == "long" else (entry.executed_entry - last_mark) / risk
    return PathResult(
        outcome="timeout",
        tp_before_sl=False,
        tp1_before_sl=tp1_before_sl,
        mfe_r=mfe_r,
        mae_r=mae_r,
        net_r_timeout=net_r_timeout,
        minutes_to_tp_or_sl=None,
        exit_ts=last_ts,
        executed_exit=last_mark,
        path_source="raw_events",
    )


def _bucket_extremes(row: pd.Series, side: str) -> tuple[float | None, float | None, float | None]:
    trade_low = float(row["trade_low"]) if pd.notna(row.get("trade_low")) else None
    trade_high = float(row["trade_high"]) if pd.notna(row.get("trade_high")) else None
    bid_low = float(row["best_bid_low"]) if pd.notna(row.get("best_bid_low")) else None
    bid_high = float(row["best_bid_high"]) if pd.notna(row.get("best_bid_high")) else None
    ask_low = float(row["best_ask_low"]) if pd.notna(row.get("best_ask_low")) else None
    ask_high = float(row["best_ask_high"]) if pd.notna(row.get("best_ask_high")) else None
    last_trade = float(row["last_trade_price"]) if pd.notna(row.get("last_trade_price")) else None
    best_bid = float(row["best_bid_price"]) if pd.notna(row.get("best_bid_price")) else None
    best_ask = float(row["best_ask_price"]) if pd.notna(row.get("best_ask_price")) else None

    if side == "long":
        favorable_candidates = [value for value in [bid_high, trade_high, best_bid, last_trade] if value is not None]
        adverse_candidates = [value for value in [bid_low, trade_low, best_bid, last_trade] if value is not None]
        favorable = max(favorable_candidates) if favorable_candidates else None
        adverse = min(adverse_candidates) if adverse_candidates else None
        timeout_mark = last_trade if last_trade is not None else best_bid
    else:
        favorable_candidates = [value for value in [ask_low, trade_low, best_ask, last_trade] if value is not None]
        adverse_candidates = [value for value in [ask_high, trade_high, best_ask, last_trade] if value is not None]
        favorable = min(favorable_candidates) if favorable_candidates else None
        adverse = max(adverse_candidates) if adverse_candidates else None
        timeout_mark = last_trade if last_trade is not None else best_ask
    return favorable, adverse, timeout_mark


def simulate_with_micro_buckets(
    candidate: CandidateEvent,
    *,
    entry: EntryDecision,
    micro_frame: pd.DataFrame,
    executed_tp: float,
    executed_tp1: float,
    stop: float,
    risk: float,
) -> PathResult | None:
    if micro_frame.empty:
        return None
    horizon_end_ts = int(candidate.ts) + candidate.timeout_bars * _trigger_ms_for_candidate(candidate)
    future = micro_frame[(micro_frame["ts"] > entry.entry_ts) & (micro_frame["ts"] <= horizon_end_ts)]
    if future.empty:
        return None

    mfe_r = 0.0
    mae_r = 0.0
    tp1_before_sl = False
    last_mark = entry.executed_entry
    last_ts = entry.entry_ts
    tp1_deadline = entry.entry_ts + (12 * 60 * 60 * 1000)

    for _, row in future.iterrows():
        favorable, adverse, timeout_mark = _bucket_extremes(row, candidate.side)
        if timeout_mark is not None:
            last_mark = timeout_mark
            last_ts = int(row["ts"])
        for observed in [favorable, adverse]:
            mfe_r, mae_r = _update_excursions(candidate.side, observed, entry=entry.executed_entry, risk=risk, mfe_r=mfe_r, mae_r=mae_r)

        if candidate.side == "long":
            tp_hit = favorable is not None and favorable >= executed_tp
            tp1_hit = favorable is not None and favorable >= executed_tp1
            stop_hit = adverse is not None and adverse <= stop
        else:
            tp_hit = favorable is not None and favorable <= executed_tp
            tp1_hit = favorable is not None and favorable <= executed_tp1
            stop_hit = adverse is not None and adverse >= stop
        if tp1_hit and int(row["ts"]) <= tp1_deadline:
            tp1_before_sl = True
        if tp_hit and stop_hit:
            minutes = max(1, math.ceil((int(row["ts"]) - entry.entry_ts) / 60_000.0))
            return PathResult(
                outcome="sl",
                tp_before_sl=False,
                tp1_before_sl=tp1_before_sl,
                mfe_r=mfe_r,
                mae_r=mae_r,
                net_r_timeout=(last_mark - entry.executed_entry) / risk if candidate.side == "long" else (entry.executed_entry - last_mark) / risk,
                minutes_to_tp_or_sl=minutes,
                exit_ts=int(row["ts"]),
                executed_exit=stop,
                path_source="micro_buckets",
            )
        if stop_hit:
            minutes = max(1, math.ceil((int(row["ts"]) - entry.entry_ts) / 60_000.0))
            return PathResult(
                outcome="sl",
                tp_before_sl=False,
                tp1_before_sl=tp1_before_sl,
                mfe_r=mfe_r,
                mae_r=mae_r,
                net_r_timeout=(last_mark - entry.executed_entry) / risk if candidate.side == "long" else (entry.executed_entry - last_mark) / risk,
                minutes_to_tp_or_sl=minutes,
                exit_ts=int(row["ts"]),
                executed_exit=stop,
                path_source="micro_buckets",
            )
        if tp_hit:
            minutes = max(1, math.ceil((int(row["ts"]) - entry.entry_ts) / 60_000.0))
            return PathResult(
                outcome="tp",
                tp_before_sl=True,
                tp1_before_sl=tp1_before_sl or int(row["ts"]) <= tp1_deadline,
                mfe_r=mfe_r,
                mae_r=mae_r,
                net_r_timeout=(last_mark - entry.executed_entry) / risk if candidate.side == "long" else (entry.executed_entry - last_mark) / risk,
                minutes_to_tp_or_sl=minutes,
                exit_ts=int(row["ts"]),
                executed_exit=executed_tp,
                path_source="micro_buckets",
            )

    net_r_timeout = (last_mark - entry.executed_entry) / risk if candidate.side == "long" else (entry.executed_entry - last_mark) / risk
    return PathResult(
        outcome="timeout",
        tp_before_sl=False,
        tp1_before_sl=tp1_before_sl,
        mfe_r=mfe_r,
        mae_r=mae_r,
        net_r_timeout=net_r_timeout,
        minutes_to_tp_or_sl=None,
        exit_ts=last_ts,
        executed_exit=last_mark,
        path_source="micro_buckets",
    )


def simulate_with_trade_bars(
    candidate: CandidateEvent,
    *,
    entry: EntryDecision,
    trade_bars: pd.DataFrame,
    executed_tp: float,
    executed_tp1: float,
    stop: float,
    risk: float,
) -> PathResult | None:
    if trade_bars.empty:
        return None
    signal_ts = pd.to_datetime(candidate.ts, unit="ms", utc=True)
    future = trade_bars.loc[trade_bars.index > signal_ts].head(candidate.timeout_bars)
    if future.empty:
        return None

    mfe_r = 0.0
    mae_r = 0.0
    tp1_before_sl = False
    trigger_minutes = _trigger_minutes_for_candidate(candidate)
    tp1_bars = _tp1_window_bars(candidate)
    for i, (_, bar) in enumerate(future.iterrows(), start=1):
        high = float(bar["high"])
        low = float(bar["low"])
        if candidate.side == "long":
            mfe_r = max(mfe_r, (high - entry.executed_entry) / risk)
            mae_r = max(mae_r, (entry.executed_entry - low) / risk)
            if high >= executed_tp1 and i <= tp1_bars:
                tp1_before_sl = True
            if low <= stop:
                return PathResult(
                    outcome="sl",
                    tp_before_sl=False,
                    tp1_before_sl=tp1_before_sl,
                    mfe_r=mfe_r,
                    mae_r=mae_r,
                    net_r_timeout=(float(future.iloc[-1]["close"]) - entry.executed_entry) / risk,
                    minutes_to_tp_or_sl=i * trigger_minutes,
                    exit_ts=int(future.index[i - 1].timestamp() * 1000),
                    executed_exit=stop,
                    path_source="trade_bars",
                )
            if high >= executed_tp:
                return PathResult(
                    outcome="tp",
                    tp_before_sl=True,
                    tp1_before_sl=tp1_before_sl,
                    mfe_r=mfe_r,
                    mae_r=mae_r,
                    net_r_timeout=(float(future.iloc[-1]["close"]) - entry.executed_entry) / risk,
                    minutes_to_tp_or_sl=i * trigger_minutes,
                    exit_ts=int(future.index[i - 1].timestamp() * 1000),
                    executed_exit=executed_tp,
                    path_source="trade_bars",
                )
        else:
            mfe_r = max(mfe_r, (entry.executed_entry - low) / risk)
            mae_r = max(mae_r, (high - entry.executed_entry) / risk)
            if low <= executed_tp1 and i <= tp1_bars:
                tp1_before_sl = True
            if high >= stop:
                return PathResult(
                    outcome="sl",
                    tp_before_sl=False,
                    tp1_before_sl=tp1_before_sl,
                    mfe_r=mfe_r,
                    mae_r=mae_r,
                    net_r_timeout=(entry.executed_entry - float(future.iloc[-1]["close"])) / risk,
                    minutes_to_tp_or_sl=i * trigger_minutes,
                    exit_ts=int(future.index[i - 1].timestamp() * 1000),
                    executed_exit=stop,
                    path_source="trade_bars",
                )
            if low <= executed_tp:
                return PathResult(
                    outcome="tp",
                    tp_before_sl=True,
                    tp1_before_sl=tp1_before_sl,
                    mfe_r=mfe_r,
                    mae_r=mae_r,
                    net_r_timeout=(entry.executed_entry - float(future.iloc[-1]["close"])) / risk,
                    minutes_to_tp_or_sl=i * trigger_minutes,
                    exit_ts=int(future.index[i - 1].timestamp() * 1000),
                    executed_exit=executed_tp,
                    path_source="trade_bars",
                )

    net_r_timeout = (float(future.iloc[-1]["close"]) - entry.executed_entry) / risk if candidate.side == "long" else (entry.executed_entry - float(future.iloc[-1]["close"])) / risk
    return PathResult(
        outcome="timeout",
        tp_before_sl=False,
        tp1_before_sl=tp1_before_sl,
        mfe_r=mfe_r,
        mae_r=mae_r,
        net_r_timeout=net_r_timeout,
        minutes_to_tp_or_sl=None,
        exit_ts=int(future.index[-1].timestamp() * 1000),
        executed_exit=float(future.iloc[-1]["close"]),
        path_source="trade_bars",
    )
