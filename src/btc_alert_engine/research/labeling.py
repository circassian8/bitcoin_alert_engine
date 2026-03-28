from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from btc_alert_engine.research.execution import (
    bars_to_frame,
    build_raw_execution_tape,
    micro_to_frame,
    resolve_entry,
    simulate_with_micro_buckets,
    simulate_with_raw_tape,
    simulate_with_trade_bars,
)
from btc_alert_engine.schemas import CandidateEvent, CandidateLabel, MicroBucket1s, PriceBar


def candidate_id(candidate: CandidateEvent) -> str:
    return f"{candidate.module}:{candidate.side}:{candidate.symbol}:{candidate.ts}"


def _resolve_target_r_multiple(candidate: CandidateEvent) -> float:
    if candidate.target_r_multiple is not None:
        return float(candidate.target_r_multiple)
    signal_risk = abs(candidate.entry - candidate.stop)
    if signal_risk <= 0:
        raise ValueError(f"Candidate {candidate_id(candidate)} has non-positive signal risk")
    return abs(candidate.tp - candidate.entry) / signal_risk


def label_candidates(
    candidates: Iterable[CandidateEvent],
    trade_bars: Iterable[PriceBar],
    *,
    micro_buckets: Iterable[MicroBucket1s] | None = None,
    raw_paths: Iterable[str | Path] | None = None,
    raw_tape: pd.DataFrame | None = None,
    slippage_bps: float = 1.0,
    latency_ms: int = 0,
    tolerate_raw_gaps: bool = True,
) -> list[CandidateLabel]:
    candidate_list = list(candidates)
    price = bars_to_frame(trade_bars)
    if price.empty or not candidate_list:
        return []
    micro_frame = micro_to_frame(micro_buckets or [])
    execution_tape = raw_tape
    if execution_tape is None and raw_paths:
        execution_tape, _ = build_raw_execution_tape(raw_paths, symbol=candidate_list[0].symbol, tolerate_gaps=tolerate_raw_gaps)

    labels: list[CandidateLabel] = []
    for candidate in candidate_list:
        entry = resolve_entry(
            candidate,
            price,
            raw_tape=execution_tape,
            micro_frame=micro_frame,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
        )
        if entry is None:
            continue

        target_r_multiple = _resolve_target_r_multiple(candidate)
        if candidate.side == "long":
            risk = entry.executed_entry - candidate.stop
            if risk <= 0:
                continue
            executed_tp = entry.executed_entry + target_r_multiple * risk
            executed_tp1 = entry.executed_entry + 1.0 * risk
            stop = candidate.stop
        else:
            risk = candidate.stop - entry.executed_entry
            if risk <= 0:
                continue
            executed_tp = entry.executed_entry - target_r_multiple * risk
            executed_tp1 = entry.executed_entry - 1.0 * risk
            stop = candidate.stop

        path = None
        if execution_tape is not None and not execution_tape.empty:
            path = simulate_with_raw_tape(
                candidate,
                entry=entry,
                raw_tape=execution_tape,
                executed_tp=executed_tp,
                executed_tp1=executed_tp1,
                stop=stop,
                risk=risk,
            )
        if path is None and not micro_frame.empty:
            path = simulate_with_micro_buckets(
                candidate,
                entry=entry,
                micro_frame=micro_frame,
                executed_tp=executed_tp,
                executed_tp1=executed_tp1,
                stop=stop,
                risk=risk,
            )
        if path is None:
            path = simulate_with_trade_bars(
                candidate,
                entry=entry,
                trade_bars=price,
                executed_tp=executed_tp,
                executed_tp1=executed_tp1,
                stop=stop,
                risk=risk,
            )
        if path is None:
            continue

        labels.append(
            CandidateLabel(
                candidate_id=candidate_id(candidate),
                ts=candidate.ts,
                module=candidate.module,
                symbol=candidate.symbol,
                side=candidate.side,
                signal_entry=candidate.entry,
                signal_tp=candidate.tp,
                entry_ts=entry.entry_ts,
                executed_entry=entry.executed_entry,
                stop=stop,
                executed_tp=executed_tp,
                exit_ts=path.exit_ts,
                executed_exit=path.executed_exit,
                target_r_multiple=target_r_multiple,
                timeout_bars=candidate.timeout_bars,
                tp_before_sl_within_horizon=path.tp_before_sl,
                tp1_before_sl_within_12h=path.tp1_before_sl,
                mfe_r_24h=path.mfe_r,
                mae_r_24h=path.mae_r,
                net_r_24h_timeout=path.net_r_timeout,
                minutes_to_tp_or_sl=path.minutes_to_tp_or_sl,
                outcome=path.outcome,
                entry_source=entry.entry_source,
                path_source=path.path_source,
            )
        )
    return labels
