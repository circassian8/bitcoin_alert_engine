from __future__ import annotations

from typing import Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

from btc_alert_engine.features.common import bars_to_ohlcv_frame, reindex_ffill
from btc_alert_engine.features.indicators import atr
from btc_alert_engine.schemas import (
    CandidateEvent,
    CrowdingFeatureSnapshot,
    MacroVetoFeatureSnapshot,
    MicroFeatureSnapshot,
    PriceBar,
    RegimeFeatureSnapshot,
    TrendFeatureSnapshot,
)
from btc_alert_engine.storage.partitioned_ndjson import iter_json_records

T = TypeVar("T")


VALID_SIDES = ("long", "short")


def _bars_to_frame(bars: Iterable[PriceBar]) -> pd.DataFrame:
    return bars_to_ohlcv_frame(bars)


def _feature_frame(items: Iterable[object]) -> pd.DataFrame:
    rows = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in items]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns=[c for c in ["ts", "symbol"] if c in df.columns])


def load_feature_models(paths: Iterable[str], model_cls: type[T]) -> list[T]:
    return [model_cls.model_validate(record) for record in iter_json_records(paths)]


def _join_without_overlaps(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right.empty:
        return left
    extra = right.drop(columns=[col for col in right.columns if col in left.columns], errors="ignore")
    return left.join(extra, how="left")


def _align_feature_frame(items: Iterable[object], price_index: pd.DatetimeIndex, *, resample_rule: str | None = None) -> pd.DataFrame:
    frame = _feature_frame(items)
    if frame.empty:
        return pd.DataFrame(index=price_index)
    if resample_rule is not None:
        frame = frame.resample(resample_rule).last()
    return reindex_ffill(frame, price_index)


def _numeric_or_nan(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _normalize_sides(sides: Sequence[str] | None) -> list[str]:
    if not sides:
        return ["long"]
    normalized: list[str] = []
    for side in sides:
        value = str(side).lower()
        if value not in VALID_SIDES:
            raise ValueError(f"Unsupported side: {side}")
        if value not in normalized:
            normalized.append(value)
    return normalized or ["long"]


def build_continuation_candidates(
    trade_bars: Iterable[PriceBar],
    trend_features: Iterable[TrendFeatureSnapshot],
    regime_features: Iterable[RegimeFeatureSnapshot],
    crowding_features: Iterable[CrowdingFeatureSnapshot],
    micro_features: Iterable[MicroFeatureSnapshot],
    *,
    symbol: str,
    macro_features: Iterable[MacroVetoFeatureSnapshot] | None = None,
    venue: str = "bybit",
    timeout_bars: int = 96,
    require_regime_gate: bool = True,
    require_crowding_veto: bool = True,
    require_micro_gate: bool = True,
    require_macro_veto: bool = False,
    stop_width_multiplier: float = 1.0,
    target_r_multiple: float = 2.0,
    sides: Sequence[str] | None = None,
    max_stress_score: float = 0.55,
    min_impulse_atr: float = 1.0,
    pullback_depth_range: tuple[float, float] = (0.20, 0.60),
    min_dist_to_level: float = -0.005,
) -> list[CandidateEvent]:
    price = _bars_to_frame(trade_bars)
    if price.empty:
        return []
    run_sides = _normalize_sides(sides)
    trend = _align_feature_frame(trend_features, price.index)
    regime = _align_feature_frame(regime_features, price.index)
    crowding = _align_feature_frame(crowding_features, price.index)
    micro_15m = _align_feature_frame(micro_features, price.index, resample_rule="15min")
    macro_15m = _align_feature_frame(macro_features or [], price.index, resample_rule="15min")

    atr15 = atr(price, 14)
    recent_pivot_high = price["high"].shift(1).rolling(4, min_periods=4).max()
    recent_stop_low = price["low"].shift(1).rolling(4, min_periods=4).min()
    recent_pivot_low = price["low"].shift(1).rolling(4, min_periods=4).min()
    recent_stop_high = price["high"].shift(1).rolling(4, min_periods=4).max()

    frame = _join_without_overlaps(price, trend)
    frame = _join_without_overlaps(frame, regime)
    frame = _join_without_overlaps(frame, crowding)
    frame = _join_without_overlaps(frame, micro_15m)
    frame = _join_without_overlaps(frame, macro_15m)
    frame["atr15"] = atr15
    frame["recent_pivot_high"] = recent_pivot_high
    frame["recent_stop_low"] = recent_stop_low
    frame["recent_pivot_low"] = recent_pivot_low
    frame["recent_stop_high"] = recent_stop_high

    candidates: list[CandidateEvent] = []
    for ts, row in frame.iterrows():
        if pd.isna(row.get("atr15")):
            continue
        ema50_gap = _numeric_or_nan(row.get("ema50_4h_gap"))
        ema50_slope = _numeric_or_nan(row.get("ema50_4h_slope"))
        ema200_slope = _numeric_or_nan(row.get("ema200_4h_slope"))
        trend_score = _numeric_or_nan(row.get("trend_score"))
        range_score = _numeric_or_nan(row.get("range_score"))
        stress_score = _numeric_or_nan(row.get("stress_score"))
        close_px = _numeric_or_nan(row.get("close"))

        for side in run_sides:
            veto_reasons: list[str] = []
            veto_col = "veto_long" if side == "long" else "veto_short"
            if require_crowding_veto and bool(row.get(veto_col, False)):
                veto_reasons.append(f"crowding_veto_{side}")
            if require_macro_veto and bool(row.get("veto_active", False)):
                event_type = row.get("veto_event_type") or "macro"
                veto_reasons.append(f"macro_veto_{event_type}")

            if side == "long":
                if pd.isna(row.get("recent_pivot_high")) or pd.isna(row.get("recent_stop_low")):
                    continue
                conds = {
                    "trend_gap_positive": ema50_gap > 0,
                    "ema50_slope_positive": ema50_slope > 0,
                    "ema200_slope_positive": ema200_slope > 0,
                    "breakout_recent": 0 <= _numeric_or_nan(row.get("breakout_age_1h")) <= 10,
                    "impulse_valid": _numeric_or_nan(row.get("impulse_atr_1h")) >= min_impulse_atr,
                    "pullback_depth_valid": pullback_depth_range[0] <= _numeric_or_nan(row.get("pullback_depth_frac")) <= pullback_depth_range[1],
                    "above_breakout_level": _numeric_or_nan(row.get("dist_to_breakout_level")) > min_dist_to_level,
                    "above_ema50_4h": _numeric_or_nan(row.get("dist_to_ema50_4h")) > 0,
                    "trigger_break": close_px > _numeric_or_nan(row.get("recent_pivot_high")),
                }
                if require_regime_gate:
                    conds["trend_regime"] = trend_score >= range_score
                    conds["stress_ok"] = stress_score < max_stress_score
                if require_micro_gate:
                    conds["micro_gate"] = bool(row.get("gate_pass_long", False))
                if require_macro_veto:
                    conds["macro_window_clear"] = not bool(row.get("veto_active", False))
                if veto_reasons or not all(conds.values()):
                    continue
                entry = float(row["close"])
                base_stop = float(min(row["recent_stop_low"] - 0.25 * row["atr15"], entry - 0.50 * row["atr15"]))
                if not np.isfinite(base_stop) or base_stop >= entry:
                    continue
                risk = (entry - base_stop) * stop_width_multiplier
                stop = float(entry - risk)
                tp = entry + target_r_multiple * risk
            else:
                if pd.isna(row.get("recent_pivot_low")) or pd.isna(row.get("recent_stop_high")):
                    continue
                conds = {
                    "trend_gap_negative": ema50_gap < 0,
                    "ema50_slope_negative": ema50_slope < 0,
                    "ema200_slope_negative": ema200_slope < 0,
                    "breakdown_recent": 0 <= _numeric_or_nan(row.get("breakdown_age_1h")) <= 10,
                    "downside_impulse_valid": _numeric_or_nan(row.get("downside_impulse_atr_1h")) >= min_impulse_atr,
                    "bounce_depth_valid": pullback_depth_range[0] <= _numeric_or_nan(row.get("bounce_depth_frac")) <= pullback_depth_range[1],
                    "below_breakdown_level": _numeric_or_nan(row.get("dist_to_breakdown_level")) > min_dist_to_level,
                    "below_ema50_4h": _numeric_or_nan(row.get("dist_below_ema50_4h")) > 0,
                    "trigger_break": close_px < _numeric_or_nan(row.get("recent_pivot_low")),
                }
                if require_regime_gate:
                    conds["trend_regime"] = trend_score >= range_score
                    conds["stress_ok"] = stress_score < max_stress_score
                if require_micro_gate:
                    conds["micro_gate"] = bool(row.get("gate_pass_short", False))
                if require_macro_veto:
                    conds["macro_window_clear"] = not bool(row.get("veto_active", False))
                if veto_reasons or not all(conds.values()):
                    continue
                entry = float(row["close"])
                base_stop = float(max(row["recent_stop_high"] + 0.25 * row["atr15"], entry + 0.50 * row["atr15"]))
                if not np.isfinite(base_stop) or base_stop <= entry:
                    continue
                risk = (base_stop - entry) * stop_width_multiplier
                stop = float(entry + risk)
                tp = entry - target_r_multiple * risk

            if not np.isfinite(stop) or risk <= 0:
                continue
            feature_payload = {
                "atr15": _optional_float(row.get("atr15")),
                "signal_risk": _optional_float(risk),
                "signal_risk_pct": _optional_float(risk / entry if entry > 0 else None),
                "side_sign": 1.0 if side == "long" else -1.0,
            }
            candidates.append(
                CandidateEvent(
                    ts=int(ts.timestamp() * 1000),
                    venue=venue,
                    symbol=symbol,
                    module="continuation_v1",
                    side=side,
                    entry=entry,
                    stop=float(stop),
                    tp=float(tp),
                    target_r_multiple=target_r_multiple,
                    timeout_bars=timeout_bars,
                    rule_reasons=[name for name, ok in conds.items() if ok],
                    veto_reasons=veto_reasons,
                    features=feature_payload,
                )
            )
    return candidates


def build_stress_reversal_candidates(
    trade_bars: Iterable[PriceBar],
    regime_features: Iterable[RegimeFeatureSnapshot],
    crowding_features: Iterable[CrowdingFeatureSnapshot],
    micro_features: Iterable[MicroFeatureSnapshot],
    *,
    symbol: str,
    macro_features: Iterable[MacroVetoFeatureSnapshot] | None = None,
    venue: str = "bybit",
    timeout_bars: int = 32,
    require_crowding_veto: bool = True,
    require_micro_gate: bool = True,
    require_macro_veto: bool = False,
    sides: Sequence[str] | None = None,
    target_r_multiple: float = 1.5,
) -> list[CandidateEvent]:
    price = _bars_to_frame(trade_bars)
    if price.empty:
        return []
    run_sides = _normalize_sides(sides)
    regime = _align_feature_frame(regime_features, price.index)
    crowding = _align_feature_frame(crowding_features, price.index)
    micro_15m = _align_feature_frame(micro_features, price.index, resample_rule="15min")
    macro_15m = _align_feature_frame(macro_features or [], price.index, resample_rule="15min")

    atr15 = atr(price, 14)
    prior20_low = price["low"].shift(1).rolling(20, min_periods=20).min()
    prior20_high = price["high"].shift(1).rolling(20, min_periods=20).max()
    close_above_midpoint = price["close"] > ((price["high"] + price["low"]) / 2.0)
    close_below_midpoint = price["close"] < ((price["high"] + price["low"]) / 2.0)
    long_sweep_low = price["low"] < (prior20_low - 0.25 * atr15)
    short_sweep_high = price["high"] > (prior20_high + 0.25 * atr15)
    prior_reversal_bar_long = long_sweep_low.shift(1).eq(True) & close_above_midpoint.shift(1).eq(True)
    prior_reversal_bar_short = short_sweep_high.shift(1).eq(True) & close_below_midpoint.shift(1).eq(True)
    local_swing_pivot_high = price["high"].shift(2).rolling(4, min_periods=4).max()
    local_swing_pivot_low = price["low"].shift(2).rolling(4, min_periods=4).min()

    frame = _join_without_overlaps(price, regime)
    frame = _join_without_overlaps(frame, crowding)
    frame = _join_without_overlaps(frame, micro_15m)
    frame = _join_without_overlaps(frame, macro_15m)
    frame["atr15"] = atr15
    frame["prior20_low"] = prior20_low
    frame["prior20_high"] = prior20_high
    frame["prior_reversal_bar_long"] = prior_reversal_bar_long
    frame["prior_reversal_bar_short"] = prior_reversal_bar_short
    frame["local_swing_pivot_high"] = local_swing_pivot_high
    frame["local_swing_pivot_low"] = local_swing_pivot_low

    candidates: list[CandidateEvent] = []
    for ts, row in frame.iterrows():
        if pd.isna(row.get("atr15")):
            continue
        stress_score = _numeric_or_nan(row.get("stress_score"))
        trend_score = _numeric_or_nan(row.get("trend_score"))
        range_score = _numeric_or_nan(row.get("range_score"))
        close_px = _numeric_or_nan(row.get("close"))

        for side in run_sides:
            veto_reasons: list[str] = []
            veto_col = "veto_long" if side == "long" else "veto_short"
            if require_crowding_veto and bool(row.get(veto_col, False)):
                veto_reasons.append(f"crowding_veto_{side}")
            if require_macro_veto and bool(row.get("veto_active", False)):
                event_type = row.get("veto_event_type") or "macro"
                veto_reasons.append(f"macro_veto_{event_type}")

            if side == "long":
                liq_score = _numeric_or_nan(row.get("liq_longs_z_1h"))
                conds = {
                    "stress_regime": stress_score > max(trend_score, range_score),
                    "liquidation_stress": liq_score >= 3.0,
                    "prior_reversal_bar": bool(row.get("prior_reversal_bar_long", False)),
                    "pivot_reclaim": close_px > _numeric_or_nan(row.get("local_swing_pivot_high")),
                }
                if require_micro_gate:
                    conds["micro_gate"] = bool(row.get("gate_pass_long", False))
                if require_macro_veto:
                    conds["macro_window_clear"] = not bool(row.get("veto_active", False))
                if veto_reasons or not all(conds.values()):
                    continue
                entry = float(row["close"])
                stop = float(row["low"] - 0.20 * row["atr15"])
                if stop >= entry:
                    continue
                risk = entry - stop
                tp = entry + target_r_multiple * risk
            else:
                liq_score = _numeric_or_nan(row.get("liq_shorts_z_1h"))
                conds = {
                    "stress_regime": stress_score > max(trend_score, range_score),
                    "liquidation_stress": liq_score >= 3.0,
                    "prior_reversal_bar": bool(row.get("prior_reversal_bar_short", False)),
                    "pivot_breakdown": close_px < _numeric_or_nan(row.get("local_swing_pivot_low")),
                }
                if require_micro_gate:
                    conds["micro_gate"] = bool(row.get("gate_pass_short", False))
                if require_macro_veto:
                    conds["macro_window_clear"] = not bool(row.get("veto_active", False))
                if veto_reasons or not all(conds.values()):
                    continue
                entry = float(row["close"])
                stop = float(row["high"] + 0.20 * row["atr15"])
                if stop <= entry:
                    continue
                risk = stop - entry
                tp = entry - target_r_multiple * risk

            feature_payload = {
                "atr15": _optional_float(row.get("atr15")),
                "signal_risk": _optional_float(risk),
                "signal_risk_pct": _optional_float(risk / entry if entry > 0 else None),
                "side_sign": 1.0 if side == "long" else -1.0,
            }
            candidates.append(
                CandidateEvent(
                    ts=int(ts.timestamp() * 1000),
                    venue=venue,
                    symbol=symbol,
                    module="stress_reversal_v0",
                    side=side,
                    entry=entry,
                    stop=float(stop),
                    tp=float(tp),
                    target_r_multiple=target_r_multiple,
                    timeout_bars=timeout_bars,
                    rule_reasons=[name for name, ok in conds.items() if ok],
                    veto_reasons=veto_reasons,
                    features=feature_payload,
                )
            )
    return candidates
