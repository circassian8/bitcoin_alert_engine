from __future__ import annotations

from typing import Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

from btc_alert_engine.features.indicators import atr
from btc_alert_engine.profiles import StrategyProfile, get_profile
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
    rows = [bar.model_dump(mode="json") if isinstance(bar, PriceBar) else bar for bar in bars]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)


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
    return frame.reindex(price_index, method="ffill")


def _numeric_or_nan(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _feature_value(row: pd.Series, generic_name: str, legacy_name: str | None = None) -> float:
    if generic_name in row and pd.notna(row.get(generic_name)):
        return _numeric_or_nan(row.get(generic_name))
    if legacy_name and legacy_name in row:
        return _numeric_or_nan(row.get(legacy_name))
    return float("nan")


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
    timeout_bars: int | None = None,
    require_regime_gate: bool = True,
    require_crowding_veto: bool = True,
    require_micro_gate: bool = True,
    require_macro_veto: bool = False,
    stop_width_multiplier: float = 1.0,
    target_r_multiple: float = 2.0,
    sides: Sequence[str] | None = None,
    profile: str | StrategyProfile = "core",
) -> list[CandidateEvent]:
    resolved_profile = get_profile(profile)
    timeout_bars = timeout_bars or resolved_profile.continuation_timeout_bars
    price = _bars_to_frame(trade_bars)
    if price.empty:
        return []
    run_sides = _normalize_sides(sides)
    trend = _align_feature_frame(trend_features, price.index)
    regime = _align_feature_frame(regime_features, price.index)
    crowding = _align_feature_frame(crowding_features, price.index)
    micro_trigger = _align_feature_frame(micro_features, price.index, resample_rule=resolved_profile.trigger_resample_rule)
    macro_trigger = _align_feature_frame(macro_features or [], price.index, resample_rule=resolved_profile.trigger_resample_rule)

    atr15 = atr(price, 14)
    recent_pivot_high = price["high"].shift(1).rolling(resolved_profile.trigger_pivot_lookback, min_periods=resolved_profile.trigger_pivot_lookback).max()
    recent_stop_low = price["low"].shift(1).rolling(resolved_profile.trigger_pivot_lookback, min_periods=resolved_profile.trigger_pivot_lookback).min()
    recent_pivot_low = price["low"].shift(1).rolling(resolved_profile.trigger_pivot_lookback, min_periods=resolved_profile.trigger_pivot_lookback).min()
    recent_stop_high = price["high"].shift(1).rolling(resolved_profile.trigger_pivot_lookback, min_periods=resolved_profile.trigger_pivot_lookback).max()

    frame = _join_without_overlaps(price, trend)
    frame = _join_without_overlaps(frame, regime)
    frame = _join_without_overlaps(frame, crowding)
    frame = _join_without_overlaps(frame, micro_trigger)
    frame = _join_without_overlaps(frame, macro_trigger)
    frame["atr15"] = atr15
    frame["recent_pivot_high"] = recent_pivot_high
    frame["recent_stop_low"] = recent_stop_low
    frame["recent_pivot_low"] = recent_pivot_low
    frame["recent_stop_high"] = recent_stop_high

    candidates: list[CandidateEvent] = []
    for ts, row in frame.iterrows():
        if pd.isna(row.get("atr15")):
            continue
        ema50_gap = _feature_value(row, "ema_fast_regime_gap", "ema50_4h_gap")
        ema50_slope = _feature_value(row, "ema_fast_regime_slope", "ema50_4h_slope")
        ema200_slope = _feature_value(row, "ema_slow_regime_slope", "ema200_4h_slope")
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
                    "breakout_recent": 0 <= _feature_value(row, "setup_break_age", "breakout_age_1h") <= resolved_profile.setup_active_bars,
                    "impulse_valid": _feature_value(row, "setup_impulse_atr", "impulse_atr_1h") >= 1.0,
                    "pullback_depth_valid": 0.20 <= _feature_value(row, "setup_pullback_depth_frac", "pullback_depth_frac") <= 0.60,
                    "above_breakout_level": _feature_value(row, "dist_to_setup_breakout_level", "dist_to_breakout_level") > -0.005,
                    "above_regime_ema": _feature_value(row, "dist_to_regime_ema", "dist_to_ema50_4h") > 0,
                    "trigger_break": close_px > _numeric_or_nan(row.get("recent_pivot_high")),
                }
                if require_regime_gate:
                    conds["trend_regime"] = trend_score >= range_score
                    conds["stress_ok"] = stress_score < 0.55
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
                    "breakdown_recent": 0 <= _feature_value(row, "setup_breakdown_age", "breakdown_age_1h") <= resolved_profile.setup_active_bars,
                    "downside_impulse_valid": _feature_value(row, "setup_downside_impulse_atr", "downside_impulse_atr_1h") >= 1.0,
                    "bounce_depth_valid": 0.20 <= _feature_value(row, "setup_bounce_depth_frac", "bounce_depth_frac") <= 0.60,
                    "below_breakdown_level": _feature_value(row, "dist_to_setup_breakdown_level", "dist_to_breakdown_level") > -0.005,
                    "below_regime_ema": _feature_value(row, "dist_below_regime_ema", "dist_below_ema50_4h") > 0,
                    "trigger_break": close_px < _numeric_or_nan(row.get("recent_pivot_low")),
                }
                if require_regime_gate:
                    conds["trend_regime"] = trend_score >= range_score
                    conds["stress_ok"] = stress_score < 0.55
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
                    module=resolved_profile.continuation_module,
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
    timeout_bars: int | None = None,
    require_crowding_veto: bool = True,
    require_micro_gate: bool = True,
    require_macro_veto: bool = False,
    sides: Sequence[str] | None = None,
    profile: str | StrategyProfile = "core",
) -> list[CandidateEvent]:
    resolved_profile = get_profile(profile)
    timeout_bars = timeout_bars or resolved_profile.stress_timeout_bars
    price = _bars_to_frame(trade_bars)
    if price.empty:
        return []
    run_sides = _normalize_sides(sides)
    regime = _align_feature_frame(regime_features, price.index)
    crowding = _align_feature_frame(crowding_features, price.index)
    micro_trigger = _align_feature_frame(micro_features, price.index, resample_rule=resolved_profile.trigger_resample_rule)
    macro_trigger = _align_feature_frame(macro_features or [], price.index, resample_rule=resolved_profile.trigger_resample_rule)

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
    frame = _join_without_overlaps(frame, micro_trigger)
    frame = _join_without_overlaps(frame, macro_trigger)
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
                target_r_multiple = 1.5
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
                target_r_multiple = 1.5
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
                    module=resolved_profile.stress_reversal_module,
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
