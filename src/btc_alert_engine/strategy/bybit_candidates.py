from __future__ import annotations

from typing import Iterable, TypeVar

import numpy as np
import pandas as pd

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


def _normalize_requested_sides(sides: Iterable[str] | None) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(side.lower() for side in (sides or ("long",))))
    valid = tuple(side for side in requested if side in {"long", "short"})
    return valid or ("long",)


def _micro_gate_for_side(row: pd.Series, side: str) -> bool:
    if side == "long":
        legacy_gate = row.get("gate_pass")
        if legacy_gate is not None and not pd.isna(legacy_gate) and bool(legacy_gate):
            return True
        gate_value = row.get("gate_pass_long")
        if gate_value is not None and not pd.isna(gate_value):
            return bool(gate_value)
    else:
        gate_value = row.get("gate_pass_short")
        if gate_value is not None and not pd.isna(gate_value):
            return bool(gate_value)
    ofi_60s = _numeric_or_nan(row.get("ofi_60s"))
    median_bookimb = _numeric_or_nan(row.get("median_bookimb_l10_60s"))
    spread_z = _numeric_or_nan(row.get("spread_z"))
    vwap_mid_dev_z = _numeric_or_nan(row.get("vwap_mid_dev_30s_z"))
    if side == "long":
        return ofi_60s > 0 and median_bookimb > 0.05 and spread_z < 1.5 and abs(vwap_mid_dev_z) < 2.0
    return ofi_60s < 0 and median_bookimb < -0.05 and spread_z < 1.5 and abs(vwap_mid_dev_z) < 2.0


def _continuation_feature_payload(
    row: pd.Series,
    *,
    side: str,
    atr15_value: float,
    setup_age: float,
    setup_impulse_atr: float,
    setup_depth_frac: float,
    setup_dist_to_level: float,
    setup_dist_to_ema50: float,
) -> dict[str, float | bool | None]:
    crowding_key = "crowding_long_score" if side == "long" else "crowding_short_score"
    return {
        "side_sign": 1.0 if side == "long" else -1.0,
        "trend_score": _optional_float(row.get("trend_score")),
        "stress_score": _optional_float(row.get("stress_score")),
        "crowding_score": _optional_float(row.get(crowding_key)),
        "ofi_60s": _optional_float(row.get("ofi_60s")),
        "median_bookimb_l10_60s": _optional_float(row.get("median_bookimb_l10_60s")),
        "atr15": atr15_value,
        "setup_age_1h": setup_age,
        "setup_impulse_atr_1h": setup_impulse_atr,
        "setup_pullback_depth_frac": setup_depth_frac,
        "setup_dist_to_level_aligned": setup_dist_to_level,
        "setup_dist_to_ema50_4h_aligned": setup_dist_to_ema50,
        "veto_active": bool(row.get("veto_active", False)) if pd.notna(row.get("veto_active")) else False,
    }


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
    sides: Iterable[str] | None = None,
) -> list[CandidateEvent]:
    """Build mirrored continuation candidates for the requested sides."""
    price = _bars_to_frame(trade_bars)
    if price.empty:
        return []
    trend = _align_feature_frame(trend_features, price.index)
    regime = _align_feature_frame(regime_features, price.index)
    crowding = _align_feature_frame(crowding_features, price.index)
    micro_15m = _align_feature_frame(micro_features, price.index, resample_rule="15min")
    macro_15m = _align_feature_frame(macro_features or [], price.index, resample_rule="15min")

    atr15 = atr(price, 14)
    recent_pivot_high = price["high"].shift(1).rolling(4, min_periods=4).max()
    recent_pivot_low = price["low"].shift(1).rolling(4, min_periods=4).min()
    recent_stop_low = price["low"].shift(1).rolling(4, min_periods=4).min()
    recent_stop_high = price["high"].shift(1).rolling(4, min_periods=4).max()

    frame = _join_without_overlaps(price, trend)
    frame = _join_without_overlaps(frame, regime)
    frame = _join_without_overlaps(frame, crowding)
    frame = _join_without_overlaps(frame, micro_15m)
    frame = _join_without_overlaps(frame, macro_15m)
    frame["atr15"] = atr15
    frame["recent_pivot_high"] = recent_pivot_high
    frame["recent_pivot_low"] = recent_pivot_low
    frame["recent_stop_low"] = recent_stop_low
    frame["recent_stop_high"] = recent_stop_high

    requested_sides = _normalize_requested_sides(sides)
    candidates: list[CandidateEvent] = []
    for ts, row in frame.iterrows():
        if pd.isna(row.get("atr15")):
            continue
        trend_score = _numeric_or_nan(row.get("trend_score"))
        range_score = _numeric_or_nan(row.get("range_score"))
        stress_score = _numeric_or_nan(row.get("stress_score"))
        macro_active = bool(row.get("veto_active", False))
        event_type = row.get("veto_event_type") or "macro"

        for side in requested_sides:
            veto_reasons: list[str] = []
            if require_crowding_veto and bool(row.get("veto_long" if side == "long" else "veto_short", False)):
                veto_reasons.append(f"crowding_veto_{side}")
            if require_macro_veto and macro_active:
                veto_reasons.append(f"macro_veto_{event_type}")

            close_px = _numeric_or_nan(row.get("close"))
            atr15_value = _numeric_or_nan(row.get("atr15"))
            if side == "long":
                trigger_level = _numeric_or_nan(row.get("recent_pivot_high"))
                stop_anchor = _numeric_or_nan(row.get("recent_stop_low"))
                setup_age = _numeric_or_nan(row.get("breakout_age_1h"))
                setup_impulse = _numeric_or_nan(row.get("impulse_atr_1h"))
                setup_depth = _numeric_or_nan(row.get("pullback_depth_frac"))
                dist_to_level = _numeric_or_nan(row.get("dist_to_breakout_level"))
                dist_to_ema50 = _numeric_or_nan(row.get("dist_to_ema50_4h"))
                conds = {
                    "trend_gap_positive": _numeric_or_nan(row.get("ema50_4h_gap")) > 0,
                    "ema50_slope_positive": _numeric_or_nan(row.get("ema50_4h_slope")) > 0,
                    "ema200_slope_positive": _numeric_or_nan(row.get("ema200_4h_slope")) > 0,
                    "breakout_recent": 0 <= setup_age <= 10,
                    "impulse_valid": setup_impulse >= 1.0,
                    "pullback_depth_valid": 0.20 <= setup_depth <= 0.60,
                    "above_breakout_level": dist_to_level > -0.005,
                    "above_ema50_4h": dist_to_ema50 > 0,
                    "trigger_break": close_px > trigger_level,
                }
            else:
                trigger_level = _numeric_or_nan(row.get("recent_pivot_low"))
                stop_anchor = _numeric_or_nan(row.get("recent_stop_high"))
                setup_age = _numeric_or_nan(row.get("breakdown_age_1h"))
                setup_impulse = _numeric_or_nan(row.get("downside_impulse_atr_1h"))
                setup_depth = _numeric_or_nan(row.get("rebound_depth_frac"))
                raw_dist_to_level = _numeric_or_nan(row.get("dist_to_breakdown_level"))
                raw_dist_to_ema50 = _numeric_or_nan(row.get("dist_to_ema50_4h"))
                dist_to_level = -raw_dist_to_level
                dist_to_ema50 = -raw_dist_to_ema50
                conds = {
                    "trend_gap_negative": _numeric_or_nan(row.get("ema50_4h_gap")) < 0,
                    "ema50_slope_negative": _numeric_or_nan(row.get("ema50_4h_slope")) < 0,
                    "ema200_slope_negative": _numeric_or_nan(row.get("ema200_4h_slope")) < 0,
                    "breakdown_recent": 0 <= setup_age <= 10,
                    "impulse_valid": setup_impulse >= 1.0,
                    "rebound_depth_valid": 0.20 <= setup_depth <= 0.60,
                    "below_breakdown_level": raw_dist_to_level < 0.005,
                    "below_ema50_4h": raw_dist_to_ema50 < 0,
                    "trigger_break": close_px < trigger_level,
                }

            if require_regime_gate:
                conds["trend_regime"] = trend_score >= range_score
                conds["stress_ok"] = stress_score < 0.55
            if require_micro_gate:
                conds["micro_gate"] = _micro_gate_for_side(row, side)
            if require_macro_veto:
                conds["macro_window_clear"] = not macro_active

            if veto_reasons or not all(conds.values()):
                continue

            entry = float(row["close"])
            if side == "long":
                stop = float(min(stop_anchor - 0.25 * atr15_value, entry - 0.50 * atr15_value))
                if not np.isfinite(stop) or stop >= entry:
                    continue
                risk = entry - stop
                tp = entry + 2.0 * risk
            else:
                stop = float(max(stop_anchor + 0.25 * atr15_value, entry + 0.50 * atr15_value))
                if not np.isfinite(stop) or stop <= entry:
                    continue
                risk = stop - entry
                tp = entry - 2.0 * risk
            target_r_multiple = 2.0
            feature_payload = _continuation_feature_payload(
                row,
                side=side,
                atr15_value=_optional_float(row.get("atr15")) or atr15_value,
                setup_age=setup_age,
                setup_impulse_atr=setup_impulse,
                setup_depth_frac=setup_depth,
                setup_dist_to_level=dist_to_level,
                setup_dist_to_ema50=dist_to_ema50,
            )
            feature_payload["signal_risk"] = risk
            feature_payload["signal_risk_pct"] = None if entry == 0 else (risk / entry)
            candidates.append(
                CandidateEvent(
                    ts=int(ts.timestamp() * 1000),
                    venue=venue,
                    symbol=symbol,
                    module="continuation_v1",
                    side=side,
                    entry=entry,
                    stop=stop,
                    tp=tp,
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
    sides: Iterable[str] | None = None,
) -> list[CandidateEvent]:
    price = _bars_to_frame(trade_bars)
    if price.empty:
        return []
    regime = _align_feature_frame(regime_features, price.index)
    crowding = _align_feature_frame(crowding_features, price.index)
    micro_15m = _align_feature_frame(micro_features, price.index, resample_rule="15min")
    macro_15m = _align_feature_frame(macro_features or [], price.index, resample_rule="15min")

    atr15 = atr(price, 14)
    prior20_low = price["low"].shift(1).rolling(20, min_periods=20).min()
    sweep_low = price["low"] < (prior20_low - 0.25 * atr15)
    close_above_midpoint = price["close"] > ((price["high"] + price["low"]) / 2.0)
    prior_reversal_bar_long = sweep_low.shift(1).eq(True) & close_above_midpoint.shift(1).eq(True)
    local_swing_pivot_high = price["high"].shift(2).rolling(4, min_periods=4).max()

    prior20_high = price["high"].shift(1).rolling(20, min_periods=20).max()
    sweep_high = price["high"] > (prior20_high + 0.25 * atr15)
    close_below_midpoint = price["close"] < ((price["high"] + price["low"]) / 2.0)
    prior_reversal_bar_short = sweep_high.shift(1).eq(True) & close_below_midpoint.shift(1).eq(True)
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

    requested_sides = _normalize_requested_sides(sides)
    candidates: list[CandidateEvent] = []
    for ts, row in frame.iterrows():
        if pd.isna(row.get("atr15")):
            continue
        stress_score = _numeric_or_nan(row.get("stress_score"))
        trend_score = _numeric_or_nan(row.get("trend_score"))
        range_score = _numeric_or_nan(row.get("range_score"))
        macro_active = bool(row.get("veto_active", False))
        event_type = row.get("veto_event_type") or "macro"
        close_px = _numeric_or_nan(row.get("close"))
        atr15_value = _numeric_or_nan(row.get("atr15"))

        for side in requested_sides:
            veto_reasons: list[str] = []
            if require_crowding_veto and bool(row.get("veto_long" if side == "long" else "veto_short", False)):
                veto_reasons.append(f"crowding_veto_{side}")
            if require_macro_veto and macro_active:
                veto_reasons.append(f"macro_veto_{event_type}")

            if side == "long":
                liq_z = _numeric_or_nan(row.get("liq_longs_z_1h"))
                pivot = _numeric_or_nan(row.get("local_swing_pivot_high"))
                conds = {
                    "stress_regime": stress_score > max(trend_score, range_score),
                    "liquidation_stress": liq_z >= 3.0,
                    "prior_reversal_bar": bool(row.get("prior_reversal_bar_long", False)),
                    "pivot_reclaim": close_px > pivot,
                }
            else:
                liq_z = _numeric_or_nan(row.get("liq_shorts_z_1h"))
                pivot = _numeric_or_nan(row.get("local_swing_pivot_low"))
                conds = {
                    "stress_regime": stress_score > max(trend_score, range_score),
                    "liquidation_stress": liq_z >= 3.0,
                    "prior_reversal_bar": bool(row.get("prior_reversal_bar_short", False)),
                    "pivot_reclaim": close_px < pivot,
                }
            if require_micro_gate:
                conds["micro_gate"] = _micro_gate_for_side(row, side)
            if require_macro_veto:
                conds["macro_window_clear"] = not macro_active
            if veto_reasons or not all(conds.values()):
                continue

            entry = float(row["close"])
            if side == "long":
                stop = float(row["low"] - 0.20 * atr15_value)
                if stop >= entry:
                    continue
                risk = entry - stop
                tp = entry + 1.5 * risk
            else:
                stop = float(row["high"] + 0.20 * atr15_value)
                if stop <= entry:
                    continue
                risk = stop - entry
                tp = entry - 1.5 * risk
            target_r_multiple = 1.5
            feature_payload = {
                "side_sign": 1.0 if side == "long" else -1.0,
                "stress_score": _optional_float(row.get("stress_score")),
                "liq_stress_z_1h": _optional_float(liq_z),
                "ofi_60s": _optional_float(row.get("ofi_60s")),
                "veto_active": bool(row.get("veto_active", False)) if pd.notna(row.get("veto_active")) else False,
                "signal_risk": risk,
                "signal_risk_pct": None if entry == 0 else (risk / entry),
                "atr15": atr15_value,
            }
            candidates.append(
                CandidateEvent(
                    ts=int(ts.timestamp() * 1000),
                    venue=venue,
                    symbol=symbol,
                    module="stress_reversal_v0",
                    side=side,
                    entry=entry,
                    stop=stop,
                    tp=tp,
                    target_r_multiple=target_r_multiple,
                    timeout_bars=timeout_bars,
                    rule_reasons=[name for name, ok in conds.items() if ok],
                    veto_reasons=veto_reasons,
                    features=feature_payload,
                )
            )
    return candidates
