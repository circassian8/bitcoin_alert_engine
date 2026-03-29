from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from btc_alert_engine.features.indicators import adx, atr, ema, resample_ohlcv, rolling_percentile_of_last, rolling_zscore, softmax_scores
from btc_alert_engine.normalize.bybit_rest import parse_account_ratio_event, parse_funding_history_event, parse_open_interest_event
from btc_alert_engine.profiles import StrategyProfile, get_profile
from btc_alert_engine.schemas import (
    CrowdingFeatureSnapshot,
    MicroBucket1s,
    MicroFeatureSnapshot,
    PriceBar,
    RegimeFeatureSnapshot,
    TrendFeatureSnapshot,
)
from btc_alert_engine.storage.partitioned_ndjson import iter_json_records
from btc_alert_engine.storage.raw_ndjson import iter_raw_events_sorted

EPS = 1e-12


def _bars_to_frame(bars: Iterable[PriceBar]) -> pd.DataFrame:
    records = [bar.model_dump(mode="json") if isinstance(bar, PriceBar) else bar for bar in bars]
    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume", "turnover"]]


def _micro_to_frame(buckets: Iterable[MicroBucket1s | dict]) -> pd.DataFrame:
    rows = [bucket.model_dump(mode="json") if isinstance(bucket, MicroBucket1s) else bucket for bucket in buckets]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def _snapshot_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (str, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_price_bars(paths: Iterable[str]) -> list[PriceBar]:
    return [PriceBar.model_validate(record) for record in iter_json_records(paths)]


def load_micro_buckets(paths: Iterable[str]) -> list[MicroBucket1s]:
    return [MicroBucket1s.model_validate(record) for record in iter_json_records(paths)]


def _profile_metadata(profile: StrategyProfile) -> dict[str, str]:
    return {
        "profile_id": profile.id,
        "trigger_interval": profile.trigger_interval_label,
        "setup_interval": profile.setup_interval_label,
        "regime_interval": profile.regime_interval_label,
    }


def _bars_per_day(profile: StrategyProfile) -> int:
    return profile.regime_bars_per_day


def _trend_feature_frame(trigger_bars: Iterable[PriceBar], *, profile: StrategyProfile) -> pd.DataFrame:
    trigger_df = _bars_to_frame(trigger_bars)
    if trigger_df.empty:
        return pd.DataFrame()

    bars_setup = resample_ohlcv(trigger_df, profile.setup_resample_rule)
    bars_regime = resample_ohlcv(trigger_df, profile.regime_resample_rule)

    close_regime = bars_regime["close"]
    ema_fast_regime = ema(close_regime, profile.regime_ema_fast)
    ema_slow_regime = ema(close_regime, profile.regime_ema_slow)
    atr_regime = atr(bars_regime, profile.atr_period)
    adx_regime = adx(bars_regime, profile.adx_period)
    atr_pctile_90d = rolling_percentile_of_last(atr_regime, window=profile.atr_percentile_days * _bars_per_day(profile))

    regime = pd.DataFrame(index=bars_regime.index)
    regime["ret_regime_6"] = close_regime.pct_change(6)
    regime["ema_fast_regime_gap"] = (close_regime - ema_fast_regime) / close_regime.replace(0, np.nan)
    regime["ema_fast_regime_slope"] = ema_fast_regime.pct_change(profile.regime_slope_lookback, fill_method=None)
    regime["ema_slow_regime_slope"] = ema_slow_regime.pct_change(profile.regime_slope_lookback, fill_method=None)
    regime["adx_regime"] = adx_regime
    regime["atr_pctile_90d"] = atr_pctile_90d
    regime["ema_fast_regime"] = ema_fast_regime

    close_setup = bars_setup["close"]
    atr_setup = atr(bars_setup, profile.atr_period)
    prior_setup_high = bars_setup["high"].shift(1).rolling(profile.setup_breakout_lookback, min_periods=profile.setup_breakout_lookback).max()
    prior_setup_low = bars_setup["low"].shift(1).rolling(profile.setup_breakout_lookback, min_periods=profile.setup_breakout_lookback).min()
    breakout_flag = close_setup > prior_setup_high
    breakdown_flag = close_setup < prior_setup_low

    setup = pd.DataFrame(index=bars_setup.index)
    setup["ret_setup_4"] = close_setup.pct_change(4)

    breakout_age: list[float] = []
    impulse_atr: list[float] = []
    pullback_depth_frac: list[float] = []
    pullback_bars: list[float] = []
    breakout_level_vals: list[float] = []

    breakdown_age: list[float] = []
    downside_impulse_atr: list[float] = []
    bounce_depth_frac: list[float] = []
    bounce_bars: list[float] = []
    breakdown_level_vals: list[float] = []

    active_breakout_idx: int | None = None
    breakout_level = float("nan")
    long_impulse = float("nan")
    high_since_breakout = float("nan")
    bars_since_high = float("nan")

    active_breakdown_idx: int | None = None
    breakdown_level = float("nan")
    short_impulse = float("nan")
    low_since_breakdown = float("nan")
    bars_since_low = float("nan")

    for i, (_, row) in enumerate(bars_setup.iterrows()):
        current_atr = float(atr_setup.iloc[i]) if not math.isnan(float(atr_setup.iloc[i])) else float("nan")

        if bool(breakout_flag.iloc[i]) and not math.isnan(float(prior_setup_high.iloc[i])):
            active_breakout_idx = i
            breakout_level = float(prior_setup_high.iloc[i])
            long_impulse = ((float(row["close"]) - breakout_level) / current_atr) if current_atr and not math.isnan(current_atr) else float("nan")
            high_since_breakout = float(row["high"])
            bars_since_high = 0.0
        elif active_breakout_idx is not None:
            age = i - active_breakout_idx
            if age > profile.setup_active_bars:
                active_breakout_idx = None
                breakout_level = float("nan")
                long_impulse = float("nan")
                high_since_breakout = float("nan")
                bars_since_high = float("nan")
            else:
                if float(row["high"]) >= high_since_breakout:
                    high_since_breakout = float(row["high"])
                    bars_since_high = 0.0
                else:
                    bars_since_high = 0.0 if math.isnan(bars_since_high) else bars_since_high + 1.0

        if bool(breakdown_flag.iloc[i]) and not math.isnan(float(prior_setup_low.iloc[i])):
            active_breakdown_idx = i
            breakdown_level = float(prior_setup_low.iloc[i])
            short_impulse = ((breakdown_level - float(row["close"])) / current_atr) if current_atr and not math.isnan(current_atr) else float("nan")
            low_since_breakdown = float(row["low"])
            bars_since_low = 0.0
        elif active_breakdown_idx is not None:
            age = i - active_breakdown_idx
            if age > profile.setup_active_bars:
                active_breakdown_idx = None
                breakdown_level = float("nan")
                short_impulse = float("nan")
                low_since_breakdown = float("nan")
                bars_since_low = float("nan")
            else:
                if float(row["low"]) <= low_since_breakdown:
                    low_since_breakdown = float(row["low"])
                    bars_since_low = 0.0
                else:
                    bars_since_low = 0.0 if math.isnan(bars_since_low) else bars_since_low + 1.0

        if active_breakout_idx is None:
            breakout_age.append(float("nan"))
            impulse_atr.append(float("nan"))
            pullback_depth_frac.append(float("nan"))
            pullback_bars.append(float("nan"))
            breakout_level_vals.append(float("nan"))
        else:
            age = float(i - active_breakout_idx)
            breakout_age.append(age)
            impulse_atr.append(long_impulse)
            breakout_level_vals.append(breakout_level)
            denom = max(high_since_breakout - breakout_level, EPS)
            pullback_depth = max(high_since_breakout - float(row["close"]), 0.0) / denom
            pullback_depth_frac.append(pullback_depth)
            pullback_bars.append(bars_since_high)

        if active_breakdown_idx is None:
            breakdown_age.append(float("nan"))
            downside_impulse_atr.append(float("nan"))
            bounce_depth_frac.append(float("nan"))
            bounce_bars.append(float("nan"))
            breakdown_level_vals.append(float("nan"))
        else:
            age = float(i - active_breakdown_idx)
            breakdown_age.append(age)
            downside_impulse_atr.append(short_impulse)
            breakdown_level_vals.append(breakdown_level)
            denom = max(breakdown_level - low_since_breakdown, EPS)
            bounce_depth = max(float(row["close"]) - low_since_breakdown, 0.0) / denom
            bounce_depth_frac.append(bounce_depth)
            bounce_bars.append(bars_since_low)

    setup["setup_break_age"] = breakout_age
    setup["setup_impulse_atr"] = impulse_atr
    setup["setup_pullback_depth_frac"] = pullback_depth_frac
    setup["setup_pullback_bars"] = pullback_bars
    setup["setup_breakout_level"] = breakout_level_vals
    setup["setup_breakdown_age"] = breakdown_age
    setup["setup_downside_impulse_atr"] = downside_impulse_atr
    setup["setup_bounce_depth_frac"] = bounce_depth_frac
    setup["setup_bounce_bars"] = bounce_bars
    setup["setup_breakdown_level"] = breakdown_level_vals

    trend = pd.DataFrame(index=trigger_df.index)
    trend["ret_trigger_1"] = trigger_df["close"].pct_change(1)
    trend = trend.join(setup.reindex(trigger_df.index, method="ffill"), how="left")
    trend = trend.join(regime.reindex(trigger_df.index, method="ffill"), how="left")
    trend["dist_to_setup_breakout_level"] = (trigger_df["close"] - trend["setup_breakout_level"]) / trigger_df["close"].replace(0, np.nan)
    trend["dist_to_regime_ema"] = (trigger_df["close"] - trend["ema_fast_regime"]) / trigger_df["close"].replace(0, np.nan)
    trend["dist_to_setup_breakdown_level"] = (trend["setup_breakdown_level"] - trigger_df["close"]) / trigger_df["close"].replace(0, np.nan)
    trend["dist_below_regime_ema"] = (trend["ema_fast_regime"] - trigger_df["close"]) / trigger_df["close"].replace(0, np.nan)
    for key, value in _profile_metadata(profile).items():
        trend[key] = value

    if profile.id == "core":
        trend["ret_15m_1"] = trend["ret_trigger_1"]
        trend["ret_1h_4"] = trend["ret_setup_4"]
        trend["ret_4h_6"] = trend["ret_regime_6"]
        trend["ema50_4h_gap"] = trend["ema_fast_regime_gap"]
        trend["ema50_4h_slope"] = trend["ema_fast_regime_slope"]
        trend["ema200_4h_slope"] = trend["ema_slow_regime_slope"]
        trend["adx14_4h"] = trend["adx_regime"]
        trend["breakout_age_1h"] = trend["setup_break_age"]
        trend["impulse_atr_1h"] = trend["setup_impulse_atr"]
        trend["pullback_depth_frac"] = trend["setup_pullback_depth_frac"]
        trend["pullback_bars"] = trend["setup_pullback_bars"]
        trend["dist_to_breakout_level"] = trend["dist_to_setup_breakout_level"]
        trend["dist_to_ema50_4h"] = trend["dist_to_regime_ema"]
        trend["breakdown_age_1h"] = trend["setup_breakdown_age"]
        trend["downside_impulse_atr_1h"] = trend["setup_downside_impulse_atr"]
        trend["bounce_depth_frac"] = trend["setup_bounce_depth_frac"]
        trend["bounce_bars"] = trend["setup_bounce_bars"]
        trend["dist_to_breakdown_level"] = trend["dist_to_setup_breakdown_level"]
        trend["dist_below_ema50_4h"] = trend["dist_below_regime_ema"]

    columns = [
        "profile_id",
        "trigger_interval",
        "setup_interval",
        "regime_interval",
        "ret_trigger_1",
        "ret_setup_4",
        "ret_regime_6",
        "ema_fast_regime_gap",
        "ema_fast_regime_slope",
        "ema_slow_regime_slope",
        "adx_regime",
        "atr_pctile_90d",
        "setup_break_age",
        "setup_impulse_atr",
        "setup_pullback_depth_frac",
        "setup_pullback_bars",
        "dist_to_setup_breakout_level",
        "dist_to_regime_ema",
        "setup_breakdown_age",
        "setup_downside_impulse_atr",
        "setup_bounce_depth_frac",
        "setup_bounce_bars",
        "dist_to_setup_breakdown_level",
        "dist_below_regime_ema",
    ]
    if profile.id == "core":
        columns.extend([
            "ret_15m_1",
            "ret_1h_4",
            "ret_4h_6",
            "ema50_4h_gap",
            "ema50_4h_slope",
            "ema200_4h_slope",
            "adx14_4h",
            "breakout_age_1h",
            "impulse_atr_1h",
            "pullback_depth_frac",
            "pullback_bars",
            "dist_to_breakout_level",
            "dist_to_ema50_4h",
            "breakdown_age_1h",
            "downside_impulse_atr_1h",
            "bounce_depth_frac",
            "bounce_bars",
            "dist_to_breakdown_level",
            "dist_below_ema50_4h",
        ])
    return trend[columns]


def build_trend_features(trigger_bars: Iterable[PriceBar], *, symbol: str, profile: str | StrategyProfile = "core") -> list[TrendFeatureSnapshot]:
    resolved_profile = get_profile(profile)
    frame = _trend_feature_frame(trigger_bars, profile=resolved_profile)
    snapshots: list[TrendFeatureSnapshot] = []
    for ts, row in frame.iterrows():
        snapshots.append(TrendFeatureSnapshot(ts=int(ts.timestamp() * 1000), symbol=symbol, **{k: _snapshot_value(v) for k, v in row.items()}))
    return snapshots


def build_regime_features(
    trigger_bars: Iterable[PriceBar],
    index_bars: Iterable[PriceBar],
    premium_bars: Iterable[PriceBar],
    *,
    symbol: str,
    profile: str | StrategyProfile = "core",
) -> list[RegimeFeatureSnapshot]:
    resolved_profile = get_profile(profile)
    trigger_df = _bars_to_frame(trigger_bars)
    index_df = _bars_to_frame(index_bars)
    premium_df = _bars_to_frame(premium_bars)
    if trigger_df.empty:
        return []
    trend = _trend_feature_frame(trigger_bars, profile=resolved_profile)

    close = trigger_df["close"]
    log_ret = np.log(close / close.shift(1))
    trigger_bars_per_day = max((24 * 60) // resolved_profile.trigger_minutes, 1)
    rv_1d = log_ret.rolling(trigger_bars_per_day, min_periods=max(trigger_bars_per_day // 4, 4)).std(ddof=0) * np.sqrt(trigger_bars_per_day)
    rv_7d = log_ret.rolling(trigger_bars_per_day * 7, min_periods=trigger_bars_per_day).std(ddof=0) * np.sqrt(trigger_bars_per_day)
    ret_std_1d = log_ret.rolling(trigger_bars_per_day, min_periods=max(trigger_bars_per_day // 4, 4)).std(ddof=0)
    jump_intensity_1d = (log_ret.abs() > (3 * ret_std_1d)).astype(float).rolling(trigger_bars_per_day, min_periods=max(trigger_bars_per_day // 4, 4)).mean()

    regime = pd.DataFrame(index=trigger_df.index)
    regime["rv_1d"] = rv_1d
    regime["rv_7d"] = rv_7d
    regime["atr_pctile_90d"] = trend["atr_pctile_90d"]
    regime["jump_intensity_1d"] = jump_intensity_1d
    for key, value in _profile_metadata(resolved_profile).items():
        regime[key] = value

    if not index_df.empty:
        idx_close = index_df["close"].reindex(trigger_df.index, method="ffill")
        regime["mark_index_gap"] = (close - idx_close) / idx_close.replace(0, np.nan)
    else:
        regime["mark_index_gap"] = np.nan

    premium_window = trigger_bars_per_day * 7
    if not premium_df.empty:
        prem_close = premium_df["close"].reindex(trigger_df.index, method="ffill")
        regime["premium_index_trigger"] = prem_close
        if resolved_profile.id == "core":
            regime["premium_index_15m"] = prem_close
        regime["premium_z_7d"] = rolling_zscore(prem_close, premium_window, min_periods=max(trigger_bars_per_day, 16))
    else:
        regime["premium_index_trigger"] = np.nan
        regime["premium_index_15m"] = np.nan
        regime["premium_z_7d"] = np.nan

    trend_raw = (
        1.2 * (trend["adx_regime"].fillna(0) / 25.0)
        + 2.0 * trend["ema_fast_regime_gap"].abs().fillna(0) * 100
        + 0.5 * trend["ret_regime_6"].abs().fillna(0) * 10
        - 0.5 * regime["atr_pctile_90d"].fillna(0) / 100.0
    )
    range_raw = (
        -1.5 * trend["ema_fast_regime_gap"].abs().fillna(0) * 100
        - 0.8 * (trend["adx_regime"].fillna(0) / 25.0)
        + 0.6 * (1.0 - regime["atr_pctile_90d"].fillna(0) / 100.0)
        - 20.0 * regime["mark_index_gap"].abs().fillna(0)
    )
    stress_raw = (
        1.2 * regime["atr_pctile_90d"].fillna(0) / 100.0
        + 0.8 * regime["jump_intensity_1d"].fillna(0)
        + 0.5 * regime["premium_z_7d"].abs().fillna(0)
        + 20.0 * regime["mark_index_gap"].abs().fillna(0)
    )
    trend_score, range_score, stress_score = softmax_scores(trend_raw, range_raw, stress_raw)
    regime["trend_score"] = trend_score
    regime["range_score"] = range_score
    regime["stress_score"] = stress_score

    snapshots: list[RegimeFeatureSnapshot] = []
    for ts, row in regime.iterrows():
        snapshots.append(RegimeFeatureSnapshot(ts=int(ts.timestamp() * 1000), symbol=symbol, **{k: _snapshot_value(v) for k, v in row.items()}))
    return snapshots


def _load_crowding_base(raw_paths: Iterable[str], *, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    funding_rows: list[dict] = []
    oi_rows: list[dict] = []
    ratio_rows: list[dict] = []
    for event in iter_raw_events_sorted(raw_paths):
        if event.source != "bybit_rest" or event.symbol != symbol:
            continue
        funding_rows.extend(parse_funding_history_event(event))
        oi_rows.extend(parse_open_interest_event(event))
        ratio_rows.extend(parse_account_ratio_event(event))

    def _frame(rows: list[dict], ts_col: str = "ts") -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).drop_duplicates(subset=[ts_col], keep="last").sort_values(ts_col)
        df.index = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        return df.drop(columns=[ts_col])

    return _frame(funding_rows), _frame(oi_rows), _frame(ratio_rows)


def build_crowding_features(
    raw_paths: Iterable[str],
    premium_bars: Iterable[PriceBar],
    micro_buckets: Iterable[MicroBucket1s],
    *,
    symbol: str,
) -> list[CrowdingFeatureSnapshot]:
    funding_df, oi_df, ratio_df = _load_crowding_base(raw_paths, symbol=symbol)
    premium_df = _bars_to_frame(premium_bars)
    micro_df = _micro_to_frame(micro_buckets)

    frames: list[pd.DataFrame] = []
    if not premium_df.empty:
        frames.append(pd.DataFrame(index=premium_df.index))
    if not oi_df.empty:
        frames.append(pd.DataFrame(index=oi_df.index))
    if not funding_df.empty:
        frames.append(pd.DataFrame(index=funding_df.index))
    if not ratio_df.empty:
        frames.append(pd.DataFrame(index=ratio_df.index))
    if not micro_df.empty:
        frames.append(pd.DataFrame(index=micro_df.index))
    if not frames:
        return []

    base_index = frames[0].index
    for frame in frames[1:]:
        base_index = base_index.union(frame.index)
    base_index = pd.DatetimeIndex(sorted(base_index)).unique()
    base = pd.DataFrame(index=base_index)
    base = base.resample("15min").last()

    if not funding_df.empty:
        funding_15 = funding_df["funding_rate"].resample("15min").ffill().reindex(base.index, method="ffill")
        base["funding_8h"] = funding_15
        base["funding_z_7d"] = rolling_zscore(funding_15, 96 * 7, min_periods=96)
    else:
        base["funding_8h"] = np.nan
        base["funding_z_7d"] = np.nan

    if not premium_df.empty:
        prem_15 = premium_df["close"].resample("15min").last().reindex(base.index, method="ffill")
        base["premium_index_15m"] = prem_15
        base["premium_z_7d"] = rolling_zscore(prem_15, 96 * 7, min_periods=96)
    else:
        base["premium_index_15m"] = np.nan
        base["premium_z_7d"] = np.nan

    if not oi_df.empty:
        oi_15 = oi_df["open_interest"].resample("15min").ffill().reindex(base.index, method="ffill")
        base["oi_level"] = oi_15
        base["oi_change_1h"] = oi_15.pct_change(4)
        base["oi_change_4h"] = oi_15.pct_change(16)
        base["oi_change_4h_z"] = rolling_zscore(base["oi_change_4h"], 96 * 7, min_periods=96)
    else:
        base["oi_level"] = np.nan
        base["oi_change_1h"] = np.nan
        base["oi_change_4h"] = np.nan
        base["oi_change_4h_z"] = np.nan

    if not ratio_df.empty:
        ratio_15 = ratio_df[["buy_ratio", "sell_ratio"]].resample("15min").ffill().reindex(base.index, method="ffill")
        long_short_ratio = ratio_15["buy_ratio"] / ratio_15["sell_ratio"].replace(0, np.nan)
        base["long_short_ratio_1h"] = long_short_ratio
        base["long_short_ratio_z"] = rolling_zscore(np.log(long_short_ratio.replace(0, np.nan)), 96 * 7, min_periods=96)
    else:
        base["long_short_ratio_1h"] = np.nan
        base["long_short_ratio_z"] = np.nan

    if not micro_df.empty:
        micro_15 = micro_df.resample("15min").agg({
            "buy_volume": "sum",
            "sell_volume": "sum",
            "long_liq_notional": "sum",
            "short_liq_notional": "sum",
        }).reindex(base.index, fill_value=0.0)
        taker_ratio = (micro_15["buy_volume"] + EPS) / (micro_15["sell_volume"] + EPS)
        base["taker_ratio_z"] = rolling_zscore(np.log(taker_ratio), 96 * 7, min_periods=96)
        long_liq_1h = micro_15["long_liq_notional"].rolling(4, min_periods=1).sum()
        short_liq_1h = micro_15["short_liq_notional"].rolling(4, min_periods=1).sum()
        base["liq_longs_z_1h"] = rolling_zscore(long_liq_1h, 96 * 7, min_periods=96)
        base["liq_shorts_z_1h"] = rolling_zscore(short_liq_1h, 96 * 7, min_periods=96)
    else:
        base["taker_ratio_z"] = np.nan
        base["liq_longs_z_1h"] = np.nan
        base["liq_shorts_z_1h"] = np.nan

    base["crowding_long_score"] = (
        0.30 * base["funding_z_7d"].fillna(0)
        + 0.25 * base["premium_z_7d"].fillna(0)
        + 0.20 * base["oi_change_4h_z"].fillna(0)
        + 0.15 * base["long_short_ratio_z"].fillna(0)
        + 0.10 * base["taker_ratio_z"].fillna(0)
    )
    base["crowding_short_score"] = (
        -0.30 * base["funding_z_7d"].fillna(0)
        -0.25 * base["premium_z_7d"].fillna(0)
        -0.20 * base["oi_change_4h_z"].fillna(0)
        -0.15 * base["long_short_ratio_z"].fillna(0)
        -0.10 * base["taker_ratio_z"].fillna(0)
    )
    base["veto_long"] = (base["crowding_long_score"] >= 2.5) | ((base["funding_z_7d"] >= 3.0) & (base["premium_z_7d"] >= 3.0))
    base["veto_short"] = (base["crowding_short_score"] >= 2.5) | ((base["funding_z_7d"] <= -3.0) & (base["premium_z_7d"] <= -3.0))

    snapshots: list[CrowdingFeatureSnapshot] = []
    for ts, row in base.iterrows():
        payload = {k: (None if pd.isna(v) else float(v)) for k, v in row.items() if k not in {"veto_long", "veto_short"}}
        snapshots.append(
            CrowdingFeatureSnapshot(
                ts=int(ts.timestamp() * 1000),
                symbol=symbol,
                veto_long=bool(row.get("veto_long", False)),
                veto_short=bool(row.get("veto_short", False)),
                **payload,
            )
        )
    return snapshots


def build_micro_features(micro_buckets: Iterable[MicroBucket1s], *, symbol: str) -> list[MicroFeatureSnapshot]:
    micro = _micro_to_frame(micro_buckets)
    if micro.empty:
        return []
    for col in [
        "ofi_proxy",
        "cum_delta",
        "spread_bps",
        "bookimb_l1",
        "bookimb_l5",
        "bookimb_l10",
        "top10_depth_usd",
        "depth_decay",
        "vwap_mid_dev",
        "replenish_notional",
        "cancel_notional",
        "mid_price",
        "last_trade_price",
    ]:
        if col not in micro:
            micro[col] = np.nan
    micro["mid_price"] = micro["mid_price"].ffill()
    micro["spread_bps"] = micro["spread_bps"].ffill()
    micro["bookimb_l1"] = micro["bookimb_l1"].ffill()
    micro["bookimb_l5"] = micro["bookimb_l5"].ffill()
    micro["bookimb_l10"] = micro["bookimb_l10"].ffill()
    micro["top10_depth_usd"] = micro["top10_depth_usd"].ffill()
    micro["depth_decay"] = micro["depth_decay"].ffill()
    price = micro["last_trade_price"].where(micro["last_trade_price"].notna(), micro["mid_price"]).ffill()
    log_ret = np.log(price / price.shift(1))

    out = pd.DataFrame(index=micro.index)
    out["ofi_10s"] = micro["ofi_proxy"].rolling(10, min_periods=1).sum()
    out["ofi_60s"] = micro["ofi_proxy"].rolling(60, min_periods=1).sum()
    out["ofi_300s"] = micro["ofi_proxy"].rolling(300, min_periods=10).sum()
    out["cum_delta_60s"] = micro["cum_delta"].rolling(60, min_periods=1).sum()
    out["spread_bps"] = micro["spread_bps"]
    out["spread_z"] = rolling_zscore(micro["spread_bps"], 300, min_periods=30).fillna(0.0)
    out["bookimb_l1"] = micro["bookimb_l1"]
    out["bookimb_l5"] = micro["bookimb_l5"]
    out["bookimb_l10"] = micro["bookimb_l10"]
    out["top10_depth_usd"] = micro["top10_depth_usd"]
    out["depth_decay"] = micro["depth_decay"]
    out["vwap_mid_dev_30s"] = micro["vwap_mid_dev"].rolling(30, min_periods=1).mean()
    out["vwap_mid_dev_30s_z"] = rolling_zscore(out["vwap_mid_dev_30s"], 300, min_periods=30).fillna(0.0)
    add_30 = micro["replenish_notional"].rolling(30, min_periods=1).sum()
    cancel_30 = micro["cancel_notional"].rolling(30, min_periods=1).sum()
    out["replenish_rate_30s"] = add_30 / cancel_30.replace(0, np.nan)
    out["cancel_add_ratio_30s"] = cancel_30 / add_30.replace(0, np.nan)
    out["micro_vol_60s"] = log_ret.rolling(60, min_periods=10).std(ddof=0)
    out["median_bookimb_l10_60s"] = micro["bookimb_l10"].rolling(60, min_periods=10).median()
    out["gate_pass_long"] = (
        (out["ofi_60s"] > 0)
        & (out["median_bookimb_l10_60s"] > 0.05)
        & (out["spread_z"] < 1.5)
        & (out["vwap_mid_dev_30s_z"].abs() < 2)
    )
    out["gate_pass_short"] = (
        (out["ofi_60s"] < 0)
        & (out["median_bookimb_l10_60s"] < -0.05)
        & (out["spread_z"] < 1.5)
        & (out["vwap_mid_dev_30s_z"].abs() < 2)
    )

    snapshots: list[MicroFeatureSnapshot] = []
    for ts, row in out.iterrows():
        payload = {k: (None if pd.isna(v) else float(v)) for k, v in row.items() if k not in {"gate_pass_long", "gate_pass_short"}}
        snapshots.append(
            MicroFeatureSnapshot(
                ts=int(ts.timestamp() * 1000),
                symbol=symbol,
                gate_pass_long=bool(row.get("gate_pass_long", False)),
                gate_pass_short=bool(row.get("gate_pass_short", False)),
                **payload,
            )
        )
    return snapshots

