from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from btc_alert_engine.features.indicators import adx, atr, ema, resample_ohlcv, rolling_percentile_of_last, rolling_zscore, softmax_scores
from btc_alert_engine.normalize.bybit_rest import parse_account_ratio_event, parse_funding_history_event, parse_open_interest_event
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


def load_price_bars(paths: Iterable[str]) -> list[PriceBar]:
    return [PriceBar.model_validate(record) for record in iter_json_records(paths)]


def load_micro_buckets(paths: Iterable[str]) -> list[MicroBucket1s]:
    return [MicroBucket1s.model_validate(record) for record in iter_json_records(paths)]


def _breakout_state_1h(bars_1h: pd.DataFrame, atr14_1h: pd.Series, *, side: str) -> pd.DataFrame:
    close_1h = bars_1h["close"]
    if side == "long":
        trigger_level = bars_1h["high"].shift(1).rolling(20, min_periods=20).max()
        trigger_flag = close_1h > trigger_level
    else:
        trigger_level = bars_1h["low"].shift(1).rolling(20, min_periods=20).min()
        trigger_flag = close_1h < trigger_level

    age_vals: list[float] = []
    impulse_vals: list[float] = []
    depth_vals: list[float] = []
    pullback_bars_vals: list[float] = []
    level_vals: list[float] = []

    active_idx: int | None = None
    active_level = float("nan")
    impulse = float("nan")
    extreme_since_trigger = float("nan")
    bars_since_extreme = float("nan")

    for i, (_, row) in enumerate(bars_1h.iterrows()):
        level = float(trigger_level.iloc[i]) if not math.isnan(float(trigger_level.iloc[i])) else float("nan")
        if bool(trigger_flag.iloc[i]) and not math.isnan(level):
            active_idx = i
            active_level = level
            current_atr = float(atr14_1h.iloc[i]) if not math.isnan(float(atr14_1h.iloc[i])) else float("nan")
            if current_atr and not math.isnan(current_atr):
                if side == "long":
                    impulse = (float(row["close"]) - active_level) / current_atr
                    extreme_since_trigger = float(row["high"])
                else:
                    impulse = (active_level - float(row["close"])) / current_atr
                    extreme_since_trigger = float(row["low"])
            else:
                impulse = float("nan")
                extreme_since_trigger = float(row["high"] if side == "long" else row["low"])
            bars_since_extreme = 0.0
        elif active_idx is not None:
            age = i - active_idx
            if age > 10:
                active_idx = None
                active_level = float("nan")
                impulse = float("nan")
                extreme_since_trigger = float("nan")
                bars_since_extreme = float("nan")
            else:
                current_extreme = float(row["high"] if side == "long" else row["low"])
                if (side == "long" and current_extreme >= extreme_since_trigger) or (side == "short" and current_extreme <= extreme_since_trigger):
                    extreme_since_trigger = current_extreme
                    bars_since_extreme = 0.0
                else:
                    bars_since_extreme = 0.0 if math.isnan(bars_since_extreme) else bars_since_extreme + 1.0

        if active_idx is None:
            age_vals.append(float("nan"))
            impulse_vals.append(float("nan"))
            depth_vals.append(float("nan"))
            pullback_bars_vals.append(float("nan"))
            level_vals.append(float("nan"))
            continue

        age_vals.append(float(i - active_idx))
        impulse_vals.append(impulse)
        level_vals.append(active_level)
        if side == "long":
            denom = max(extreme_since_trigger - active_level, EPS)
            depth = max(extreme_since_trigger - float(row["close"]), 0.0) / denom
        else:
            denom = max(active_level - extreme_since_trigger, EPS)
            depth = max(float(row["close"]) - extreme_since_trigger, 0.0) / denom
        depth_vals.append(depth)
        pullback_bars_vals.append(bars_since_extreme)

    result = pd.DataFrame(index=bars_1h.index)
    if side == "long":
        result["breakout_age_1h"] = age_vals
        result["impulse_atr_1h"] = impulse_vals
        result["pullback_depth_frac"] = depth_vals
        result["pullback_bars"] = pullback_bars_vals
        result["breakout_level"] = level_vals
    else:
        result["breakdown_age_1h"] = age_vals
        result["downside_impulse_atr_1h"] = impulse_vals
        result["rebound_depth_frac"] = depth_vals
        result["rebound_bars"] = pullback_bars_vals
        result["breakdown_level"] = level_vals
    return result


def _trend_feature_frame(trade_bars: Iterable[PriceBar]) -> pd.DataFrame:
    trade_15m = _bars_to_frame(trade_bars)
    if trade_15m.empty:
        return pd.DataFrame()

    bars_1h = resample_ohlcv(trade_15m, "1h")
    bars_4h = resample_ohlcv(trade_15m, "4h")

    # 4h trend features
    close_4h = bars_4h["close"]
    ema50_4h = ema(close_4h, 50)
    ema200_4h = ema(close_4h, 200)
    atr14_4h = atr(bars_4h, 14)
    adx14_4h = adx(bars_4h, 14)
    atr_pctile_90d = rolling_percentile_of_last(atr14_4h, window=90 * 6)  # 90d of 4h bars

    four_h = pd.DataFrame(index=bars_4h.index)
    four_h["ret_4h_6"] = close_4h.pct_change(6)
    four_h["ema50_4h_gap"] = (close_4h - ema50_4h) / close_4h.replace(0, np.nan)
    four_h["ema50_4h_slope"] = ema50_4h.pct_change(5, fill_method=None)
    four_h["ema200_4h_slope"] = ema200_4h.pct_change(5, fill_method=None)
    four_h["adx14_4h"] = adx14_4h
    four_h["atr_pctile_90d"] = atr_pctile_90d
    four_h["ema50_4h"] = ema50_4h

    # 1h breakout / pullback state
    close_1h = bars_1h["close"]
    atr14_1h = atr(bars_1h, 14)
    one_h = pd.DataFrame(index=bars_1h.index)
    one_h["ret_1h_4"] = close_1h.pct_change(4)
    one_h = one_h.join(_breakout_state_1h(bars_1h, atr14_1h, side="long"), how="left")
    one_h = one_h.join(_breakout_state_1h(bars_1h, atr14_1h, side="short"), how="left")

    # Join onto 15m close timeline using last fully closed higher timeframe bar.
    trend = pd.DataFrame(index=trade_15m.index)
    trend["ret_15m_1"] = trade_15m["close"].pct_change(1)
    trend = trend.join(one_h.reindex(trade_15m.index, method="ffill"), how="left")
    trend = trend.join(four_h.reindex(trade_15m.index, method="ffill"), how="left")
    trend["dist_to_breakout_level"] = (trade_15m["close"] - trend["breakout_level"]) / trade_15m["close"].replace(0, np.nan)
    trend["dist_to_breakdown_level"] = (trade_15m["close"] - trend["breakdown_level"]) / trade_15m["close"].replace(0, np.nan)
    trend["dist_to_ema50_4h"] = (trade_15m["close"] - trend["ema50_4h"]) / trade_15m["close"].replace(0, np.nan)
    trend["symbol"] = "BTCUSDT"
    return trend[[
        "ret_15m_1",
        "ret_1h_4",
        "ret_4h_6",
        "ema50_4h_gap",
        "ema50_4h_slope",
        "ema200_4h_slope",
        "adx14_4h",
        "atr_pctile_90d",
        "breakout_age_1h",
        "impulse_atr_1h",
        "pullback_depth_frac",
        "pullback_bars",
        "dist_to_breakout_level",
        "breakdown_age_1h",
        "downside_impulse_atr_1h",
        "rebound_depth_frac",
        "rebound_bars",
        "dist_to_breakdown_level",
        "dist_to_ema50_4h",
    ]]

def build_trend_features(trade_bars: Iterable[PriceBar], *, symbol: str) -> list[TrendFeatureSnapshot]:
    frame = _trend_feature_frame(trade_bars)
    snapshots: list[TrendFeatureSnapshot] = []
    for ts, row in frame.iterrows():
        snapshots.append(TrendFeatureSnapshot(ts=int(ts.timestamp() * 1000), symbol=symbol, **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()}))
    return snapshots


def build_regime_features(trade_bars: Iterable[PriceBar], index_bars: Iterable[PriceBar], premium_bars: Iterable[PriceBar], *, symbol: str) -> list[RegimeFeatureSnapshot]:
    trade_15m = _bars_to_frame(trade_bars)
    index_15m = _bars_to_frame(index_bars)
    premium_15m = _bars_to_frame(premium_bars)
    if trade_15m.empty:
        return []
    trend = _trend_feature_frame(trade_bars)

    close = trade_15m["close"]
    log_ret = np.log(close / close.shift(1))
    rv_1d = log_ret.rolling(96, min_periods=24).std(ddof=0) * np.sqrt(96)
    rv_7d = log_ret.rolling(96 * 7, min_periods=96).std(ddof=0) * np.sqrt(96)
    ret_std_1d = log_ret.rolling(96, min_periods=24).std(ddof=0)
    jump_intensity_1d = (log_ret.abs() > (3 * ret_std_1d)).astype(float).rolling(96, min_periods=24).mean()

    regime = pd.DataFrame(index=trade_15m.index)
    regime["rv_1d"] = rv_1d
    regime["rv_7d"] = rv_7d
    regime["atr_pctile_90d"] = trend["atr_pctile_90d"]
    regime["jump_intensity_1d"] = jump_intensity_1d

    if not index_15m.empty:
        idx_close = index_15m["close"].reindex(trade_15m.index, method="ffill")
        regime["mark_index_gap"] = (close - idx_close) / idx_close.replace(0, np.nan)
    else:
        regime["mark_index_gap"] = np.nan

    if not premium_15m.empty:
        prem_close = premium_15m["close"].reindex(trade_15m.index, method="ffill")
        regime["premium_index_15m"] = prem_close
        regime["premium_z_7d"] = rolling_zscore(prem_close, 96 * 7, min_periods=96)
    else:
        regime["premium_index_15m"] = np.nan
        regime["premium_z_7d"] = np.nan

    trend_raw = (
        1.2 * (trend["adx14_4h"].fillna(0) / 25.0)
        + 2.0 * trend["ema50_4h_gap"].abs().fillna(0) * 100
        + 0.5 * trend["ret_4h_6"].abs().fillna(0) * 10
        - 0.5 * regime["atr_pctile_90d"].fillna(0) / 100.0
    )
    range_raw = (
        -1.5 * trend["ema50_4h_gap"].abs().fillna(0) * 100
        - 0.8 * (trend["adx14_4h"].fillna(0) / 25.0)
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
        snapshots.append(RegimeFeatureSnapshot(ts=int(ts.timestamp() * 1000), symbol=symbol, **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()}))
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
    out["gate_pass"] = out["gate_pass_long"]

    snapshots: list[MicroFeatureSnapshot] = []
    for ts, row in out.iterrows():
        payload = {
            k: (None if pd.isna(v) else float(v))
            for k, v in row.items()
            if k not in {"gate_pass", "gate_pass_long", "gate_pass_short"}
        }
        snapshots.append(
            MicroFeatureSnapshot(
                ts=int(ts.timestamp() * 1000),
                symbol=symbol,
                gate_pass=bool(row.get("gate_pass", False)),
                gate_pass_long=bool(row.get("gate_pass_long", False)),
                gate_pass_short=bool(row.get("gate_pass_short", False)),
                **payload,
            )
        )
    return snapshots
