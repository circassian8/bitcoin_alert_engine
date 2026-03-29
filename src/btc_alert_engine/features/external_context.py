from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from btc_alert_engine.features.common import bars_to_full_frame, reindex_ffill
from btc_alert_engine.features.indicators import rolling_zscore
from btc_alert_engine.schemas import (
    CoinGlassAggregateFeatureSnapshot,
    CryptoQuantOnChainFeatureSnapshot,
    DeribitOptionsFeatureSnapshot,
    GlassnodeOnChainFeatureSnapshot,
    GlassnodeOptionsFeatureSnapshot,
    MacroVetoFeatureSnapshot,
    PriceBar,
    RawEvent,
)
from btc_alert_engine.storage.raw_ndjson import iter_raw_events_sorted


def _bars_to_frame(bars: Iterable[PriceBar]) -> pd.DataFrame:
    return bars_to_full_frame(bars)


def _as_utc_ts(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if number > 1e15:
                return pd.to_datetime(int(number), unit="ns", utc=True)
            if number > 1e12:
                return pd.to_datetime(int(number), unit="ms", utc=True)
            if number > 1e10:
                return pd.to_datetime(int(number), unit="s", utc=True)
            if number > 1e9:
                return pd.to_datetime(int(number), unit="s", utc=True)
            return pd.to_datetime(int(number), unit="ms", utc=True)
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception as exc:
        logger.debug("Failed to parse timestamp value %r: %s", value, exc)
        return None


def _unwrap_rows(payload: dict[str, Any]) -> list[Any]:
    root: Any = payload
    if isinstance(root, dict) and "result" in root:
        root = root["result"]
    if isinstance(root, dict):
        for key in ("data", "series", "list", "rows", "points", "values", "items"):
            if key in root:
                root = root[key]
                break
    if root is None:
        return []
    if isinstance(root, list):
        return list(root)
    return [root]


def _row_value(row: dict[str, Any], metric_name: str | None = None) -> float | None:
    candidates: list[str] = []
    if metric_name is not None:
        candidates.extend([metric_name, metric_name.lower(), metric_name.upper()])
    candidates.extend(["v", "value", "close", "c", "y", "result"])
    for key in candidates:
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except Exception as exc:
                logger.debug("Cannot convert row[%r]=%r to float: %s", key, row[key], exc)
                continue
    ts_like = {"t", "ts", "time", "timestamp", "date", "datetime"}
    numeric_values: list[float] = []
    for key, value in row.items():
        if key in ts_like or isinstance(value, (list, dict)) or value is None:
            continue
        try:
            numeric_values.append(float(value))
        except Exception:
            continue  # non-numeric field, expected
    if len(numeric_values) == 1:
        return numeric_values[0]
    return None


def _extract_single_metric_series(
    raw_paths: Iterable[str | Any],
    *,
    source_aliases: Iterable[str],
    topic_aliases: Iterable[str],
    metric_name: str | None = None,
) -> pd.Series:
    rows: list[tuple[pd.Timestamp, float]] = []
    source_aliases_l = [item.lower() for item in source_aliases]
    topic_aliases_l = [item.lower() for item in topic_aliases]

    for event in iter_raw_events_sorted(raw_paths):
        event: RawEvent
        source_l = event.source.lower()
        topic_l = event.topic.lower()
        if source_aliases_l and not any(alias in source_l for alias in source_aliases_l):
            continue
        if topic_aliases_l and not any(alias in topic_l for alias in topic_aliases_l):
            continue
        for row in _unwrap_rows(event.payload):
            ts: pd.Timestamp | None = None
            value: float | None = None
            if isinstance(row, dict):
                ts = _as_utc_ts(
                    row.get("t")
                    or row.get("timestamp")
                    or row.get("time")
                    or row.get("ts")
                    or row.get("date")
                    or event.exchange_ts
                    or event.local_received_ts
                )
                value = _row_value(row, metric_name)
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                ts = _as_utc_ts(row[0])
                try:
                    value = float(row[1])
                except Exception as exc:
                    logger.debug("Cannot convert row element to float: %s", exc)
                    value = None
            if ts is None or value is None or not np.isfinite(value):
                continue
            rows.append((ts, float(value)))

    if not rows:
        return pd.Series(dtype=float)
    series = pd.Series({ts: value for ts, value in rows}).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.astype(float)


def _extract_deribit_dvol_series(raw_paths: Iterable[str | Any], *, currency: str = "BTC") -> pd.Series:
    rows: list[tuple[pd.Timestamp, float]] = []
    currency_l = currency.lower()
    for event in iter_raw_events_sorted(raw_paths):
        if "deribit" not in event.source.lower():
            continue
        if currency_l not in event.topic.lower() and currency_l not in event.symbol.lower():
            continue
        if "volatility_index" not in event.topic.lower():
            continue
        for row in _unwrap_rows(event.payload):
            ts: pd.Timestamp | None = None
            close_value: float | None = None
            if isinstance(row, dict):
                ts = _as_utc_ts(row.get("t") or row.get("timestamp") or row.get("time") or row.get("ts") or event.exchange_ts)
                for key in ("close", "c", "value", "v"):
                    if row.get(key) is not None:
                        try:
                            close_value = float(row[key])
                            break
                        except Exception as exc:
                            logger.debug("Cannot convert dvol row[%r] to float: %s", key, exc)
                            continue
            elif isinstance(row, (list, tuple)):
                if len(row) >= 5:
                    ts = _as_utc_ts(row[0])
                    try:
                        close_value = float(row[4])
                    except Exception as exc:
                        logger.debug("Cannot convert dvol OHLCV close to float: %s", exc)
                        close_value = None
                elif len(row) >= 2:
                    ts = _as_utc_ts(row[0])
                    try:
                        close_value = float(row[1])
                    except Exception as exc:
                        logger.debug("Cannot convert dvol value to float: %s", exc)
                        close_value = None
            if ts is None or close_value is None or not np.isfinite(close_value):
                continue
            rows.append((ts, float(close_value)))
    if not rows:
        return pd.Series(dtype=float)
    series = pd.Series({ts: value for ts, value in rows}).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.astype(float)


def _align_series_map_to_trade_bars(
    trade_bars: Iterable[PriceBar],
    series_map: dict[str, pd.Series],
    *,
    lag: str | pd.Timedelta = "0s",
) -> pd.DataFrame:
    bars = _bars_to_frame(trade_bars)
    if bars.empty:
        return pd.DataFrame()
    lag_delta = pd.Timedelta(lag)
    aligned = pd.DataFrame(index=bars.index)
    for name, series in series_map.items():
        if series.empty:
            aligned[name] = np.nan
            continue
        ser = series.copy()
        ser.index = pd.DatetimeIndex(ser.index).tz_convert("UTC") + lag_delta
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        aligned[name] = reindex_ffill(ser, aligned.index)
    aligned["ts"] = bars["ts"].astype(int)
    return aligned


def _snapshots_from_frame(frame: pd.DataFrame, model_cls: type, *, symbol: str) -> list[Any]:
    if frame.empty:
        return []
    snapshots: list[Any] = []
    for _, row in frame.iterrows():
        payload: dict[str, Any] = {}
        for key, value in row.items():
            if key == "ts":
                continue
            if pd.isna(value):
                payload[key] = None
            elif isinstance(value, (np.bool_, bool)):
                payload[key] = bool(value)
            else:
                payload[key] = float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
        snapshots.append(model_cls(ts=int(row["ts"]), symbol=symbol, **payload))
    return snapshots


def build_options_deribit_features(raw_paths: Iterable[str | Any], trade_bars: Iterable[PriceBar], *, symbol: str) -> list[DeribitOptionsFeatureSnapshot]:
    dvol = _extract_deribit_dvol_series(raw_paths, currency="BTC")
    if dvol.empty:
        return []
    aligned = _align_series_map_to_trade_bars(trade_bars, {"dvol_level": dvol})
    if aligned.empty:
        return []
    aligned["dvol_z_30d"] = rolling_zscore(aligned["dvol_level"], 96 * 30, min_periods=96 * 3)
    aligned["dvol_change_1h"] = aligned["dvol_level"].pct_change(4)
    return _snapshots_from_frame(aligned[["ts", "dvol_level", "dvol_z_30d", "dvol_change_1h"]], DeribitOptionsFeatureSnapshot, symbol=symbol)


_MACRO_EVENT_TYPES = {
    "fomc": "fomc",
    "fed": "fomc",
    "cpi": "cpi",
    "inflation": "cpi",
    "employment": "nfp",
    "nfp": "nfp",
    "payroll": "nfp",
    "pce": "pce",
    "gdp": "gdp",
}

_DEFAULT_PRE_WINDOWS = {
    "fomc": 120.0,
    "cpi": 60.0,
    "nfp": 60.0,
    "pce": 45.0,
    "gdp": 45.0,
}

_DEFAULT_POST_WINDOWS = {
    "fomc": 180.0,
    "cpi": 90.0,
    "nfp": 90.0,
    "pce": 60.0,
    "gdp": 60.0,
}


def _normalize_macro_event_type(raw: str | None) -> str | None:
    if raw is None:
        return None
    lowered = raw.lower()
    for key, value in _MACRO_EVENT_TYPES.items():
        if key in lowered:
            return value
    return None


def _signed_minutes_to_nearest(index: pd.DatetimeIndex, event_times_ms: list[int]) -> pd.Series:
    if not event_times_ms:
        return pd.Series(index=index, dtype=float)
    idx_ms = (index.view("int64") // 1_000_000).astype(np.int64)
    event_arr = np.array(sorted(event_times_ms), dtype=np.int64)
    positions = np.searchsorted(event_arr, idx_ms)
    out = np.full(len(idx_ms), np.nan, dtype=float)
    for i, (ts_ms, pos) in enumerate(zip(idx_ms, positions, strict=False)):
        candidates: list[int] = []
        if pos < len(event_arr):
            candidates.append(int(event_arr[pos]))
        if pos > 0:
            candidates.append(int(event_arr[pos - 1]))
        if not candidates:
            continue
        nearest = min(candidates, key=lambda event_ms: abs(event_ms - int(ts_ms)))
        out[i] = (nearest - int(ts_ms)) / 60000.0
    return pd.Series(out, index=index, dtype=float)


def build_macro_veto_features(
    raw_paths: Iterable[str | Any],
    trade_bars: Iterable[PriceBar],
    *,
    symbol: str,
    pre_event_windows: dict[str, float] | None = None,
    post_event_windows: dict[str, float] | None = None,
) -> list[MacroVetoFeatureSnapshot]:
    bars = _bars_to_frame(trade_bars)
    if bars.empty:
        return []
    event_times: dict[str, list[int]] = defaultdict(list)
    for event in iter_raw_events_sorted(raw_paths):
        topic_l = event.topic.lower()
        if not topic_l.startswith("macro.") and "macro" not in event.source.lower():
            continue
        event_type = _normalize_macro_event_type(event.payload.get("event_type") if isinstance(event.payload, dict) else None)
        if event_type is None:
            event_type = _normalize_macro_event_type(topic_l)
        if event_type is None:
            continue
        ts = event.exchange_ts or event.local_received_ts
        event_times[event_type].append(int(ts))

    if not event_times:
        return []

    pre = dict(_DEFAULT_PRE_WINDOWS)
    post = dict(_DEFAULT_POST_WINDOWS)
    if pre_event_windows:
        pre.update(pre_event_windows)
    if post_event_windows:
        post.update(post_event_windows)

    frame = pd.DataFrame(index=bars.index)
    name_map = {
        "fomc": "mins_to_fomc",
        "cpi": "mins_to_cpi",
        "nfp": "mins_to_nfp",
        "pce": "mins_to_pce",
        "gdp": "mins_to_gdp",
    }
    for event_type, column in name_map.items():
        frame[column] = _signed_minutes_to_nearest(frame.index, event_times.get(event_type, []))

    veto_active: list[bool] = []
    veto_event_type: list[str | None] = []
    for _, row in frame.iterrows():
        active_types: list[tuple[str, float]] = []
        for event_type, column in name_map.items():
            value = row[column]
            if pd.isna(value):
                continue
            if (0.0 <= value <= pre[event_type]) or (-post[event_type] <= value < 0.0):
                active_types.append((event_type, abs(float(value))))
        if active_types:
            active_types.sort(key=lambda item: item[1])
            veto_active.append(True)
            veto_event_type.append(active_types[0][0])
        else:
            veto_active.append(False)
            veto_event_type.append(None)
    frame["veto_active"] = veto_active
    frame["veto_event_type"] = veto_event_type
    frame["ts"] = bars["ts"].astype(int)
    return _snapshots_from_frame(frame[["ts", *name_map.values(), "veto_active", "veto_event_type"]], MacroVetoFeatureSnapshot, symbol=symbol)


_GLASSNODE_OPTIONS_ALIASES = {
    "iv_7d": ["options_iv_7d", "iv_7d"],
    "iv_30d": ["options_iv_30d", "iv_30d"],
    "skew_25d": ["options_skew_25d", "skew_25d"],
    "put_call_iv_spread": ["options_put_call_iv_spread", "put_call_iv_spread"],
}


_GLASSNODE_ONCHAIN_ALIASES = {
    "exchange_balance": ["exchange_balance"],
    "exchange_netflow": ["exchange_netflow", "netflow"],
    "sopr": ["sopr"],
    "nrpl": ["nrpl"],
    "etf_netflow": ["etf_netflow", "etf_flows", "etf_flow"],
}


_CRYPTOQUANT_ALIASES = {
    "reserve": ["reserve"],
    "inflow": ["inflow"],
    "outflow": ["outflow"],
    "inflow_spot_exchange": ["inflow_spot_exchange", "spot_exchange_inflow", "inflow_spot"],
    "inflow_derivative_exchange": ["inflow_derivative_exchange", "derivative_exchange_inflow", "inflow_derivative"],
}


_COINGLASS_ALIASES = {
    "btc_agg_oi": ["btc_agg_oi", "aggregated_open_interest", "agg_open_interest", "open_interest"],
    "btc_agg_funding": ["btc_agg_funding", "aggregated_funding", "agg_funding", "funding"],
    "btc_global_account_ratio": ["btc_global_account_ratio", "global_account_ratio", "account_ratio"],
}


def _build_multi_metric_snapshots(
    trade_bars: Iterable[PriceBar],
    *,
    source_aliases: Iterable[str],
    metric_aliases: dict[str, list[str]],
    raw_paths: Iterable[str | Any],
    symbol: str,
    model_cls: type,
    lag: str,
    postprocess: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> list[Any]:
    series_map: dict[str, pd.Series] = {}
    for metric, aliases in metric_aliases.items():
        series_map[metric] = _extract_single_metric_series(
            raw_paths,
            source_aliases=source_aliases,
            topic_aliases=aliases,
            metric_name=metric,
        )
    if not any(not series.empty for series in series_map.values()):
        return []
    frame = _align_series_map_to_trade_bars(trade_bars, series_map, lag=lag)
    if frame.empty:
        return []
    if postprocess is not None:
        frame = postprocess(frame)
    columns = ["ts", *[col for col in frame.columns if col != "ts"]]
    return _snapshots_from_frame(frame[columns], model_cls, symbol=symbol)


def build_options_glassnode_features(raw_paths: Iterable[str | Any], trade_bars: Iterable[PriceBar], *, symbol: str) -> list[GlassnodeOptionsFeatureSnapshot]:
    def _post(frame: pd.DataFrame) -> pd.DataFrame:
        frame["iv_term_slope"] = frame["iv_30d"] - frame["iv_7d"]
        return frame

    return _build_multi_metric_snapshots(
        trade_bars,
        source_aliases=["glassnode"],
        metric_aliases=_GLASSNODE_OPTIONS_ALIASES,
        raw_paths=raw_paths,
        symbol=symbol,
        model_cls=GlassnodeOptionsFeatureSnapshot,
        lag="12h",
        postprocess=_post,
    )


def build_onchain_glassnode_features(raw_paths: Iterable[str | Any], trade_bars: Iterable[PriceBar], *, symbol: str) -> list[GlassnodeOnChainFeatureSnapshot]:
    return _build_multi_metric_snapshots(
        trade_bars,
        source_aliases=["glassnode"],
        metric_aliases=_GLASSNODE_ONCHAIN_ALIASES,
        raw_paths=raw_paths,
        symbol=symbol,
        model_cls=GlassnodeOnChainFeatureSnapshot,
        lag="24h",
    )


def build_onchain_cryptoquant_features(raw_paths: Iterable[str | Any], trade_bars: Iterable[PriceBar], *, symbol: str) -> list[CryptoQuantOnChainFeatureSnapshot]:
    return _build_multi_metric_snapshots(
        trade_bars,
        source_aliases=["cryptoquant"],
        metric_aliases=_CRYPTOQUANT_ALIASES,
        raw_paths=raw_paths,
        symbol=symbol,
        model_cls=CryptoQuantOnChainFeatureSnapshot,
        lag="24h",
    )


def build_aggregate_derivs_coinglass_features(raw_paths: Iterable[str | Any], trade_bars: Iterable[PriceBar], *, symbol: str) -> list[CoinGlassAggregateFeatureSnapshot]:
    return _build_multi_metric_snapshots(
        trade_bars,
        source_aliases=["coinglass"],
        metric_aliases=_COINGLASS_ALIASES,
        raw_paths=raw_paths,
        symbol=symbol,
        model_cls=CoinGlassAggregateFeatureSnapshot,
        lag="2h",
    )
