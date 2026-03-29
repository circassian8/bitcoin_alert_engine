from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from btc_alert_engine.schemas import PriceBar


def bars_to_ohlcv_frame(bars: Iterable[PriceBar]) -> pd.DataFrame:
    """Convert PriceBars to a DatetimeIndex DataFrame with OHLCV columns only."""
    records = [bar.model_dump(mode="json") if isinstance(bar, PriceBar) else bar for bar in bars]
    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume", "turnover"]]


def reindex_ffill(source: pd.DataFrame | pd.Series, target_index: pd.DatetimeIndex) -> pd.DataFrame | pd.Series:
    """Reindex *source* onto *target_index* with forward-fill semantics.

    Equivalent to the deprecated ``source.reindex(target_index, method="ffill")``
    but avoids the FutureWarning by merging both indices first.
    """
    combined = source.index.union(target_index).sort_values()
    result = source.reindex(combined).ffill()
    if hasattr(result, "infer_objects"):
        result = result.infer_objects(copy=False)
    return result.reindex(target_index)


def bars_to_full_frame(bars: Iterable[PriceBar]) -> pd.DataFrame:
    """Convert PriceBars to a DatetimeIndex DataFrame retaining ts, symbol, and OHLCV columns."""
    rows = [bar.model_dump(mode="json") if isinstance(bar, PriceBar) else bar for bar in bars]
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "turnover"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df
