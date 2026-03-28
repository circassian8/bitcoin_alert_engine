from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_series = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_series.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_series.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_p = min_periods or max(5, window // 4)
    mean = series.rolling(window=window, min_periods=min_p).mean()
    std = series.rolling(window=window, min_periods=min_p).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def rolling_percentile_of_last(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_p = min_periods or max(10, window // 10)

    def _percentile(values: np.ndarray) -> float:
        vals = values[~np.isnan(values)]
        if len(vals) == 0:
            return float("nan")
        last = vals[-1]
        return float((vals <= last).sum() / len(vals) * 100.0)

    return series.rolling(window=window, min_periods=min_p).apply(_percentile, raw=True)


def softmax_scores(*series: pd.Series) -> list[pd.Series]:
    frame = pd.concat(series, axis=1)
    frame = frame.astype(float)
    max_vals = frame.max(axis=1)
    exps = np.exp(frame.sub(max_vals, axis=0))
    denom = exps.sum(axis=1).replace(0, np.nan)
    outputs = exps.div(denom, axis=0)
    return [outputs.iloc[:, i] for i in range(outputs.shape[1])]


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "turnover": "sum",
    }
    result = df.resample(freq, label="right", closed="right").agg(agg)
    return result.dropna(subset=["open", "high", "low", "close"])
