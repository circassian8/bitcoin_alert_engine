from __future__ import annotations

from decimal import Decimal
import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RawEvent(BaseModel):
    source: str
    topic: str
    symbol: str
    exchange_ts: int | None = None
    local_received_ts: int
    payload: dict[str, Any]
    connection_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BookLevel(BaseModel):
    price: Decimal
    size: Decimal

    @field_validator("price", "size", mode="before")
    @classmethod
    def _to_decimal(cls, value: Any) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))


class BybitOrderBookMessage(BaseModel):
    topic: str
    message_type: Literal["snapshot", "delta"]
    ts: int
    symbol: str
    update_id: int
    seq: int | None = None
    cts: int | None = None
    bids: list[BookLevel]
    asks: list[BookLevel]


class NormalizedTrade(BaseModel):
    ts: int
    symbol: str
    price: Decimal
    size: Decimal
    side: Literal["Buy", "Sell"]
    trade_id: str
    seq: int | None = None
    tick_direction: str | None = None
    block_trade: bool = False
    rpi_trade: bool = False

    @field_validator("price", "size", mode="before")
    @classmethod
    def _to_decimal(cls, value: Any) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))


class NormalizedLiquidation(BaseModel):
    ts: int
    symbol: str
    side: Literal["Buy", "Sell"]
    size: Decimal
    bankruptcy_price: Decimal

    @field_validator("size", "bankruptcy_price", mode="before")
    @classmethod
    def _to_decimal(cls, value: Any) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))


class TopOfBookState(BaseModel):
    ts: int
    symbol: str
    best_bid_price: Decimal | None = None
    best_bid_size: Decimal | None = None
    best_ask_price: Decimal | None = None
    best_ask_size: Decimal | None = None
    spread: Decimal | None = None
    update_id: int | None = None
    seq: int | None = None

    @property
    def has_book(self) -> bool:
        return self.best_bid_price is not None and self.best_ask_price is not None


class PriceBar(BaseModel):
    ts: int
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    turnover: float = 0.0
    source_topic: str | None = None

    @field_validator("open", "high", "low", "close", "volume", "turnover", mode="before")
    @classmethod
    def _to_float(cls, value: Any) -> float:
        if value is None:
            return 0.0
        return float(value)


class MicroBucket1s(BaseModel):
    ts: int
    symbol: str
    best_bid_price: float | None = None
    best_bid_size: float | None = None
    best_ask_price: float | None = None
    best_ask_size: float | None = None
    best_bid_low: float | None = None
    best_bid_high: float | None = None
    best_ask_low: float | None = None
    best_ask_high: float | None = None
    mid_price: float | None = None
    spread: float | None = None
    spread_bps: float | None = None
    trade_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    vwap: float | None = None
    vwap_mid_dev: float = 0.0
    first_trade_price: float | None = None
    last_trade_price: float | None = None
    trade_low: float | None = None
    trade_high: float | None = None
    cum_delta: float = 0.0
    ofi_proxy: float = 0.0
    bookimb_l1: float | None = None
    bookimb_l5: float | None = None
    bookimb_l10: float | None = None
    top10_depth_usd: float | None = None
    depth_decay: float | None = None
    replenish_notional: float = 0.0
    cancel_notional: float = 0.0
    replenish_rate: float = 0.0
    cancel_add_ratio: float = 0.0
    micro_vol: float = 0.0
    long_liq_size: float = 0.0
    short_liq_size: float = 0.0
    long_liq_notional: float = 0.0
    short_liq_notional: float = 0.0
    quote_updates: int = 0


class TrendFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    ret_15m_1: float | None = None
    ret_1h_4: float | None = None
    ret_4h_6: float | None = None
    ema50_4h_gap: float | None = None
    ema50_4h_slope: float | None = None
    ema200_4h_slope: float | None = None
    adx14_4h: float | None = None
    atr_pctile_90d: float | None = None
    breakout_age_1h: float | None = None
    impulse_atr_1h: float | None = None
    pullback_depth_frac: float | None = None
    pullback_bars: float | None = None
    dist_to_breakout_level: float | None = None
    dist_to_ema50_4h: float | None = None
    breakdown_age_1h: float | None = None
    downside_impulse_atr_1h: float | None = None
    bounce_depth_frac: float | None = None
    bounce_bars: float | None = None
    dist_to_breakdown_level: float | None = None
    dist_below_ema50_4h: float | None = None


class RegimeFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    rv_1d: float | None = None
    rv_7d: float | None = None
    atr_pctile_90d: float | None = None
    jump_intensity_1d: float | None = None
    mark_index_gap: float | None = None
    premium_index_15m: float | None = None
    premium_z_7d: float | None = None
    stress_score: float | None = None
    trend_score: float | None = None
    range_score: float | None = None


class CrowdingFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    funding_8h: float | None = None
    funding_z_7d: float | None = None
    premium_index_15m: float | None = None
    premium_z_7d: float | None = None
    oi_level: float | None = None
    oi_change_1h: float | None = None
    oi_change_4h: float | None = None
    long_short_ratio_1h: float | None = None
    liq_longs_z_1h: float | None = None
    liq_shorts_z_1h: float | None = None
    crowding_long_score: float | None = None
    crowding_short_score: float | None = None
    veto_long: bool = False
    veto_short: bool = False


class MicroFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    ofi_10s: float | None = None
    ofi_60s: float | None = None
    ofi_300s: float | None = None
    cum_delta_60s: float | None = None
    spread_bps: float | None = None
    spread_z: float | None = None
    bookimb_l1: float | None = None
    bookimb_l5: float | None = None
    bookimb_l10: float | None = None
    top10_depth_usd: float | None = None
    depth_decay: float | None = None
    vwap_mid_dev_30s: float | None = None
    vwap_mid_dev_30s_z: float | None = None
    replenish_rate_30s: float | None = None
    cancel_add_ratio_30s: float | None = None
    micro_vol_60s: float | None = None
    median_bookimb_l10_60s: float | None = None
    gate_pass_long: bool = False
    gate_pass_short: bool = False

    @model_validator(mode="before")
    @classmethod
    def _compat_gate_pass(cls, values: Any) -> Any:
        if isinstance(values, dict) and "gate_pass" in values:
            gate_value = bool(values.get("gate_pass"))
            values.setdefault("gate_pass_long", gate_value)
            values.setdefault("gate_pass_short", False)
        return values


class DeribitOptionsFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    dvol_level: float | None = None
    dvol_z_30d: float | None = None
    dvol_change_1h: float | None = None


class GlassnodeOptionsFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    iv_7d: float | None = None
    iv_30d: float | None = None
    iv_term_slope: float | None = None
    skew_25d: float | None = None
    put_call_iv_spread: float | None = None


class GlassnodeOnChainFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    exchange_balance: float | None = None
    exchange_netflow: float | None = None
    sopr: float | None = None
    nrpl: float | None = None
    etf_netflow: float | None = None


class CryptoQuantOnChainFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    reserve: float | None = None
    inflow: float | None = None
    outflow: float | None = None
    inflow_spot_exchange: float | None = None
    inflow_derivative_exchange: float | None = None


class CoinGlassAggregateFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    btc_agg_oi: float | None = None
    btc_agg_funding: float | None = None
    btc_global_account_ratio: float | None = None


class MacroVetoFeatureSnapshot(BaseModel):
    ts: int
    symbol: str
    mins_to_fomc: float | None = None
    mins_to_cpi: float | None = None
    mins_to_nfp: float | None = None
    mins_to_pce: float | None = None
    mins_to_gdp: float | None = None
    veto_active: bool = False
    veto_event_type: str | None = None


class CandidateEvent(BaseModel):
    ts: int
    venue: str
    symbol: str
    module: Literal["continuation_v1", "stress_reversal_v0"]
    side: Literal["long", "short"]
    entry: float
    stop: float
    tp: float
    target_r_multiple: float | None = None
    timeout_bars: int
    rule_reasons: list[str]
    veto_reasons: list[str] = Field(default_factory=list)
    features: dict[str, float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_target_consistency(self) -> "CandidateEvent":
        if self.target_r_multiple is None:
            return self
        signal_risk = abs(self.entry - self.stop)
        if signal_risk <= 0:
            raise ValueError("CandidateEvent requires positive signal risk")
        if self.target_r_multiple <= 0:
            raise ValueError("CandidateEvent target_r_multiple must be positive")
        direction = 1.0 if self.side == "long" else -1.0
        expected_tp = self.entry + direction * self.target_r_multiple * signal_risk
        if not math.isclose(self.tp, expected_tp, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"CandidateEvent tp {self.tp} is inconsistent with entry={self.entry}, stop={self.stop}, "
                f"side={self.side}, target_r_multiple={self.target_r_multiple}"
            )
        return self


class CandidateLabel(BaseModel):
    candidate_id: str
    ts: int
    module: str
    symbol: str
    side: str
    signal_entry: float
    signal_tp: float
    entry_ts: int
    executed_entry: float
    stop: float
    executed_tp: float
    exit_ts: int | None = None
    executed_exit: float | None = None
    target_r_multiple: float
    timeout_bars: int
    tp_before_sl_within_horizon: bool
    tp1_before_sl_within_12h: bool
    mfe_r_24h: float
    mae_r_24h: float
    net_r_24h_timeout: float
    minutes_to_tp_or_sl: int | None = None
    outcome: Literal["tp", "sl", "timeout"]
    entry_source: Literal["raw_quote", "micro_quote", "bar_open"] = "bar_open"
    path_source: Literal["raw_events", "micro_buckets", "trade_bars"] = "trade_bars"
