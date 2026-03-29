from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from btc_alert_engine.features.bybit_foundation import load_micro_buckets, load_price_bars
from btc_alert_engine.research.labeling import candidate_id, label_candidates
from btc_alert_engine.schemas import (
    CandidateEvent,
    CoinGlassAggregateFeatureSnapshot,
    CrowdingFeatureSnapshot,
    CryptoQuantOnChainFeatureSnapshot,
    DeribitOptionsFeatureSnapshot,
    GlassnodeOnChainFeatureSnapshot,
    GlassnodeOptionsFeatureSnapshot,
    MacroVetoFeatureSnapshot,
    MicroFeatureSnapshot,
    PriceBar,
    RegimeFeatureSnapshot,
    TrendFeatureSnapshot,
)
from btc_alert_engine.storage.partitioned_ndjson import iter_json_records
from btc_alert_engine.strategy.bybit_candidates import (
    build_continuation_candidates,
    build_stress_reversal_candidates,
    load_feature_models,
)

BLOCK_NAMESPACE_ALIASES: dict[str, str] = {
    "trend_bybit": "features/trend_bybit",
    "regime_bybit": "features/regime_bybit",
    "crowding_bybit_feature": "features/crowding_bybit",
    "crowding_bybit_veto": "features/crowding_bybit",
    "micro_bybit_score": "features/micro_bybit",
    "micro_bybit_gate": "features/micro_bybit",
    "options_deribit": "features/options_deribit",
    "options_glassnode": "features/options_glassnode",
    "macro_veto": "features/macro_veto",
    "onchain_glassnode": "features/onchain_glassnode",
    "onchain_cryptoquant": "features/onchain_cryptoquant",
    "aggregate_derivs_coinglass": "features/aggregate_derivs_coinglass",
}

FEATURE_ONLY_BLOCKS = {
    "trend_bybit",
    "regime_bybit",
    "crowding_bybit_feature",
    "micro_bybit_score",
    "options_deribit",
    "options_glassnode",
    "onchain_glassnode",
    "onchain_cryptoquant",
    "aggregate_derivs_coinglass",
}

KNOWN_OPTIONAL_BLOCKS = {
    "options_deribit",
    "options_glassnode",
    "macro_veto",
    "onchain_glassnode",
    "onchain_cryptoquant",
    "aggregate_derivs_coinglass",
}


@dataclass(slots=True)
class ExperimentDataset:
    experiment_id: str
    generator: str
    blocks: list[str]
    frame: pd.DataFrame
    candidates: list[CandidateEvent]
    labels: pd.DataFrame
    skip_reason: str | None = None


@dataclass(slots=True)
class ExperimentCandidateConfig:
    require_regime_gate: bool = False
    require_crowding_veto: bool = False
    require_micro_gate: bool = False
    require_macro_veto: bool = False
    stop_width_multiplier: float = 1.0
    target_r_multiple: float = 2.0
    sides: tuple[str, ...] = ("long",)


@dataclass(slots=True)
class ExperimentAvailability:
    available_blocks: set[str]
    missing_blocks: set[str]


FEATURE_SCHEMA_MAP: dict[str, type] = {
    "trend_bybit": TrendFeatureSnapshot,
    "regime_bybit": RegimeFeatureSnapshot,
    "crowding_bybit_feature": CrowdingFeatureSnapshot,
    "crowding_bybit_veto": CrowdingFeatureSnapshot,
    "micro_bybit_score": MicroFeatureSnapshot,
    "micro_bybit_gate": MicroFeatureSnapshot,
    "options_deribit": DeribitOptionsFeatureSnapshot,
    "options_glassnode": GlassnodeOptionsFeatureSnapshot,
    "macro_veto": MacroVetoFeatureSnapshot,
    "onchain_glassnode": GlassnodeOnChainFeatureSnapshot,
    "onchain_cryptoquant": CryptoQuantOnChainFeatureSnapshot,
    "aggregate_derivs_coinglass": CoinGlassAggregateFeatureSnapshot,
}


INTRINSIC_CANDIDATE_FEATURES_BY_MODULE: dict[str, set[str]] = {
    "continuation_v1": {"atr15", "signal_risk", "signal_risk_pct", "side_sign"},
    "stress_reversal_v0": {"atr15", "signal_risk", "signal_risk_pct", "side_sign"},
}


def block_path(derived_dir: Path, symbol: str, block_name: str) -> Path:
    namespace = BLOCK_NAMESPACE_ALIASES.get(block_name, f"features/{block_name}")
    return derived_dir / Path(namespace) / symbol


def load_block_frame(derived_dir: Path, symbol: str, block_name: str) -> pd.DataFrame | None:
    path = block_path(derived_dir, symbol, block_name)
    if not path.exists():
        return None
    rows = list(iter_json_records([path]))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "ts" not in df.columns:
        return None
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    df = df.drop(columns=["symbol"], errors="ignore")
    prefix = f"{block_name}__"
    feature_cols = [c for c in df.columns if c != "ts"]
    renamed = {col: f"{prefix}{col}" for col in feature_cols}
    df = df.rename(columns=renamed)
    return df


def load_trade_bars(derived_dir: Path, symbol: str) -> list[PriceBar]:
    path = derived_dir / "bars" / "bybit" / "trade_15m" / symbol
    if not path.exists():
        return []
    return load_price_bars([path])


def _to_label_frame(labels: list) -> pd.DataFrame:
    if not labels:
        return pd.DataFrame()
    df = pd.DataFrame([label.model_dump(mode="json") for label in labels]).sort_values("ts")
    df["y"] = df["tp_before_sl_within_horizon"].astype(int)
    df["realized_r"] = -1.0
    tp_mask = df["outcome"] == "tp"
    timeout_mask = df["outcome"] == "timeout"
    df.loc[tp_mask, "realized_r"] = df.loc[tp_mask, "target_r_multiple"].astype(float)
    df.loc[timeout_mask, "realized_r"] = df.loc[timeout_mask, "net_r_24h_timeout"].astype(float)
    df["holding_minutes"] = df["minutes_to_tp_or_sl"].fillna(df["timeout_bars"].astype(float) * 15.0)
    return df


def _to_candidate_frame(candidates: list[CandidateEvent]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = candidate.model_dump(mode="json")
        features = row.pop("features", {}) or {}
        intrinsic_allowlist = INTRINSIC_CANDIDATE_FEATURES_BY_MODULE.get(candidate.module, set())
        for key, value in features.items():
            if key not in intrinsic_allowlist:
                continue
            row[f"candidate__{key}"] = value
        row["candidate_id"] = candidate_id(candidate)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("ts")
    return df


def _merge_feature_blocks(base: pd.DataFrame, derived_dir: Path, symbol: str, blocks: list[str]) -> tuple[pd.DataFrame, ExperimentAvailability]:
    frame = base.sort_values("ts").copy()
    available: set[str] = set()
    missing: set[str] = set()
    for block in blocks:
        block_df = load_block_frame(derived_dir, symbol, block)
        if block_df is None:
            missing.add(block)
            continue
        available.add(block)
        block_df = block_df.sort_values("ts")
        frame = pd.merge_asof(frame, block_df, on="ts", direction="backward")
    return frame, ExperimentAvailability(available_blocks=available, missing_blocks=missing)


def _load_typed_features(derived_dir: Path, symbol: str, block_name: str) -> list[Any]:
    model_cls = FEATURE_SCHEMA_MAP.get(block_name)
    if model_cls is None:
        return []
    path = block_path(derived_dir, symbol, block_name)
    if not path.exists():
        return []
    return load_feature_models([path], model_cls)


def derive_candidate_config(experiment: dict[str, Any]) -> ExperimentCandidateConfig:
    blocks = list(experiment.get("blocks", []))
    generator_params = dict(experiment.get("generator_params", {}) or {})
    stop_width_multiplier = generator_params.get("stop_width_multiplier", 1.0)
    try:
        stop_width_multiplier = float(stop_width_multiplier)
    except (TypeError, ValueError):
        stop_width_multiplier = 1.0
    if not math.isfinite(stop_width_multiplier) or stop_width_multiplier <= 0:
        stop_width_multiplier = 1.0
    target_r_multiple = generator_params.get("target_r_multiple", 2.0)
    try:
        target_r_multiple = float(target_r_multiple)
    except (TypeError, ValueError):
        target_r_multiple = 2.0
    if not math.isfinite(target_r_multiple) or target_r_multiple <= 0:
        target_r_multiple = 2.0
    requested_sides = experiment.get("sides") or ["long"]
    sides: list[str] = []
    for side in requested_sides:
        value = str(side).lower()
        if value not in {"long", "short"}:
            continue
        if value not in sides:
            sides.append(value)
    if not sides:
        sides = ["long"]
    return ExperimentCandidateConfig(
        require_regime_gate="regime_bybit" in blocks,
        require_crowding_veto="crowding_bybit_veto" in blocks,
        require_micro_gate="micro_bybit_gate" in blocks,
        require_macro_veto="macro_veto" in blocks,
        stop_width_multiplier=stop_width_multiplier,
        target_r_multiple=target_r_multiple,
        sides=tuple(sides),
    )


def _load_generator_inputs(
    derived_dir: Path,
    symbol: str,
    *,
    generator: str,
    config: ExperimentCandidateConfig,
) -> dict[str, list[Any]]:
    inputs: dict[str, list[Any]] = {
        "trend": [],
        "regime": [],
        "crowding": [],
        "micro": [],
        "macro": [],
    }
    if generator == "continuation_v1":
        inputs["trend"] = _load_typed_features(derived_dir, symbol, "trend_bybit")
        if config.require_regime_gate:
            inputs["regime"] = _load_typed_features(derived_dir, symbol, "regime_bybit")
        if config.require_crowding_veto:
            inputs["crowding"] = _load_typed_features(derived_dir, symbol, "crowding_bybit_veto")
        if config.require_micro_gate:
            inputs["micro"] = _load_typed_features(derived_dir, symbol, "micro_bybit_score")
        if config.require_macro_veto:
            inputs["macro"] = _load_typed_features(derived_dir, symbol, "macro_veto")
        return inputs
    if generator == "stress_reversal_v0":
        inputs["regime"] = _load_typed_features(derived_dir, symbol, "regime_bybit")
        if config.require_crowding_veto:
            inputs["crowding"] = _load_typed_features(derived_dir, symbol, "crowding_bybit_veto")
        if config.require_micro_gate:
            inputs["micro"] = _load_typed_features(derived_dir, symbol, "micro_bybit_score")
        if config.require_macro_veto:
            inputs["macro"] = _load_typed_features(derived_dir, symbol, "macro_veto")
        return inputs
    return inputs


def build_experiment_event_frame(
    derived_dir: Path,
    symbol: str,
    experiment: dict[str, Any],
    *,
    raw_paths: Iterable[str | Path] | None = None,
    raw_tape: pd.DataFrame | None = None,
    slippage_bps: float = 1.0,
    latency_ms: int = 0,
    skip_missing: bool = True,
) -> ExperimentDataset:
    experiment_id = str(experiment["id"])
    generator = str(experiment["generator"])
    blocks = list(experiment.get("blocks", []))

    trade_bars = load_trade_bars(derived_dir, symbol)
    micro_path = derived_dir / "micro" / "bybit" / "1s" / symbol
    micro_buckets = load_micro_buckets([micro_path]) if micro_path.exists() else []
    if not trade_bars:
        return ExperimentDataset(experiment_id, generator, blocks, pd.DataFrame(), [], pd.DataFrame(), skip_reason="missing_trade_bars")

    config = derive_candidate_config(experiment)
    generator_inputs = _load_generator_inputs(derived_dir, symbol, generator=generator, config=config)

    if generator == "continuation_v1":
        candidates = build_continuation_candidates(
            trade_bars,
            generator_inputs["trend"],
            generator_inputs["regime"],
            generator_inputs["crowding"],
            generator_inputs["micro"],
            symbol=symbol,
            require_regime_gate=config.require_regime_gate,
            require_crowding_veto=config.require_crowding_veto,
            require_micro_gate=config.require_micro_gate,
            require_macro_veto=config.require_macro_veto,
            macro_features=generator_inputs["macro"],
            stop_width_multiplier=config.stop_width_multiplier,
            target_r_multiple=config.target_r_multiple,
            sides=config.sides,
        )
    elif generator == "stress_reversal_v0":
        candidates = build_stress_reversal_candidates(
            trade_bars,
            generator_inputs["regime"],
            generator_inputs["crowding"],
            generator_inputs["micro"],
            symbol=symbol,
            require_crowding_veto=config.require_crowding_veto,
            require_micro_gate=config.require_micro_gate,
            require_macro_veto=config.require_macro_veto,
            macro_features=generator_inputs["macro"],
            sides=config.sides,
        )
    else:
        return ExperimentDataset(experiment_id, generator, blocks, pd.DataFrame(), [], pd.DataFrame(), skip_reason=f"unknown_generator:{generator}")

    if not candidates:
        return ExperimentDataset(experiment_id, generator, blocks, pd.DataFrame(), [], pd.DataFrame(), skip_reason="no_candidates")

    labels = label_candidates(
        candidates,
        trade_bars,
        micro_buckets=micro_buckets,
        raw_paths=raw_paths,
        raw_tape=raw_tape,
        slippage_bps=slippage_bps,
        latency_ms=latency_ms,
    )
    label_frame = _to_label_frame(labels)
    if label_frame.empty:
        return ExperimentDataset(experiment_id, generator, blocks, pd.DataFrame(), candidates, label_frame, skip_reason="no_labels")

    candidate_frame = _to_candidate_frame(candidates)
    base = candidate_frame.merge(label_frame, on=["candidate_id", "ts", "module", "symbol", "side"], how="inner")
    if base.empty:
        return ExperimentDataset(experiment_id, generator, blocks, pd.DataFrame(), candidates, label_frame, skip_reason="candidate_label_join_empty")

    feature_blocks = [block for block in blocks if block in FEATURE_ONLY_BLOCKS or block in {"crowding_bybit_veto", "micro_bybit_gate", "macro_veto"}]
    merged, availability = _merge_feature_blocks(base, derived_dir, symbol, feature_blocks)
    if skip_missing and availability.missing_blocks:
        return ExperimentDataset(
            experiment_id,
            generator,
            blocks,
            pd.DataFrame(),
            candidates,
            label_frame,
            skip_reason=f"missing_blocks:{','.join(sorted(availability.missing_blocks))}",
        )

    merged["ts_dt"] = pd.to_datetime(merged["ts"], unit="ms", utc=True)
    merged = merged.sort_values("ts").reset_index(drop=True)
    return ExperimentDataset(experiment_id, generator, blocks, merged, candidates, label_frame)


def feature_columns_for_experiment(frame: pd.DataFrame, blocks: list[str]) -> list[str]:
    columns: list[str] = []
    for block in blocks:
        if block == "crowding_bybit_veto":
            continue
        if block == "micro_bybit_gate":
            continue
        if block == "macro_veto":
            continue
        prefix = f"{block}__"
        block_cols = [col for col in frame.columns if col.startswith(prefix)]
        if block == "micro_bybit_score":
            block_cols = [
                col
                for col in block_cols
                if not col.endswith("__gate_pass")
                and not col.endswith("__gate_pass_long")
                and not col.endswith("__gate_pass_short")
            ]
        if block == "crowding_bybit_feature":
            block_cols = [col for col in block_cols if not col.endswith("__veto_long") and not col.endswith("__veto_short")]
        columns.extend(block_cols)
    intrinsic_candidate_cols = {
        f"candidate__{name}"
        for names in INTRINSIC_CANDIDATE_FEATURES_BY_MODULE.values()
        for name in names
    }
    candidate_cols = [col for col in frame.columns if col in intrinsic_candidate_cols]
    columns.extend(candidate_cols)
    seen: set[str] = set()
    unique: list[str] = []
    for col in columns:
        if col not in seen and col in frame.columns:
            seen.add(col)
            unique.append(col)
    return unique
