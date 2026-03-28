from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ExecutionConfig(BaseModel):
    venue: str
    symbol: str
    category: str
    mode: str
    market_truth: str


class TimeframesConfig(BaseModel):
    micro: str
    trigger: str
    setup: str
    regime: str


class ResearchRegistry(BaseModel):
    program: dict[str, Any]
    execution: ExecutionConfig
    goal: dict[str, Any]
    timeframes: TimeframesConfig
    data_sources: dict[str, Any]
    pit_policy: dict[str, Any]
    candidate_generators: dict[str, Any]
    labeling: dict[str, Any]
    models: dict[str, Any]
    experiments: list[dict[str, Any]]
    validation: dict[str, Any]
    execution_costs: dict[str, Any]
    promotion_rules: dict[str, Any] = Field(default_factory=dict)
    guardrails: list[str] = Field(default_factory=list)


class FeatureContracts(BaseModel):
    feature_blocks: dict[str, Any]
    schemas: dict[str, Any]


class RuntimeSettings(BaseModel):
    data_dir: Path = Field(default=Path("./data"))
    bybit_testnet: bool = False
    deribit_testnet: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        data_dir = Path(os.getenv("BTC_ALERT_ENGINE_DATA_DIR", "./data"))
        bybit_testnet = os.getenv("BTC_ALERT_ENGINE_BYBIT_TESTNET", "false").lower() == "true"
        deribit_testnet = os.getenv("BTC_ALERT_ENGINE_DERIBIT_TESTNET", "false").lower() == "true"
        log_level = os.getenv("BTC_ALERT_ENGINE_LOG_LEVEL", "INFO")
        return cls(
            data_dir=data_dir,
            bybit_testnet=bybit_testnet,
            deribit_testnet=deribit_testnet,
            log_level=log_level,
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_research_registry(path: str | Path) -> ResearchRegistry:
    return ResearchRegistry.model_validate(_read_yaml(Path(path)))


def load_feature_contracts(path: str | Path) -> FeatureContracts:
    return FeatureContracts.model_validate(_read_yaml(Path(path)))


def default_reports_root(data_dir: str | Path) -> Path:
    data_path = Path(data_dir)
    name = data_path.name or data_path.resolve().name or "data"
    return Path("reports") / name
