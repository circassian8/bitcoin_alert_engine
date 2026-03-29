from pathlib import Path

from btc_alert_engine.config import default_reports_root, load_feature_contracts, load_research_registry


def test_load_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_research_registry(root / "research_registry.yaml")
    assert registry.execution.venue == "bybit"
    assert registry.execution.symbol == "BTCUSDT"
    assert registry.timeframes.micro == "1s"
    assert len(registry.experiments) >= 1


def test_load_feature_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    contracts = load_feature_contracts(root / "feature_contracts.yaml")
    assert "trend_bybit" in contracts.feature_blocks
    assert "raw_event" in contracts.schemas


def test_default_reports_root_uses_top_level_reports_dir() -> None:
    assert default_reports_root("./data-pilot-long") == Path("reports") / "data-pilot-long"


def test_symmetric_registries_include_reversal_experiment() -> None:
    root = Path(__file__).resolve().parents[1]
    for registry_name in ("research_registry_smoke_symmetric.yaml", "research_registry_broad_test_symmetric.yaml"):
        registry = load_research_registry(root / registry_name)
        reversal = [exp for exp in registry.experiments if exp["generator"] == "stress_reversal_v0"]
        assert reversal, registry_name
        assert reversal[0]["sides"] == ["long", "short"], registry_name
