from pathlib import Path

from btc_alert_engine.cli import _load_foundation_inputs
from btc_alert_engine.schemas import PriceBar
from btc_alert_engine.storage.partitioned_ndjson import PartitionedNdjsonWriter


def test_load_foundation_inputs_tolerates_missing_optional_blocks(tmp_path: Path) -> None:
    derived_dir = tmp_path / "derived"
    bar = PriceBar(
        ts=1_700_000_000_000,
        symbol="BTCUSDT",
        interval="15",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
        turnover=1000.0,
    )
    with PartitionedNdjsonWriter(derived_dir) as writer:
        writer.write(namespace="bars/bybit/trade_15m", symbol="BTCUSDT", ts_ms=bar.ts, record=bar)

    trade_bars, index_bars, premium_bars, micro_buckets = _load_foundation_inputs(derived_dir, "BTCUSDT")

    assert len(trade_bars) == 1
    assert index_bars == []
    assert premium_bars == []
    assert micro_buckets == []
