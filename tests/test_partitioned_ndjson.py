from __future__ import annotations

from pathlib import Path

from btc_alert_engine.storage.partitioned_ndjson import PartitionedNdjsonWriter


def test_writer_evicts_old_handles_when_many_partitions(tmp_path: Path) -> None:
    writer = PartitionedNdjsonWriter(tmp_path, max_open_handles=4)
    try:
        for hour in range(12):
            ts_ms = 1_700_000_000_000 + hour * 60 * 60 * 1000
            writer.write(
                namespace="bars/bybit/premium_index_15m",
                symbol="BTCUSDT",
                ts_ms=ts_ms,
                record={"ts": ts_ms, "close": hour},
            )
        assert len(writer._handles) <= 4
        files = sorted(tmp_path.rglob("*.ndjson"))
        assert len(files) == 12
        for idx, file_path in enumerate(files):
            content = file_path.read_text().strip()
            assert content
    finally:
        writer.close()
