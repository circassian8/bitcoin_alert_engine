from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from btc_alert_engine.importers import import_bybit_history_orderbook, import_bybit_history_trades
from btc_alert_engine.materialize.bybit_foundation import materialize_micro_buckets
from btc_alert_engine.storage.raw_ndjson import RawEventWriter, iter_raw_events


def test_import_bybit_history_trades_and_orderbook_feed_existing_materializers(tmp_path: Path) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    trades_zip = downloads / "2026-03-01_BTCUSDT_trade.csv.zip"
    with zipfile.ZipFile(trades_zip, "w") as archive:
        rows = [
            {"timestamp": "1200", "symbol": "BTCUSDT", "price": "101.0", "size": "0.2", "side": "Buy", "execId": "t1", "seq": "1"},
            {"timestamp": "2200", "symbol": "BTCUSDT", "price": "100.5", "size": "0.3", "side": "Sell", "execId": "t2", "seq": "2"},
        ]
        with archive.open("2026-03-01_BTCUSDT_trade.csv", "w") as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", newline="")
            writer = csv.DictWriter(text_handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            text_handle.flush()

    orderbook_zip = downloads / "2026-03-01_BTCUSDT_ob500.data.zip"
    orderbook_rows = [
        {
            "topic": "orderbook.500.BTCUSDT",
            "type": "snapshot",
            "ts": 1000,
            "cts": 1000,
            "data": {"s": "BTCUSDT", "u": 1, "seq": 1, "b": [["100.0", "1.5"]], "a": [["101.0", "1.0"]]},
        },
        {
            "topic": "orderbook.500.BTCUSDT",
            "type": "delta",
            "ts": 2000,
            "cts": 2000,
            "data": {"s": "BTCUSDT", "u": 2, "seq": 2, "b": [["100.0", "2.0"]], "a": [["101.0", "0.8"]]},
        },
    ]
    with zipfile.ZipFile(orderbook_zip, "w") as archive:
        with archive.open("2026-03-01_BTCUSDT_ob500.data", "w") as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8")
            for row in orderbook_rows:
                text_handle.write(json.dumps(row) + "\n")
            text_handle.flush()

    raw_dir = tmp_path / "raw"
    with RawEventWriter(raw_dir) as writer:
        trade_counts = import_bybit_history_trades([trades_zip], writer=writer, symbol="BTCUSDT", batch_size=1)
        book_counts = import_bybit_history_orderbook([orderbook_zip], writer=writer, symbol="BTCUSDT", depth=500)

    assert trade_counts == {"trade_rows": 2, "trade_events": 2}
    assert book_counts == {"orderbook_rows": 2, "orderbook_events": 2}

    raw_events = list(iter_raw_events([raw_dir]))
    assert len(raw_events) == 4
    assert {event.source for event in raw_events} == {"bybit_ws"}
    assert {event.topic for event in raw_events} == {"publicTrade.BTCUSDT", "orderbook.500.BTCUSDT"}
    assert all(event.metadata.get("imported") is True for event in raw_events)

    buckets = materialize_micro_buckets([raw_dir], symbol="BTCUSDT")
    assert len(buckets) == 2
    assert buckets[0].best_bid_price == 100.0
    assert buckets[0].best_ask_price == 101.0
    assert buckets[0].buy_volume == 0.2
    assert buckets[1].sell_volume == 0.3
    assert buckets[1].quote_updates == 1
