import json
from pathlib import Path

from btc_alert_engine.normalize.bybit_public import (
    parse_liquidation_payload,
    parse_orderbook_payload,
    parse_trade_payload,
)


def load_json(name: str) -> dict:
    path = Path(__file__).parent / "data" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_parse_trade_payload() -> None:
    payload = load_json("bybit_trade.json")
    trades = parse_trade_payload(payload)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.symbol == "BTCUSDT"
    assert str(trade.price) == "16578.50"
    assert trade.side == "Buy"
    assert trade.seq == 1783284617


def test_parse_liquidation_payload() -> None:
    payload = load_json("bybit_liquidation.json")
    liqs = parse_liquidation_payload(payload)
    assert len(liqs) == 1
    liq = liqs[0]
    assert liq.symbol == "ROSEUSDT"
    assert liq.side == "Sell"
    assert str(liq.bankruptcy_price) == "0.04499"


def test_parse_orderbook_payload() -> None:
    payload = load_json("bybit_orderbook_snapshot.json")
    message = parse_orderbook_payload(payload)
    assert message.symbol == "BTCUSDT"
    assert message.message_type == "snapshot"
    assert message.update_id == 18521288
    assert len(message.bids) == 2
