from btc_alert_engine.normalize.bybit_rest import parse_account_ratio_event, parse_funding_history_event, parse_open_interest_event
from btc_alert_engine.schemas import RawEvent


def test_parse_open_interest_event_falls_back_to_event_symbol() -> None:
    event = RawEvent(
        source="bybit_rest",
        topic="rest.open_interest_1h.BTCUSDT",
        symbol="BTCUSDT",
        exchange_ts=1_000,
        local_received_ts=1_000,
        payload={"result": {"list": [{"openInterest": "68598.07300000", "timestamp": "1757397600000"}]}},
    )

    rows = parse_open_interest_event(event)

    assert rows == [{"ts": 1757397600000, "symbol": "BTCUSDT", "open_interest": 68598.073}]


def test_parse_funding_and_ratio_events_accept_missing_row_symbol() -> None:
    funding_event = RawEvent(
        source="bybit_rest",
        topic="rest.funding_history.BTCUSDT",
        symbol="BTCUSDT",
        exchange_ts=1_000,
        local_received_ts=1_000,
        payload={"result": {"list": [{"fundingRate": "0.00001288", "fundingRateTimestamp": "1759248000000"}]}},
    )
    ratio_event = RawEvent(
        source="bybit_rest",
        topic="rest.account_ratio_1h.BTCUSDT",
        symbol="BTCUSDT",
        exchange_ts=1_000,
        local_received_ts=1_000,
        payload={"result": {"list": [{"buyRatio": "0.6346", "sellRatio": "0.3654", "timestamp": "1758477600000"}]}},
    )

    funding_rows = parse_funding_history_event(funding_event)
    ratio_rows = parse_account_ratio_event(ratio_event)

    assert funding_rows == [{"ts": 1759248000000, "symbol": "BTCUSDT", "funding_rate": 0.00001288}]
    assert ratio_rows == [{"ts": 1758477600000, "symbol": "BTCUSDT", "buy_ratio": 0.6346, "sell_ratio": 0.3654}]
