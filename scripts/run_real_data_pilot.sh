#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
SYMBOL="${SYMBOL:-BTCUSDT}"
START="${START:-2025-09-01T00:00:00Z}"
END="${END:-2026-03-26T00:00:00Z}"
TRADE_GLOB="${TRADE_GLOB:-./downloads/bybit/trades/*.zip}"
ORDERBOOK_GLOB="${ORDERBOOK_GLOB:-./downloads/bybit/orderbook/*.zip}"
MACRO_CSV="${MACRO_CSV:-./macro_events.csv}"

btc-alert-engine backfill bybit-rest-history \
  --symbol "$SYMBOL" \
  --category linear \
  --start "$START" \
  --end "$END" \
  --oi-interval 1h \
  --account-ratio-period 1h \
  --data-dir "$DATA_DIR"

if compgen -G "$TRADE_GLOB" > /dev/null; then
  btc-alert-engine import bybit-history-trades \
    --input $TRADE_GLOB \
    --symbol "$SYMBOL" \
    --batch-size 500 \
    --data-dir "$DATA_DIR"
fi

if compgen -G "$ORDERBOOK_GLOB" > /dev/null; then
  btc-alert-engine import bybit-history-orderbook \
    --input $ORDERBOOK_GLOB \
    --symbol "$SYMBOL" \
    --depth 500 \
    --data-dir "$DATA_DIR"
fi

btc-alert-engine backfill deribit-dvol-history \
  --currency BTC \
  --resolution 60 \
  --start "$START" \
  --end "$END" \
  --data-dir "$DATA_DIR"

if [ -f "$MACRO_CSV" ]; then
  btc-alert-engine import-macro-csv --csv "$MACRO_CSV" --data-dir "$DATA_DIR"
fi

btc-alert-engine materialize bybit-bars --input "$DATA_DIR/raw" --data-dir "$DATA_DIR"
btc-alert-engine materialize micro-buckets --input "$DATA_DIR/raw" --data-dir "$DATA_DIR"
btc-alert-engine features bybit-foundation --data-dir "$DATA_DIR"
btc-alert-engine features options-deribit --data-dir "$DATA_DIR"
btc-alert-engine features macro-veto --data-dir "$DATA_DIR"

btc-alert-engine verify project \
  --data-dir "$DATA_DIR" \
  --registry ./research_registry_pilot.yaml \
  --contracts ./feature_contracts.yaml

btc-alert-engine research walkforward \
  --data-dir "$DATA_DIR" \
  --registry ./research_registry_pilot.yaml \
  --experiments A0_native_continuation A1_native_plus_soft_regime A3_native_plus_bybit_crowding_veto A4_native_plus_micro_score A5_native_plus_micro_gate A6_plus_deribit_options A8_plus_macro_veto R1_stress_reversal \
  --skip-missing
