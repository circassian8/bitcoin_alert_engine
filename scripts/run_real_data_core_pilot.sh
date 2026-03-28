#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-./data-pilot-core}"
DATA_DIR_BASENAME="$(basename "${DATA_DIR%/}")"
REPORTS_ROOT="${REPORTS_ROOT:-./reports/$DATA_DIR_BASENAME}"
SYMBOL="${SYMBOL:-BTCUSDT}"
START="${START:-2025-09-01T00:00:00Z}"
END="${END:-2026-03-26T00:00:00Z}"
TRADE_GLOB="${TRADE_GLOB:-./downloads/bybit/trades/*.zip}"
ORDERBOOK_GLOB="${ORDERBOOK_GLOB:-./downloads/bybit/orderbook/*.zip}"
MACRO_CSV="${MACRO_CSV:-./samples/macro_events.csv}"

run_engine() {
  "$PYTHON_BIN" -m btc_alert_engine.cli "$@"
}

echo "Using data dir: $DATA_DIR"
echo "Using reports dir: $REPORTS_ROOT"
mkdir -p "$DATA_DIR" "$REPORTS_ROOT"

run_engine backfill bybit-rest-history \
  --symbol "$SYMBOL" \
  --category linear \
  --start "$START" \
  --end "$END" \
  --datasets kline_15 index_price_kline_15 premium_index_price_kline_15 funding_history open_interest account_ratio \
  --oi-interval 1h \
  --account-ratio-period 1h \
  --data-dir "$DATA_DIR"

if compgen -G "$TRADE_GLOB" > /dev/null; then
  run_engine import bybit-history-trades \
    --input $TRADE_GLOB \
    --symbol "$SYMBOL" \
    --batch-size 500 \
    --data-dir "$DATA_DIR"
else
  echo "Skipping Bybit trade import: no files matched $TRADE_GLOB"
fi

if compgen -G "$ORDERBOOK_GLOB" > /dev/null; then
  run_engine import bybit-history-orderbook \
    --input $ORDERBOOK_GLOB \
    --symbol "$SYMBOL" \
    --depth 500 \
    --data-dir "$DATA_DIR"
else
  echo "Skipping Bybit order book import: no files matched $ORDERBOOK_GLOB"
fi

run_engine backfill deribit-dvol-history \
  --currency BTC \
  --resolution 60 \
  --start "$START" \
  --end "$END" \
  --data-dir "$DATA_DIR"

if [ -f "$MACRO_CSV" ]; then
  run_engine import-macro-csv --csv "$MACRO_CSV" --data-dir "$DATA_DIR"
else
  echo "Skipping macro import: $MACRO_CSV not found"
fi

run_engine materialize bybit-bars --input "$DATA_DIR/raw" --data-dir "$DATA_DIR"

if [ -d "$DATA_DIR/raw/bybit_ws" ]; then
  run_engine materialize micro-buckets --input "$DATA_DIR/raw" --data-dir "$DATA_DIR"
else
  echo "Skipping micro-bucket materialization: no Bybit websocket raw data present"
fi

run_engine features bybit-foundation --data-dir "$DATA_DIR"
run_engine features options-deribit --data-dir "$DATA_DIR"
run_engine features macro-veto --data-dir "$DATA_DIR"

run_engine verify project \
  --data-dir "$DATA_DIR" \
  --registry ./research_registry_pilot.yaml \
  --contracts ./feature_contracts.yaml \
  --output-dir "$REPORTS_ROOT/verification"

run_engine research walkforward \
  --data-dir "$DATA_DIR" \
  --registry ./research_registry_pilot.yaml \
  --output-dir "$REPORTS_ROOT/walkforward" \
  --models baseline \
  --experiments A0_native_continuation A1_native_plus_soft_regime A3_native_plus_bybit_crowding_veto A6_plus_deribit_options A8_plus_macro_veto \
  --skip-missing
