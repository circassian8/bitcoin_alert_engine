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
FORCE="${FORCE:-0}"
RESET_FROM="${RESET_FROM:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DATA_DIR/.checkpoints/core_pilot}"

STEP_ORDER=(
  backfill_bybit_rest_history
  import_bybit_history_trades
  import_bybit_history_orderbook
  backfill_deribit_dvol_history
  import_macro_csv
  materialize_bybit_bars
  materialize_micro_buckets
  features_bybit_foundation
  features_options_deribit
  features_macro_veto
  verify_project
  walkforward_core
)

run_engine() {
  "$PYTHON_BIN" -m btc_alert_engine.cli "$@"
}

validate_reset_from() {
  if [ -z "$RESET_FROM" ]; then
    return 0
  fi
  local item
  for item in "${STEP_ORDER[@]}"; do
    if [ "$item" = "$RESET_FROM" ]; then
      return 0
    fi
  done
  echo "Unknown RESET_FROM step: $RESET_FROM" >&2
  exit 1
}

dir_has_ndjson() {
  local path="$1"
  [ -d "$path" ] && find "$path" -type f -name '*.ndjson' -print -quit | grep -q .
}

mark_step_done() {
  local step="$1"
  mkdir -p "$CHECKPOINT_DIR"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$CHECKPOINT_DIR/$step.done"
}

step_output_ready() {
  local step="$1"
  case "$step" in
    backfill_bybit_rest_history)
      dir_has_ndjson "$DATA_DIR/raw/bybit_rest/rest.kline_15.$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/raw/bybit_rest/rest.index_price_kline_15.$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/raw/bybit_rest/rest.premium_index_price_kline_15.$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/raw/bybit_rest/rest.funding_history.$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/raw/bybit_rest/rest.open_interest_1h.$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/raw/bybit_rest/rest.account_ratio_1h.$SYMBOL"
      ;;
    import_bybit_history_trades)
      dir_has_ndjson "$DATA_DIR/raw/bybit_ws/publicTrade.$SYMBOL"
      ;;
    import_bybit_history_orderbook)
      find "$DATA_DIR/raw/bybit_ws" -type d -name "orderbook.*.$SYMBOL" -print -quit 2>/dev/null | grep -q .
      ;;
    backfill_deribit_dvol_history)
      dir_has_ndjson "$DATA_DIR/raw/deribit_http/deribit.volatility_index.BTC.60"
      ;;
    import_macro_csv)
      dir_has_ndjson "$DATA_DIR/raw/macro_csv"
      ;;
    materialize_bybit_bars)
      dir_has_ndjson "$DATA_DIR/derived/bars/bybit/trade_15m/$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/derived/bars/bybit/index_price_15m/$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/derived/bars/bybit/premium_index_15m/$SYMBOL"
      ;;
    materialize_micro_buckets)
      dir_has_ndjson "$DATA_DIR/derived/micro/bybit/1s/$SYMBOL"
      ;;
    features_bybit_foundation)
      dir_has_ndjson "$DATA_DIR/derived/features/trend_bybit/$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/derived/features/regime_bybit/$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/derived/features/crowding_bybit/$SYMBOL" &&
      dir_has_ndjson "$DATA_DIR/derived/features/micro_bybit/$SYMBOL"
      ;;
    features_options_deribit)
      dir_has_ndjson "$DATA_DIR/derived/features/options_deribit/$SYMBOL"
      ;;
    features_macro_veto)
      dir_has_ndjson "$DATA_DIR/derived/features/macro_veto/$SYMBOL"
      ;;
    verify_project)
      [ -f "$REPORTS_ROOT/verification/verification_report.md" ] &&
      [ -f "$REPORTS_ROOT/verification/verification_report.json" ]
      ;;
    walkforward_core)
      [ -f "$REPORTS_ROOT/walkforward/manifest.json" ] &&
      [ -f "$REPORTS_ROOT/walkforward/promotion_report.md" ]
      ;;
    *)
      return 1
      ;;
  esac
}

step_should_force() {
  local step="$1"
  if [ "$FORCE" = "1" ]; then
    return 0
  fi
  if [ -z "$RESET_FROM" ]; then
    return 1
  fi
  local active=0
  for item in "${STEP_ORDER[@]}"; do
    if [ "$item" = "$RESET_FROM" ]; then
      active=1
    fi
    if [ "$item" = "$step" ]; then
      if [ "$active" -eq 1 ]; then
        return 0
      fi
      return 1
    fi
  done
  echo "Unknown step ordering for RESET_FROM=$RESET_FROM or step=$step" >&2
  exit 1
}

run_step() {
  local step="$1"
  shift

  if step_should_force "$step"; then
    echo "Running $step (forced)"
    "$@"
    mark_step_done "$step"
    return 0
  fi

  if [ -f "$CHECKPOINT_DIR/$step.done" ]; then
    echo "Skipping $step (checkpoint)"
    return 0
  fi

  if step_output_ready "$step"; then
    echo "Skipping $step (adopted existing output)"
    mark_step_done "$step"
    return 0
  fi

  echo "Running $step"
  "$@"
  mark_step_done "$step"
}

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$REPORTS_ROOT"
validate_reset_from
echo "Using data dir: $DATA_DIR"
echo "Using reports dir: $REPORTS_ROOT"
if [ "$FORCE" = "1" ]; then
  echo "FORCE=1 set: all steps will rerun"
fi
if [ -n "$RESET_FROM" ]; then
  echo "RESET_FROM=$RESET_FROM set: steps from there onward will rerun"
fi

run_step backfill_bybit_rest_history \
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
  run_step import_bybit_history_trades \
    run_engine import bybit-history-trades \
      --input $TRADE_GLOB \
      --symbol "$SYMBOL" \
      --batch-size 500 \
      --data-dir "$DATA_DIR"
else
  echo "Skipping import_bybit_history_trades (no files matched $TRADE_GLOB)"
fi

if compgen -G "$ORDERBOOK_GLOB" > /dev/null; then
  run_step import_bybit_history_orderbook \
    run_engine import bybit-history-orderbook \
      --input $ORDERBOOK_GLOB \
      --symbol "$SYMBOL" \
      --depth 500 \
      --data-dir "$DATA_DIR"
else
  echo "Skipping import_bybit_history_orderbook (no files matched $ORDERBOOK_GLOB)"
fi

run_step backfill_deribit_dvol_history \
  run_engine backfill deribit-dvol-history \
    --currency BTC \
    --resolution 60 \
    --start "$START" \
    --end "$END" \
    --data-dir "$DATA_DIR"

if [ -f "$MACRO_CSV" ]; then
  run_step import_macro_csv \
    run_engine import-macro-csv --csv "$MACRO_CSV" --data-dir "$DATA_DIR"
else
  echo "Skipping import_macro_csv ($MACRO_CSV not found)"
fi

run_step materialize_bybit_bars \
  run_engine materialize bybit-bars --input "$DATA_DIR/raw" --data-dir "$DATA_DIR"

if [ -d "$DATA_DIR/raw/bybit_ws" ]; then
  run_step materialize_micro_buckets \
    run_engine materialize micro-buckets --input "$DATA_DIR/raw" --data-dir "$DATA_DIR"
else
  echo "Skipping materialize_micro_buckets (no Bybit websocket raw data present)"
fi

run_step features_bybit_foundation \
  run_engine features bybit-foundation --data-dir "$DATA_DIR"

run_step features_options_deribit \
  run_engine features options-deribit --data-dir "$DATA_DIR"

run_step features_macro_veto \
  run_engine features macro-veto --data-dir "$DATA_DIR"

run_step verify_project \
  run_engine verify project \
    --data-dir "$DATA_DIR" \
    --registry ./research_registry_pilot.yaml \
    --contracts ./feature_contracts.yaml \
    --output-dir "$REPORTS_ROOT/verification"

run_step walkforward_core \
  run_engine research walkforward \
    --data-dir "$DATA_DIR" \
    --registry ./research_registry_pilot.yaml \
    --output-dir "$REPORTS_ROOT/walkforward" \
    --models baseline \
    --experiments A0_native_continuation A1_native_plus_soft_regime A3_native_plus_bybit_crowding_veto A6_plus_deribit_options A8_plus_macro_veto \
    --skip-missing
