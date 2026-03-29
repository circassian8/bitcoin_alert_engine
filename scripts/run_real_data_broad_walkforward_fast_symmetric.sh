#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-./data-pilot-long}"
DATA_DIR_BASENAME="$(basename "${DATA_DIR%/}")"
REPORTS_ROOT="${REPORTS_ROOT:-./reports/$DATA_DIR_BASENAME}"
REGISTRY="${REGISTRY:-./research_registry_broad_test_fast_symmetric.yaml}"
CONTRACTS="${CONTRACTS:-./feature_contracts.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPORTS_ROOT/walkforward_broad_test_fast_symmetric}"

"$PYTHON_BIN" -m btc_alert_engine.cli verify project \
  --data-dir "$DATA_DIR" \
  --registry "$REGISTRY" \
  --contracts "$CONTRACTS" \
  --output-dir "$REPORTS_ROOT/verification"

"$PYTHON_BIN" -m btc_alert_engine.cli research walkforward \
  --data-dir "$DATA_DIR" \
  --registry "$REGISTRY" \
  --output-dir "$OUTPUT_DIR" \
  --models baseline \
  --skip-missing
