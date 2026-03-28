#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATA_DIR="${DATA_DIR:-./data-pilot-long}"
export START="${START:-2024-01-01T00:00:00Z}"
export END="${END:-2026-03-26T00:00:00Z}"

exec bash "$ROOT_DIR/scripts/run_real_data_core_pilot_checkpoint.sh"
