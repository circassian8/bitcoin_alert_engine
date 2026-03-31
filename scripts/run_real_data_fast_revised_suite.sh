#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/run_real_data_smoke_walkforward_fast_revised_first_signal.sh
bash scripts/run_real_data_smoke_walkforward_fast_revised_highest_probability.sh
bash scripts/run_real_data_broad_walkforward_fast_revised_first_signal.sh
bash scripts/run_real_data_broad_walkforward_fast_revised_highest_probability.sh
