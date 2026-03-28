# Real-data pilot runbook

This runbook is for the **first real-data BTC / Bybit pilot**.

It is intentionally narrower than the full registry:
- use `research_registry_pilot.yaml`
- start with the core experiments: `A0_native_continuation`, `A1_native_plus_soft_regime`, `A3_native_plus_bybit_crowding_veto`
- add `A4` / `A5` only after importing historical trades + order book from Bybit website downloads
- add `A6` / `A8` after Deribit DVOL and macro events are present

If you want a safer first pass with a clean output directory, baseline-only models, and no dependency on local Bybit archive downloads, use:

```bash
bash scripts/run_real_data_core_pilot.sh
```

If you want the same narrower pilot but with resumable checkpoints, use:

```bash
bash scripts/run_real_data_core_pilot_checkpoint.sh
```

If you want a longer real-data pass with safer defaults and a clean output directory, use:

```bash
bash scripts/run_real_data_long_pilot.sh
```

Default presets:
- `DATA_DIR=./data-pilot-long`
- `START=2024-01-01T00:00:00Z`
- `END=2026-03-26T00:00:00Z`

If the long pilot finishes but the standard pilot registry still produces empty walk-forward metrics, run the smoke walk-forward preset against the same derived data:

```bash
bash scripts/run_real_data_smoke_walkforward.sh
```

That preset uses [research_registry_smoke.yaml](/Users/nasrenkaraf/Documents/Projects/btc_bybit_alert_engine_starter_v2/research_registry_smoke.yaml), which:
- removes microstructure-dependent experiments
- uses lighter outer walk-forward windows
- uses a final frozen split aligned to the months where the current long pilot actually has labeled events

If you want a broader untouched test window for a less noisy success-rate read on the promoted setup, run:

```bash
bash scripts/run_real_data_broad_walkforward.sh
```

If you want the same smoke / broad presets but with **mirrored long + short** continuation and stress-reversal candidates, use:

```bash
bash scripts/run_real_data_smoke_walkforward_symmetric.sh
bash scripts/run_real_data_broad_walkforward_symmetric.sh
```

These symmetric registries use `research_registry_smoke_symmetric.yaml` and `research_registry_broad_test_symmetric.yaml` respectively, and now include both mirrored continuation and stress-reversal experiments.

That preset uses [research_registry_broad_test.yaml](/Users/nasrenkaraf/Documents/Projects/btc_bybit_alert_engine_starter_v2/research_registry_broad_test.yaml), which keeps the same smoke experiments but widens the final untouched test to July 1, 2025 through March 26, 2026, with calibration on April 1, 2025 through June 30, 2025. The broader walk-forward report is written to `./reports/data-pilot-long/walkforward_broad_test` by default.

To visualize any fired-alert CSV as an interactive HTML dashboard, run:

```bash
python3 scripts/render_alert_dashboard.py \
  ./reports/data-pilot-long/walkforward_broad_test/S0_native_continuation_final_alerts.csv
```

That writes an HTML file next to the CSV by default. If you are setting up a fresh environment, install the optional visualization dependency with `pip install -e ".[viz]"`.

Useful controls:

```bash
RESET_FROM=features_bybit_foundation bash scripts/run_real_data_core_pilot_checkpoint.sh
FORCE=1 bash scripts/run_real_data_core_pilot_checkpoint.sh
```

Valid `RESET_FROM` steps are:
- `backfill_bybit_rest_history`
- `import_bybit_history_trades`
- `import_bybit_history_orderbook`
- `backfill_deribit_dvol_history`
- `import_macro_csv`
- `materialize_bybit_bars`
- `materialize_micro_buckets`
- `features_bybit_foundation`
- `features_options_deribit`
- `features_macro_veto`
- `verify_project`
- `walkforward_core`

## 1. Backfill Bybit REST history

Use wider intervals for open interest and account ratio in the pilot so the backfill is tractable:

```bash
btc-alert-engine backfill bybit-rest-history \
  --symbol BTCUSDT \
  --category linear \
  --start 2025-09-01T00:00:00Z \
  --end 2026-03-26T00:00:00Z \
  --oi-interval 1h \
  --account-ratio-period 1h \
  --data-dir ./data
```

Optional dataset narrowing while iterating:

```bash
btc-alert-engine backfill bybit-rest-history \
  --symbol BTCUSDT \
  --category linear \
  --start 2025-09-01T00:00:00Z \
  --end 2026-03-26T00:00:00Z \
  --datasets kline_15 index_price_kline_15 premium_index_price_kline_15 funding_history open_interest account_ratio \
  --oi-interval 1h \
  --account-ratio-period 1h \
  --data-dir ./data
```

## 2. Import Bybit historical trades and order book

Download the Bybit historical **trades** and **orderbook** files from the historical market data website, then ingest them into the same raw-event store used by live collectors.

### Trades

```bash
btc-alert-engine import bybit-history-trades \
  --input ./downloads/bybit/trades/*.zip \
  --symbol BTCUSDT \
  --batch-size 500 \
  --data-dir ./data
```

### Order book

```bash
btc-alert-engine import bybit-history-orderbook \
  --input ./downloads/bybit/orderbook/*.zip \
  --symbol BTCUSDT \
  --depth 500 \
  --data-dir ./data
```

These importers convert website history into the same Bybit websocket raw schema used by the rest of the repo, so replay, micro-bucket materialization, labeling, and verification stay consistent.

## 3. Backfill Deribit DVOL

```bash
btc-alert-engine backfill deribit-dvol-history \
  --currency BTC \
  --resolution 60 \
  --start 2025-09-01T00:00:00Z \
  --end 2026-03-26T00:00:00Z \
  --data-dir ./data
```

## 4. Import macro events

Create a canonical CSV with columns:

```text
ts_utc_ms,event_type,source,importance,notes
```

Then ingest:

```bash
btc-alert-engine import-macro-csv --csv ./macro_events.csv --data-dir ./data
```

## 5. Materialize and build features

```bash
btc-alert-engine materialize bybit-bars --input ./data/raw --data-dir ./data
btc-alert-engine materialize micro-buckets --input ./data/raw --data-dir ./data
btc-alert-engine features bybit-foundation --data-dir ./data  # rerun after pulling the mirrored short-support update
btc-alert-engine features options-deribit --data-dir ./data
btc-alert-engine features macro-veto --data-dir ./data
```

## 6. Verify before running experiments

```bash
btc-alert-engine verify project \
  --data-dir ./data \
  --registry ./research_registry_pilot.yaml \
  --contracts ./feature_contracts.yaml
```

## 7. Run the core pilot ablations

```bash
btc-alert-engine research walkforward \
  --data-dir ./data \
  --registry ./research_registry_pilot.yaml \
  --experiments A0_native_continuation A1_native_plus_soft_regime A3_native_plus_bybit_crowding_veto A4_native_plus_micro_score A5_native_plus_micro_gate A6_plus_deribit_options A8_plus_macro_veto R1_stress_reversal \
  --skip-missing
```

## 8. Promotion logic for the pilot

For the pilot, answer these questions first:
- does soft regime beat the native continuation baseline?
- does crowding help more as a veto than as a passive feature?
- do imported microstructure signals help enough to justify maintaining execution-tape history?
- do Deribit DVOL and macro veto meaningfully improve tail behavior?

Only after the pilot is stable should you switch back to the full `research_registry.yaml`.
