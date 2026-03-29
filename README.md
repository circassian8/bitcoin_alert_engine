# BTC / Bybit alert engine starter

This repo now covers the full **research-foundation path through walk-forward experiments and project verification**:

- raw public market data collectors for Bybit
- raw REST pollers for Bybit market state
- a Deribit DVOL poller for options-risk context
- durable raw event storage
- deterministic normalization, replay, and replay hashing
- a tested order book builder with gap/out-of-order detection
- Bybit bar and 1-second micro-bucket materializers
- Bybit-native trend / regime / crowding / micro feature jobs
- external context feature jobs for Deribit, macro calendars, Glassnode, CryptoQuant, and CoinGlass overlays
- continuation and stress-reversal candidate generators with mirrored long/short support
- target-aware, quote-aware, first-touch barrier label generation for event datasets
- walk-forward experiment runner with promotion reports
- project verification checks for contracts, derived blocks, and experiment readiness
- frozen research contracts loaded from YAML

## What is included

- `research_registry.yaml`: experiment registry and PIT policy
- `feature_contracts.yaml`: feature block contracts and schemas
- `implementation_plan.md`: implementation order and acceptance checks
- `src/btc_alert_engine/collectors`: Bybit / Deribit collection code
- `src/btc_alert_engine/normalize`: parsers, order book builder, replay tools
- `src/btc_alert_engine/storage`: NDJSON raw event storage
- `src/btc_alert_engine/features`: Bybit-native and external-context feature jobs
- `src/btc_alert_engine/research`: labeling, experiment assembly, walk-forward runner
- `src/btc_alert_engine/verification`: project verification checks and report writer
- `tests/`: deterministic unit and integration tests

## What is intentionally not included yet

- private authenticated execution on Bybit
- Telegram / Slack notifiers
- live probability serving and online recalibration
- portfolio-level orchestration beyond the current research reports

Those belong in later phases after the data plane, feature jobs, and experiment registry are stable.

## Quickstart

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

For the full research stack, including the LightGBM challenger, install:

```bash
pip install -e ".[dev,research]"
```

## Core collectors

```bash
btc-alert-engine collect bybit-orderbook --symbol BTCUSDT --data-dir ./data
btc-alert-engine collect bybit-trades --symbol BTCUSDT --data-dir ./data
btc-alert-engine collect bybit-liquidations --symbol BTCUSDT --data-dir ./data
btc-alert-engine collect bybit-rest --symbol BTCUSDT --data-dir ./data
btc-alert-engine collect deribit-dvol --currency BTC --data-dir ./data
```

Replay a raw file directory, inspect deterministic top-of-book states, and print a replay hash:

```bash
btc-alert-engine replay top-of-book --input ./data/raw/bybit_ws
```

## Historical backfill

Use backfill commands to seed the raw event store before or alongside live collection:

```bash
btc-alert-engine backfill bybit-rest-history --symbol BTCUSDT --start 2024-01-01T00:00:00Z --end 2024-01-07T00:00:00Z --data-dir ./data
btc-alert-engine backfill bybit-rest-history --symbol BTCUSDT --start 2025-09-01T00:00:00Z --end 2026-03-26T00:00:00Z --oi-interval 1h --account-ratio-period 1h --data-dir ./data
btc-alert-engine backfill deribit-dvol-history --currency BTC --start 2024-01-01T00:00:00Z --end 2024-01-07T00:00:00Z --data-dir ./data
```

For Bybit REST history, `--datasets` accepts either the exact spec names (for example `open_interest_1h`) or the generic aliases `open_interest` / `account_ratio`.

These commands write raw NDJSON events in the same storage layout used by the collectors, so replay, feature jobs, and research code can consume live and backfilled data through the same path.

## Importing Bybit website history

Bybit also exposes historical market data downloads on its website. This repo now includes importers that convert those downloaded trade and order-book files into the same raw websocket schema used by live collection.

```bash
btc-alert-engine import bybit-history-trades --input ./downloads/bybit/trades/*.zip --symbol BTCUSDT --data-dir ./data
btc-alert-engine import bybit-history-orderbook --input ./downloads/bybit/orderbook/*.zip --symbol BTCUSDT --depth 500 --data-dir ./data
```

Use these importers to unlock the microstructure experiments (`A4`, `A5`) and the full quote-aware intrabar labeling path on historical data.

## Macro events

Official macro calendars come in different formats and change over time. This starter ships a simple CSV ingestor instead of brittle HTML scrapers.

Use `samples/macro_events.csv` style rows and ingest them into the raw-event store:

```bash
btc-alert-engine import-macro-csv --csv ./macro_events.csv --data-dir ./data
```

## Research pipeline

1. Collect raw data on Bybit / Deribit / macro calendars.
2. Materialize 15m bars and 1s micro buckets.
3. Build Bybit-native trend, regime, crowding, and micro features.
4. Build external context blocks as they become available.
5. Generate `continuation_v1` and `stress_reversal_v0` candidates.
6. Label the candidates with a target-aware barrier-based event dataset.
7. Run walk-forward experiments.
8. Run project verification.

Example:

```bash
btc-alert-engine materialize bybit-bars --input ./data/raw
btc-alert-engine materialize micro-buckets --input ./data/raw
btc-alert-engine features bybit-foundation --data-dir ./data
btc-alert-engine features options-deribit --data-dir ./data
btc-alert-engine features macro-veto --data-dir ./data
btc-alert-engine features glassnode-options --data-dir ./data
btc-alert-engine features glassnode-onchain --data-dir ./data
btc-alert-engine features cryptoquant-onchain --data-dir ./data
btc-alert-engine features coinglass-overlay --data-dir ./data
btc-alert-engine signals bybit-candidates --data-dir ./data --enable-macro-veto --sides long short
btc-alert-engine research label-candidates --data-dir ./data --latency-ms 0
btc-alert-engine research walkforward --data-dir ./data --registry ./research_registry.yaml --skip-missing
btc-alert-engine verify project --data-dir ./data --registry ./research_registry.yaml --contracts ./feature_contracts.yaml --strict
```

Derived outputs are written under `data/derived/` in partitioned NDJSON form.

## Labeling semantics

- entries prefer the first recoverable raw Bybit best bid/ask after the trigger closes
- if raw quote recovery is unavailable, labeling falls back to 1-second micro-bucket quotes, then to the next 15-minute trade-bar open
- stop / target evaluation prefers sequential first-touch simulation on raw quote and trade events
- if only 1-second buckets are available, same-bucket double touches resolve conservatively as stop-first because the exact within-second order is unknown
- take-profit is module-specific: `continuation_v1` uses **2.0R**, `stress_reversal_v0` uses **1.5R**
- realized `R` in research outputs is derived from the candidate target, not hard-coded

## Walk-forward experiments

The walk-forward runner reads `research_registry.yaml`, rebuilds experiment-specific candidate sets, trains baseline / challenger models, calibrates probabilities, evaluates both event-level and one-position-at-a-time executable portfolio metrics, and writes promotion-ready reports under:

```text
reports/<data-dir-name>/walkforward/
```

Files:
- `manifest.json` (now includes report provenance hashes for code, registry, contracts, and data state)
- `fold_metrics.csv`
- `summary_metrics.csv`
- `promotion_decisions.csv`
- `promotion_report.md`

Important semantics:
- generic metrics such as `expectancy_r_per_alert`, `alerts_per_week`, and `worst_20_trade_drawdown_r` now refer to the executable one-position-at-a-time portfolio path
- matching `event_*` columns in the CSV outputs preserve the unconstrained event-level view for diagnostics
- alert budgets are computed against the actual split window duration, not the candidate timestamp span

Experiments that require unavailable feature blocks, such as optional vendor overlays, are skipped automatically when `--skip-missing` is set.

For the first real-data pass, use `research_registry_pilot.yaml` together with `real_data_pilot_runbook.md`. The pilot registry shortens the train/calibration/test windows so you can validate the pipeline on a fresh multi-month slice before moving to the full registry.

A convenience script is also included at `scripts/run_real_data_pilot.sh`.

Symmetric long/short research presets are included as:
- `research_registry_smoke_symmetric.yaml`
- `research_registry_broad_test_symmetric.yaml`
- `scripts/run_real_data_smoke_walkforward_symmetric.sh`
- `scripts/run_real_data_broad_walkforward_symmetric.sh`


## Verification

Project verification writes reports under:

```text
reports/<data-dir-name>/verification/
```

Files:
- `verification_report.json`
- `verification_report.md`

Verification checks:
- registry blocks vs feature contracts
- contract fields vs implemented schema classes
- purge / embargo discipline versus label horizon
- derived block parseability and timestamp hygiene
- experiment build readiness and skip reasons
- warnings when execution falls back entirely to bar-level entry/path logic
- warnings when microstructure experiments are requested without usable micro feature rows
- raw execution-tape availability and gap/recovery status

## Design rules

1. Persist raw messages before normalization.
2. Never let soft or mutable vendor data leak into entry labels or intrabar gates.
3. Keep Bybit as execution truth for fills, slippage, and live gating.
4. Make replay deterministic before doing any model work.
5. Prefer small, explicit components over a big framework.
