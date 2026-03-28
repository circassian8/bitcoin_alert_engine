# BTC / Bybit alert engine: implementation order

## 1) Freeze the contracts

Commit these first:
- `research_registry.yaml`
- `feature_contracts.yaml`
- `README.md` with design principles

Do not start model work before these are committed.

## 2) Build the raw collectors

Priority order:
1. Bybit order book collector (`orderbook.200.BTCUSDT`)
2. Bybit public trade collector (`publicTrade.BTCUSDT`)
3. Bybit liquidation collector (`allLiquidation.BTCUSDT`)
4. Bybit REST poller (`funding_history`, `open_interest`, `account_ratio`, `premium_index_price_kline`, `index_price_kline`, `kline`, `tickers`, `instruments_info`)
5. Deribit DVOL collector
6. Macro calendar collector
7. Glassnode slow-context poller

### Acceptance checks
- Raw messages are persisted before normalization
- Every message has both exchange timestamp and local receive timestamp
- Reconnect + resubscribe works
- Order book can be rebuilt from snapshot + deltas deterministically
- Gap alarms are emitted when sequence / snapshot logic fails

## 3) Build replay and normalization

Deliverables:
- raw -> normalized parsers
- deterministic replay runner
- L2 book builder
- 1s micro-bucket materializer
- 15m / 1h / 4h bar builders

### Acceptance checks
- Replay of the same raw files yields identical normalized outputs
- Best bid/ask from rebuilt book matches snapshot state after full replay
- Book reset logic handles fresh snapshots correctly

## 4) Build feature jobs

Deliverables:
- trend / regime feature job
- crowding feature job
- microstructure feature job
- options feature job
- macro-event feature job
- optional Glassnode / CryptoQuant jobs with conservative lagging

### Acceptance checks
- Every feature is traceable back to raw inputs
- PIT policy is enforced in code
- Slow / mutable features cannot be used in labels or intrabar gates

## 5) Build candidate generators and labeler

Deliverables:
- `continuation_v1`
- `stress_reversal_v0`
- barrier-based labeler
- slippage / fee / funding model

### Acceptance checks
- Candidate generation is deterministic
- Labels do not read future data beyond the defined horizon
- Entry/exit prices use Bybit-native market truth only

## 6) Build the experiment runner

Deliverables:
- walk-forward splitter
- CPCV selector for inner model search
- logistic baseline
- LightGBM challenger
- probability calibration module
- report generator

### Acceptance checks
- Experiment configs are loaded from YAML only
- Every run writes config hash + git commit + dataset hash
- Reports break out pre/post 2023-10-23

## 7) Run the ablations in this exact order

1. A0_native_continuation
2. A1_native_plus_soft_regime
3. A3_native_plus_bybit_crowding_veto
4. A4_native_plus_micro_score
5. A5_native_plus_micro_gate
6. A6_plus_deribit_options
7. A8_plus_macro_veto
8. A7_plus_glassnode_options
9. A9_plus_glassnode_onchain
10. R1_stress_reversal
11. A10 / A11 only if the prior stack is already strong

## 8) Promotion to shadow-live

A model gets promoted only if it:
- beats its parent at the 3-alerts/week budget
- survives stressed cost assumptions
- remains calibrated on the final untouched test
- does not rely on mutable vendor features for entry timing

Shadow-live mode should produce alerts only. No auto-execution.

## 9) Only after shadow-live is stable

Then add:
- Bybit private order / position / wallet streams
- execution journal
- alert delivery (Telegram / Slack)
- trade review UI

## 10) Things to avoid early

Do **not** build these before the above is working:
- deep learning direction models
- multi-coin engines
- auto-execution
- market making
- giant on-chain feature zoos
