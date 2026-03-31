# Project verification report

Errors: **0**
Warnings: **9**

## Static checks

| level | code | message |
|---|---|---|
| warning | execution_tape_empty | Raw execution tape exists but contains 0 rows; entry and path simulation will fall back to micro buckets or bars. |
| warning | bar_only_entry_fallback | Experiment `S0_native_continuation` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset. |
| warning | bar_only_path_fallback | Experiment `S0_native_continuation` labeled all paths from trade_bars; intrabar barrier fidelity is degraded. |
| warning | bar_only_entry_fallback | Experiment `S1_native_plus_regime` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset. |
| warning | bar_only_path_fallback | Experiment `S1_native_plus_regime` labeled all paths from trade_bars; intrabar barrier fidelity is degraded. |
| warning | bar_only_entry_fallback | Experiment `S2_plus_crowding_feature` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset. |
| warning | bar_only_path_fallback | Experiment `S2_plus_crowding_feature` labeled all paths from trade_bars; intrabar barrier fidelity is degraded. |
| warning | bar_only_entry_fallback | Experiment `S3_plus_crowding_veto` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset. |
| warning | bar_only_path_fallback | Experiment `S3_plus_crowding_veto` labeled all paths from trade_bars; intrabar barrier fidelity is degraded. |

## Block statuses

| block | exists | rows | monotonic ts | unique ts | parse errors |
|---|---:|---:|---:|---:|---:|
| trend_bybit | yes | 78124 | yes | yes | 0 |
| regime_bybit | yes | 78124 | yes | yes | 0 |
| crowding_bybit_feature | yes | 78241 | yes | yes | 0 |
| crowding_bybit_veto | yes | 78241 | yes | yes | 0 |
| micro_bybit_score | no | 0 | no | no | 0 |
| micro_bybit_gate | no | 0 | no | no | 0 |
| options_deribit | yes | 78124 | yes | yes | 0 |
| options_glassnode | no | 0 | no | no | 0 |
| onchain_glassnode | no | 0 | no | no | 0 |
| onchain_cryptoquant | no | 0 | no | no | 0 |
| aggregate_derivs_coinglass | no | 0 | no | no | 0 |
| macro_veto | yes | 78124 | yes | yes | 0 |

## Execution tape

| exists | rows | gaps | out-of-order | state errors |
|---:|---:|---:|---:|---:|
| yes | 0 | 0 | 0 | 0 |

## Experiment statuses

| experiment | blocks | skip reason | rows | candidates | labels | entry sources | path sources |
|---|---|---|---:|---:|---:|---|---|
| S0_native_continuation | trend_bybit | - | 167 | 167 | 167 | bar_open:167 | trade_bars:167 |
| S1_native_plus_regime | trend_bybit, regime_bybit | - | 166 | 166 | 166 | bar_open:166 | trade_bars:166 |
| S2_plus_crowding_feature | trend_bybit, regime_bybit, crowding_bybit_feature | - | 166 | 166 | 166 | bar_open:166 | trade_bars:166 |
| S3_plus_crowding_veto | trend_bybit, regime_bybit, crowding_bybit_veto | - | 159 | 159 | 159 | bar_open:159 | trade_bars:159 |
