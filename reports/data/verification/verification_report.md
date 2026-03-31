# Project verification report

Errors: **0**
Warnings: **0**

## Static checks

No static issues were found.

## Block statuses

| block | exists | rows | monotonic ts | unique ts | parse errors |
|---|---:|---:|---:|---:|---:|
| trend_bybit | yes | 19747 | yes | yes | 0 |
| regime_bybit | yes | 19747 | yes | yes | 0 |
| crowding_bybit_feature | yes | 19777 | yes | yes | 0 |
| crowding_bybit_veto | yes | 19777 | yes | yes | 0 |
| micro_bybit_score | no | 0 | no | no | 0 |
| micro_bybit_gate | no | 0 | no | no | 0 |
| options_deribit | yes | 19747 | yes | yes | 0 |
| options_glassnode | no | 0 | no | no | 0 |
| onchain_glassnode | no | 0 | no | no | 0 |
| onchain_cryptoquant | no | 0 | no | no | 0 |
| aggregate_derivs_coinglass | no | 0 | no | no | 0 |
| macro_veto | yes | 19747 | yes | yes | 0 |

## Execution tape

| exists | rows | gaps | out-of-order | state errors |
|---:|---:|---:|---:|---:|
| yes | 0 | 0 | 0 | 0 |

## Experiment statuses

| experiment | skip reason | rows | candidates | labels |
|---|---|---:|---:|---:|
| A0_native_continuation | - | 20 | 20 | 20 |
| A1_native_plus_soft_regime | - | 20 | 20 | 20 |
| A2_native_plus_bybit_crowding_feature | - | 20 | 20 | 20 |
| A3_native_plus_bybit_crowding_veto | - | 20 | 20 | 20 |
| A4_native_plus_micro_score | - | 20 | 20 | 20 |
| A5_native_plus_micro_gate | no_candidates | 0 | 0 | 0 |
| A6_plus_deribit_options | - | 20 | 20 | 20 |
| A7_plus_glassnode_options | - | 20 | 20 | 20 |
| A8_plus_macro_veto | - | 20 | 20 | 20 |
| A9_plus_glassnode_onchain | - | 20 | 20 | 20 |
| A10_plus_coinglass_overlay | - | 20 | 20 | 20 |
| A11_cryptoquant_instead_of_glassnode | - | 20 | 20 | 20 |
| R1_stress_reversal | no_candidates | 0 | 0 | 0 |
