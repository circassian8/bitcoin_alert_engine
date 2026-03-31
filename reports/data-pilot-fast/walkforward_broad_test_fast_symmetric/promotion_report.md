# Walk-forward promotion report: btc_bybit_alert_engine_broad_test_fast_symmetric_v1

Primary budget: **3 alerts/week**
Portfolio evaluation: **one_position_at_a_time / first_signal**, cooldown **0 min**

## Selected models

- `S0_native_continuation` → `baseline`
- `S1_native_plus_regime` → `baseline`
- `S2_plus_crowding_feature` → `baseline`
- `S3_plus_crowding_veto` → `baseline`

## Final split summary

| experiment | model | portfolio expectancy R/trade | portfolio alerts/week | portfolio precision | event alerts/week | event precision | worst 20-trade DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0_native_continuation | baseline | 0.1342 | 2.63 | 0.356 | 4.97 | 0.377 | 8.742 |
| S1_native_plus_regime | baseline | 0.0838 | 2.58 | 0.343 | 4.71 | 0.365 | 11.000 |
| S2_plus_crowding_feature | baseline | 0.1734 | 2.78 | 0.364 | 5.46 | 0.376 | 10.742 |
| S3_plus_crowding_veto | baseline | 0.1501 | 2.52 | 0.361 | 4.68 | 0.378 | 7.742 |

## Promotion decisions

| experiment | selected model | parent | promote | reason | Δ expectancy | DD improvement % |
|---|---|---|---:|---|---:|---:|
| S0_native_continuation | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| S1_native_plus_regime | baseline | S0_native_continuation | no | did_not_meet_promotion_rules | -0.0504 | -25.8 |
| S2_plus_crowding_feature | baseline | S1_native_plus_regime | no | did_not_meet_promotion_rules | 0.0895 | 2.3 |
| S3_plus_crowding_veto | baseline | S1_native_plus_regime | no | did_not_meet_promotion_rules | 0.0662 | 29.6 |
