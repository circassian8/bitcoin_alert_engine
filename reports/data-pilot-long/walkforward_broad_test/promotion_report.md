# Walk-forward promotion report: btc_bybit_alert_engine_broad_test_v1

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
| S0_native_continuation | baseline | 0.5222 | 0.39 | 0.467 | 1.25 | 0.375 | 3.000 |
| S1_native_plus_regime | baseline | 0.5830 | 0.42 | 0.438 | 1.22 | 0.383 | 3.000 |
| S2_plus_crowding_feature | baseline | 0.2108 | 0.47 | 0.278 | 1.38 | 0.340 | 3.000 |
| S3_plus_crowding_veto | baseline | 0.4885 | 0.39 | 0.400 | 1.17 | 0.378 | 3.000 |

## Promotion decisions

| experiment | selected model | parent | promote | reason | Δ expectancy | DD improvement % |
|---|---|---|---:|---|---:|---:|
| S0_native_continuation | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| S1_native_plus_regime | baseline | S0_native_continuation | no | did_not_meet_promotion_rules | 0.0608 | 0.0 |
| S2_plus_crowding_feature | baseline | S1_native_plus_regime | no | did_not_meet_promotion_rules | -0.3722 | 0.0 |
| S3_plus_crowding_veto | baseline | S1_native_plus_regime | no | did_not_meet_promotion_rules | -0.0945 | 0.0 |
