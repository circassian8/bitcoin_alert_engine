# Walk-forward promotion report: btc_bybit_alert_engine_smoke_fast_symmetric_v1

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
| S0_native_continuation | baseline | -0.6667 | 2.42 | 0.111 | 3.23 | 0.083 | 5.000 |
| S1_native_plus_regime | baseline | -0.1936 | 2.42 | 0.222 | 3.77 | 0.143 | 3.000 |
| S2_plus_crowding_feature | baseline | -0.5000 | 1.62 | 0.167 | 2.15 | 0.125 | 2.000 |
| S3_plus_crowding_veto | baseline | -0.5714 | 1.88 | 0.143 | 2.69 | 0.100 | 3.000 |

## Promotion decisions

| experiment | selected model | parent | promote | reason | Δ expectancy | DD improvement % |
|---|---|---|---:|---|---:|---:|
| S0_native_continuation | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| S1_native_plus_regime | baseline | S0_native_continuation | no | did_not_meet_promotion_rules | 0.4731 | 40.0 |
| S2_plus_crowding_feature | baseline | S1_native_plus_regime | no | did_not_meet_promotion_rules | -0.3064 | 33.3 |
| S3_plus_crowding_veto | baseline | S1_native_plus_regime | no | did_not_meet_promotion_rules | -0.3778 | 0.0 |
