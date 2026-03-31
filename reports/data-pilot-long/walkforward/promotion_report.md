# Walk-forward promotion report: btc_bybit_alert_engine_smoke_v1

Primary budget: **3 alerts/week**

## Selected models

- `S0_native_continuation` → `baseline`
- `S1_native_plus_regime` → `baseline`
- `S2_plus_crowding_feature` → `baseline`
- `S3_plus_crowding_veto` → `baseline`

## Final split summary

| experiment | model | expectancy R/alert | alerts/week | precision | worst 20-trade DD |
|---|---:|---:|---:|---:|---:|
| S0_native_continuation | baseline | 1.4000 | 12.00 | 0.800 | 1.000 |
| S1_native_plus_regime | baseline | -1.0000 | 2.40 | 0.000 | 0.000 |
| S2_plus_crowding_feature | baseline | nan | 0.00 | nan | nan |
| S3_plus_crowding_veto | baseline | nan | 0.00 | nan | nan |

## Promotion decisions

| experiment | selected model | parent | promote | reason | Δ expectancy | DD improvement % |
|---|---|---|---:|---|---:|---:|
| S0_native_continuation | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| S1_native_plus_regime | baseline | S0_native_continuation | no | did_not_meet_promotion_rules | -2.4000 | 100.0 |
| S2_plus_crowding_feature | baseline | S1_native_plus_regime | no | did_not_meet_promotion_rules | 0.0000 | 0.0 |
| S3_plus_crowding_veto | baseline | S1_native_plus_regime | no | did_not_meet_promotion_rules | 0.0000 | 0.0 |
