# Walk-forward promotion report: btc_bybit_alert_engine_broad_stop_sensitivity_v1

Primary budget: **3 alerts/week**
Portfolio evaluation: **one_position_at_a_time / first_signal**, cooldown **0 min**

## Selected models

- `S0_native_continuation` → `baseline`
- `S0_native_continuation_stop150` → `baseline`
- `S0_native_continuation_stop175` → `baseline`

## Final split summary

| experiment | model | portfolio expectancy R/trade | portfolio alerts/week | portfolio precision | event alerts/week | event precision | worst 20-trade DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0_native_continuation | baseline | 0.5222 | 0.39 | 0.467 | 1.25 | 0.375 | 3.000 |
| S0_native_continuation_stop150 | baseline | 0.2932 | 0.44 | 0.294 | 1.38 | 0.321 | 5.001 |
| S0_native_continuation_stop175 | baseline | 0.2649 | 0.42 | 0.250 | 1.38 | 0.283 | 4.143 |

## Promotion decisions

| experiment | selected model | parent | promote | reason | Δ expectancy | DD improvement % |
|---|---|---|---:|---|---:|---:|
| S0_native_continuation | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| S0_native_continuation_stop150 | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| S0_native_continuation_stop175 | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
