# Walk-forward promotion report: btc_bybit_alert_engine_broad_fast_revised_first_signal_v1

Primary budget: **3 alerts/week**
Portfolio evaluation: **one_position_at_a_time / first_signal**, cooldown **0 min**

## Selected models

- `S2_current_fast_crowding_feature` → `baseline`
- `F1_setup_anchor_smallbuf_tp125` → `baseline`
- `F2_setup_anchor_modbuf_tp150` → `baseline`
- `F3_setup_anchor_modbuf_tp125` → `baseline`

## Final split summary

| experiment | model | portfolio expectancy R/trade | portfolio alerts/week | portfolio precision | event alerts/week | event precision | worst 20-trade DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1_setup_anchor_smallbuf_tp125 | baseline | 0.1585 | 4.03 | 0.497 | 4.74 | 0.484 | 8.702 |
| F2_setup_anchor_modbuf_tp150 | baseline | 0.0983 | 3.49 | 0.410 | 4.09 | 0.401 | 8.214 |
| F3_setup_anchor_modbuf_tp125 | baseline | 0.1514 | 4.01 | 0.494 | 4.74 | 0.489 | 8.714 |
| S2_current_fast_crowding_feature | baseline | 0.0984 | 3.49 | 0.410 | 4.09 | 0.408 | 8.208 |

## Promotion decisions

| experiment | selected model | parent | promote | reason | Δ expectancy | DD improvement % |
|---|---|---|---:|---|---:|---:|
| S2_current_fast_crowding_feature | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| F1_setup_anchor_smallbuf_tp125 | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| F2_setup_anchor_modbuf_tp150 | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
| F3_setup_anchor_modbuf_tp125 | baseline | - | yes | root_experiment | 0.0000 | 0.0 |
