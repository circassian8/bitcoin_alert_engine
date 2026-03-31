# Project verification report

Errors: **0**
Warnings: **32**
Strict failures: **32**

## Current provenance

```json
{
  "created_at_utc": "2026-03-30T14:35:44.093891+00:00",
  "package_version": "0.7.0",
  "project_root": "/Users/nasrenkaraf/Documents/Projects/bitcoin_alert_engine",
  "git_commit": "0b586005d5bb5ef72296bfa3dc038aef96af64b8",
  "source_tree_sha256": "69ca4ac8dd250485464f61c0185df4451dff132f1d08547afc7963062ce6f72d",
  "data_dir": "/Users/nasrenkaraf/Documents/Projects/bitcoin_alert_engine/data-pilot-fast",
  "data_state_sha256": "642cdf903f0b4e6d41d176294b0188272e707838644f86b9900506fe037c4a06",
  "registry_path": "/Users/nasrenkaraf/Documents/Projects/bitcoin_alert_engine/research_registry_broad_test_fast_revised_first_signal.yaml",
  "registry_sha256": "15d1ed2cb4720f72e98c8976f025a799cc9b703d98369b9d63c83dc4fd350fe2",
  "feature_contracts_path": "/Users/nasrenkaraf/Documents/Projects/bitcoin_alert_engine/feature_contracts.yaml",
  "feature_contracts_sha256": "49c99e675a975eb3cbc34f065aa87c24a147fa617d501e9345b038934fe4c729"
}
```

## Static checks

| level | code | message |
|---|---|---|
| warning | execution_tape_empty | Raw execution tape exists but contains 0 rows; entry and path simulation will fall back to micro buckets or bars. |
| warning | bar_only_entry_fallback | Experiment `S2_current_fast_crowding_feature` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset. |
| warning | bar_only_path_fallback | Experiment `S2_current_fast_crowding_feature` labeled all paths from trade_bars; intrabar barrier fidelity is degraded. |
| warning | bar_only_entry_fallback | Experiment `F1_setup_anchor_smallbuf_tp125` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset. |
| warning | bar_only_path_fallback | Experiment `F1_setup_anchor_smallbuf_tp125` labeled all paths from trade_bars; intrabar barrier fidelity is degraded. |
| warning | bar_only_entry_fallback | Experiment `F2_setup_anchor_modbuf_tp150` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset. |
| warning | bar_only_path_fallback | Experiment `F2_setup_anchor_modbuf_tp150` labeled all paths from trade_bars; intrabar barrier fidelity is degraded. |
| warning | bar_only_entry_fallback | Experiment `F3_setup_anchor_modbuf_tp125` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset. |
| warning | bar_only_path_fallback | Experiment `F3_setup_anchor_modbuf_tp125` labeled all paths from trade_bars; intrabar barrier fidelity is degraded. |
| warning | report_manifest_data_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_revised_first_signal/manifest.json` does not match the current data state hash. |
| warning | report_manifest_registry_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_revised_highest_probability/manifest.json` was generated from a different registry configuration. |
| warning | report_manifest_data_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_revised_highest_probability/manifest.json` does not match the current data state hash. |
| warning | report_manifest_registry_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_symmetric/manifest.json` was generated from a different registry configuration. |
| warning | report_manifest_code_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_symmetric/manifest.json` does not match the current source tree hash. |
| warning | report_manifest_data_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_symmetric/manifest.json` does not match the current data state hash. |
| warning | report_manifest_contracts_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_symmetric/manifest.json` does not match the current feature contracts hash. |
| warning | report_manifest_registry_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_symmetric_pre_warning_fix/manifest.json` was generated from a different registry configuration. |
| warning | report_manifest_code_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_symmetric_pre_warning_fix/manifest.json` does not match the current source tree hash. |
| warning | report_manifest_data_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_symmetric_pre_warning_fix/manifest.json` does not match the current data state hash. |
| warning | report_manifest_contracts_mismatch | Report manifest `reports/data-pilot-fast/walkforward_broad_test_fast_symmetric_pre_warning_fix/manifest.json` does not match the current feature contracts hash. |
| warning | report_manifest_registry_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_revised_first_signal/manifest.json` was generated from a different registry configuration. |
| warning | report_manifest_data_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_revised_first_signal/manifest.json` does not match the current data state hash. |
| warning | report_manifest_registry_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_revised_highest_probability/manifest.json` was generated from a different registry configuration. |
| warning | report_manifest_data_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_revised_highest_probability/manifest.json` does not match the current data state hash. |
| warning | report_manifest_registry_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_symmetric/manifest.json` was generated from a different registry configuration. |
| warning | report_manifest_code_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_symmetric/manifest.json` does not match the current source tree hash. |
| warning | report_manifest_data_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_symmetric/manifest.json` does not match the current data state hash. |
| warning | report_manifest_contracts_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_symmetric/manifest.json` does not match the current feature contracts hash. |
| warning | report_manifest_registry_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_symmetric_pre_warning_fix/manifest.json` was generated from a different registry configuration. |
| warning | report_manifest_code_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_symmetric_pre_warning_fix/manifest.json` does not match the current source tree hash. |
| warning | report_manifest_data_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_symmetric_pre_warning_fix/manifest.json` does not match the current data state hash. |
| warning | report_manifest_contracts_mismatch | Report manifest `reports/data-pilot-fast/walkforward_smoke_fast_symmetric_pre_warning_fix/manifest.json` does not match the current feature contracts hash. |

## Block statuses

| block | exists | rows | monotonic ts | unique ts | parse errors | missing expected fields |
|---|---:|---:|---:|---:|---:|---|
| trend_bybit | yes | 156248 | no | no | 0 | - |
| trend_bybit_fast | yes | 703347 | no | no | 0 | - |
| regime_bybit | yes | 156248 | no | no | 0 | - |
| regime_bybit_fast | yes | 703347 | no | no | 0 | - |
| crowding_bybit_feature | yes | 234723 | no | no | 0 | - |
| crowding_bybit_veto | yes | 234723 | no | no | 0 | - |
| micro_bybit_score | no | 0 | no | no | 0 | - |
| micro_bybit_gate | no | 0 | no | no | 0 | - |
| options_deribit | no | 0 | no | no | 0 | - |
| options_glassnode | no | 0 | no | no | 0 | - |
| onchain_glassnode | no | 0 | no | no | 0 | - |
| onchain_cryptoquant | no | 0 | no | no | 0 | - |
| aggregate_derivs_coinglass | no | 0 | no | no | 0 | - |
| macro_veto | no | 0 | no | no | 0 | - |

## Execution tape

| exists | rows | gaps | out-of-order | state errors |
|---:|---:|---:|---:|---:|
| yes | 0 | 0 | 0 | 0 |

## Experiment statuses

| experiment | blocks | skip reason | rows | candidates | labels | entry sources | path sources |
|---|---|---|---:|---:|---:|---|---|
| S2_current_fast_crowding_feature | trend_bybit_fast, regime_bybit_fast, crowding_bybit_feature | - | 556 | 556 | 556 | bar_open:556 | trade_bars:556 |
| F1_setup_anchor_smallbuf_tp125 | trend_bybit_fast, regime_bybit_fast, crowding_bybit_feature | - | 556 | 556 | 556 | bar_open:556 | trade_bars:556 |
| F2_setup_anchor_modbuf_tp150 | trend_bybit_fast, regime_bybit_fast, crowding_bybit_feature | - | 556 | 556 | 556 | bar_open:556 | trade_bars:556 |
| F3_setup_anchor_modbuf_tp125 | trend_bybit_fast, regime_bybit_fast, crowding_bybit_feature | - | 556 | 556 | 556 | bar_open:556 | trade_bars:556 |

## Existing report manifests

| path | has provenance | registry | code | data | contracts | program |
|---|---:|---:|---:|---:|---:|---|
| reports/data-pilot-fast/walkforward_broad_test_fast_revised_first_signal/manifest.json | yes | yes | yes | no | yes | btc_bybit_alert_engine_broad_fast_revised_first_signal_v1 |
| reports/data-pilot-fast/walkforward_broad_test_fast_revised_highest_probability/manifest.json | yes | no | yes | no | yes | btc_bybit_alert_engine_broad_fast_revised_highest_probability_v1 |
| reports/data-pilot-fast/walkforward_broad_test_fast_symmetric/manifest.json | yes | no | no | no | no | btc_bybit_alert_engine_broad_test_fast_symmetric_v1 |
| reports/data-pilot-fast/walkforward_broad_test_fast_symmetric_pre_warning_fix/manifest.json | yes | no | no | no | no | btc_bybit_alert_engine_broad_test_fast_symmetric_v1 |
| reports/data-pilot-fast/walkforward_smoke_fast_revised_first_signal/manifest.json | yes | no | yes | no | yes | btc_bybit_alert_engine_smoke_fast_revised_first_signal_v1 |
| reports/data-pilot-fast/walkforward_smoke_fast_revised_highest_probability/manifest.json | yes | no | yes | no | yes | btc_bybit_alert_engine_smoke_fast_revised_highest_probability_v1 |
| reports/data-pilot-fast/walkforward_smoke_fast_symmetric/manifest.json | yes | no | no | no | no | btc_bybit_alert_engine_smoke_fast_symmetric_v1 |
| reports/data-pilot-fast/walkforward_smoke_fast_symmetric_pre_warning_fix/manifest.json | yes | no | no | no | no | btc_bybit_alert_engine_smoke_fast_symmetric_v1 |
