from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_export_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "export_final_walkforward_alerts.py"
    spec = importlib.util.spec_from_file_location("export_final_walkforward_alerts", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_enrich_keeps_short_risk_and_pnl_signs_consistent() -> None:
    module = _load_export_module()
    frame = pd.DataFrame(
        [
            {
                "ts_dt": "2026-01-01T00:00:00Z",
                "side": "short",
                "realized_r": -1.0,
                "executed_entry": 100.0,
                "stop_y": 101.0,
            },
            {
                "ts_dt": "2026-01-01T01:00:00Z",
                "side": "short",
                "realized_r": 1.25,
                "executed_entry": 100.0,
                "stop_y": 101.0,
            },
        ]
    )

    enriched = module._enrich(frame, notional_usd=5000.0)

    assert enriched["signal_risk"].tolist() == [1.0, 1.0]
    assert enriched["risk_usd"].tolist() == [50.0, 50.0]
    assert enriched["pnl_usd"].tolist() == [-50.0, 62.5]
    assert enriched["cumulative_pnl_usd"].tolist() == [-50.0, 12.5]
