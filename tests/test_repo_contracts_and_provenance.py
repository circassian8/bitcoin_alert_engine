from __future__ import annotations

import json
from pathlib import Path

from btc_alert_engine.cli import build_parser
from btc_alert_engine.provenance import report_provenance
from btc_alert_engine.verification.project import inspect_report_manifests


def test_readme_references_existing_symmetric_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    for relative_path in [
        "research_registry_smoke_symmetric.yaml",
        "research_registry_broad_test_symmetric.yaml",
        "scripts/run_real_data_smoke_walkforward_symmetric.sh",
        "scripts/run_real_data_broad_walkforward_symmetric.sh",
    ]:
        assert relative_path in readme
        assert (repo_root / relative_path).exists()


def test_cli_supports_signal_sides_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["signals", "bybit-candidates", "--sides", "long", "short"])
    assert args.sides == ["long", "short"]


def test_report_manifest_provenance_roundtrip(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    (data_dir / "derived").mkdir(parents=True)
    provenance = report_provenance(
        project_root=project_root,
        data_dir=data_dir,
        registry_path=project_root / "research_registry.yaml",
        contracts_path=project_root / "feature_contracts.yaml",
    )
    reports_root = tmp_path / "reports"
    walk_dir = reports_root / "walkforward_test"
    walk_dir.mkdir(parents=True)
    (walk_dir / "manifest.json").write_text(json.dumps({"program_id": "demo", "provenance": provenance}), encoding="utf-8")
    statuses = inspect_report_manifests(reports_root, provenance)
    assert len(statuses) == 1
    status = statuses[0]
    assert status.has_provenance is True
    assert status.registry_match is True
    assert status.source_tree_match is True
    assert status.data_state_match is True
    assert status.feature_contracts_match is True
