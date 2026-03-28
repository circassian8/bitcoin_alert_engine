from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from btc_alert_engine.config import default_reports_root, load_feature_contracts, load_research_registry
from btc_alert_engine.research.execution import build_raw_execution_tape
from btc_alert_engine.research.experiments import (
    BLOCK_NAMESPACE_ALIASES,
    FEATURE_SCHEMA_MAP,
    build_experiment_event_frame,
    block_path,
)
from btc_alert_engine.storage.partitioned_ndjson import iter_json_records


@dataclass(slots=True)
class StaticCheckResult:
    level: str
    code: str
    message: str


@dataclass(slots=True)
class BlockDataStatus:
    block: str
    exists: bool
    rows: int
    monotonic_ts: bool
    unique_ts: bool
    parse_errors: int


@dataclass(slots=True)
class ExecutionTapeStatus:
    exists: bool
    rows: int
    gap_count: int
    out_of_order_count: int
    state_error_count: int


@dataclass(slots=True)
class ExperimentStatus:
    experiment_id: str
    blocks: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    rows: int = 0
    candidate_count: int = 0
    label_count: int = 0
    entry_source_counts: dict[str, int] = field(default_factory=dict)
    path_source_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationReport:
    static_checks: list[StaticCheckResult]
    block_statuses: list[BlockDataStatus]
    execution_tape_status: ExecutionTapeStatus | None
    experiment_statuses: list[ExperimentStatus]

    @property
    def errors(self) -> list[StaticCheckResult]:
        return [item for item in self.static_checks if item.level == "error"]

    @property
    def warnings(self) -> list[StaticCheckResult]:
        return [item for item in self.static_checks if item.level == "warning"]


def _model_field_names(model_cls: type) -> set[str]:
    return set(getattr(model_cls, "model_fields", {}).keys())


def _max_label_horizon_hours(registry: Any) -> int:
    timeout_map = registry.labeling.get("timeout_15m_bars_by_module") or {}
    if timeout_map:
        max_bars = max(int(value) for value in timeout_map.values())
    else:
        max_bars = int(registry.labeling.get("timeout_15m_bars", 0))
    return max_bars * 15 // 60


def run_static_contract_checks(registry_path: str | Path, contracts_path: str | Path) -> list[StaticCheckResult]:
    registry = load_research_registry(registry_path)
    contracts = load_feature_contracts(contracts_path)
    results: list[StaticCheckResult] = []

    contract_blocks = set(contracts.feature_blocks.keys())
    registry_blocks = {block for experiment in registry.experiments for block in experiment.get("blocks", [])}

    primary_label = registry.labeling.get("primary_label")
    if primary_label not in {None, "tp_before_sl_within_horizon"}:
        results.append(
            StaticCheckResult(
                "warning",
                "deprecated_primary_label",
                f"Registry primary_label `{primary_label}` is deprecated; use `tp_before_sl_within_horizon` for target-aware labeling.",
            )
        )

    target_map = registry.labeling.get("take_profit_r_by_module") or {}
    fixed_take_profit = registry.labeling.get("take_profit_r")
    referenced_generators = {str(experiment.get("generator")) for experiment in registry.experiments}
    if referenced_generators and not target_map and fixed_take_profit is not None and len(referenced_generators) > 1:
        results.append(
            StaticCheckResult(
                "warning",
                "fixed_take_profit_r_across_generators",
                "Registry uses a single take_profit_r across multiple generators; prefer take_profit_r_by_module for target-aware labels.",
            )
        )
    for generator in sorted(referenced_generators):
        if target_map and generator not in target_map:
            results.append(
                StaticCheckResult(
                    "warning",
                    "generator_missing_take_profit_mapping",
                    f"Generator `{generator}` is referenced by experiments but missing from labeling.take_profit_r_by_module.",
                )
            )

    for block in sorted(registry_blocks - contract_blocks):
        results.append(StaticCheckResult("error", "registry_block_missing_from_contracts", f"Block `{block}` is used by the registry but missing from feature contracts."))
    for block in sorted(contract_blocks - set(BLOCK_NAMESPACE_ALIASES.keys())):
        results.append(StaticCheckResult("warning", "contract_block_missing_namespace_alias", f"Block `{block}` has no namespace alias entry; default namespace resolution will be used."))

    for block, spec in contracts.feature_blocks.items():
        model_cls = FEATURE_SCHEMA_MAP.get(block)
        if model_cls is None:
            results.append(StaticCheckResult("warning", "contract_block_missing_schema_map", f"Block `{block}` has no schema class mapped in code."))
            continue
        missing_fields = set(spec.get("fields", [])) - _model_field_names(model_cls)
        if missing_fields:
            results.append(
                StaticCheckResult(
                    "error",
                    "contract_fields_missing_from_schema",
                    f"Block `{block}` is missing schema fields: {', '.join(sorted(missing_fields))}",
                )
            )

    for experiment in registry.experiments:
        exp_id = str(experiment["id"])
        for block in experiment.get("blocks", []):
            if block not in FEATURE_SCHEMA_MAP and block in contract_blocks:
                results.append(StaticCheckResult("warning", "experiment_block_not_implemented", f"Experiment `{exp_id}` uses `{block}`, which does not have a concrete schema implementation."))

    max_horizon_hours = _max_label_horizon_hours(registry)
    outer = registry.validation.get("outer_walk_forward", {})
    purge_hours = int(outer.get("purge_hours", 0) or 0)
    embargo_hours = int(outer.get("embargo_hours", 0) or 0)
    if max_horizon_hours > 0 and purge_hours < max_horizon_hours:
        results.append(
            StaticCheckResult(
                "warning",
                "purge_shorter_than_label_horizon",
                f"Outer walk-forward purge_hours={purge_hours} is shorter than the max label horizon of {max_horizon_hours}h.",
            )
        )
    if max_horizon_hours > 0 and embargo_hours < max_horizon_hours:
        results.append(
            StaticCheckResult(
                "warning",
                "embargo_shorter_than_label_horizon",
                f"Outer walk-forward embargo_hours={embargo_hours} is shorter than the max label horizon of {max_horizon_hours}h.",
            )
        )

    return results


def inspect_derived_blocks(derived_dir: str | Path, symbol: str, contracts_path: str | Path) -> list[BlockDataStatus]:
    contracts = load_feature_contracts(contracts_path)
    statuses: list[BlockDataStatus] = []
    for block in contracts.feature_blocks.keys():
        path = block_path(Path(derived_dir), symbol, block)
        if not path.exists():
            statuses.append(BlockDataStatus(block=block, exists=False, rows=0, monotonic_ts=False, unique_ts=False, parse_errors=0))
            continue
        rows = 0
        parse_errors = 0
        ts_values: list[int] = []
        model_cls = FEATURE_SCHEMA_MAP.get(block)
        for record in iter_json_records([path]):
            rows += 1
            try:
                if model_cls is not None:
                    model_cls.model_validate(record)
                ts_values.append(int(record["ts"]))
            except Exception:
                parse_errors += 1
        monotonic = ts_values == sorted(ts_values) if ts_values else True
        unique_ts = len(ts_values) == len(set(ts_values))
        statuses.append(BlockDataStatus(block=block, exists=True, rows=rows, monotonic_ts=monotonic, unique_ts=unique_ts, parse_errors=parse_errors))
    return statuses


def inspect_execution_tape(*, data_dir: str | Path, symbol: str) -> ExecutionTapeStatus | None:
    raw_dir = Path(data_dir) / "raw"
    if not raw_dir.exists():
        return None
    tape, stats = build_raw_execution_tape([raw_dir], symbol=symbol, tolerate_gaps=True)
    return ExecutionTapeStatus(
        exists=True,
        rows=int(len(tape)),
        gap_count=int(stats.gap_count),
        out_of_order_count=int(stats.out_of_order_count),
        state_error_count=int(stats.state_error_count),
    )


def inspect_experiments(
    *,
    derived_dir: str | Path,
    registry_path: str | Path,
    symbol: str,
    slippage_bps: float = 1.0,
) -> list[ExperimentStatus]:
    registry = load_research_registry(registry_path)
    statuses: list[ExperimentStatus] = []
    raw_dir = Path(derived_dir).parent / "raw"
    raw_paths = [raw_dir] if raw_dir.exists() else None
    raw_tape = None
    if raw_paths:
        raw_tape, _ = build_raw_execution_tape(raw_paths, symbol=symbol, tolerate_gaps=True)
    latency_ms = int(registry.labeling.get("latency_ms", 0))
    for experiment in registry.experiments:
        dataset = build_experiment_event_frame(
            Path(derived_dir),
            symbol,
            experiment,
            raw_paths=raw_paths,
            raw_tape=raw_tape,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            skip_missing=False,
        )
        entry_source_counts = (
            dataset.labels["entry_source"].value_counts(dropna=False).astype(int).to_dict()
            if not dataset.labels.empty and "entry_source" in dataset.labels.columns
            else {}
        )
        path_source_counts = (
            dataset.labels["path_source"].value_counts(dropna=False).astype(int).to_dict()
            if not dataset.labels.empty and "path_source" in dataset.labels.columns
            else {}
        )
        statuses.append(
            ExperimentStatus(
                experiment_id=str(experiment["id"]),
                blocks=list(experiment.get("blocks", [])),
                skip_reason=dataset.skip_reason,
                rows=len(dataset.frame),
                candidate_count=len(dataset.candidates),
                label_count=len(dataset.labels),
                entry_source_counts=entry_source_counts,
                path_source_counts=path_source_counts,
            )
        )
    return statuses


def _reports_exist_without_inputs(data_dir: Path, experiment_statuses: list[ExperimentStatus]) -> bool:
    report_roots = [default_reports_root(data_dir), data_dir / "reports"]
    report_files: list[Path] = []
    for reports_dir in report_roots:
        if reports_dir.exists():
            report_files.extend(reports_dir.rglob("summary_metrics.csv"))
            report_files.extend(reports_dir.rglob("promotion_report.md"))
    if not report_files:
        return False
    if not experiment_statuses:
        return False
    return all((status.skip_reason or "").startswith("missing_") or status.rows == 0 for status in experiment_statuses)


def generate_verification_report(
    *,
    data_dir: str | Path,
    registry_path: str | Path,
    contracts_path: str | Path,
    symbol: str,
    slippage_bps: float = 1.0,
) -> VerificationReport:
    data_path = Path(data_dir)
    derived_dir = data_path / "derived"
    static_checks = run_static_contract_checks(registry_path, contracts_path)
    block_statuses = inspect_derived_blocks(derived_dir, symbol, contracts_path)
    execution_tape_status = inspect_execution_tape(data_dir=data_dir, symbol=symbol)
    experiment_statuses = inspect_experiments(derived_dir=derived_dir, registry_path=registry_path, symbol=symbol, slippage_bps=slippage_bps)

    block_map = {item.block: item for item in block_statuses}
    if execution_tape_status is not None and execution_tape_status.rows == 0:
        static_checks.append(
            StaticCheckResult(
                "warning",
                "execution_tape_empty",
                "Raw execution tape exists but contains 0 rows; entry and path simulation will fall back to micro buckets or bars.",
            )
        )

    if _reports_exist_without_inputs(data_path, experiment_statuses):
        static_checks.append(
            StaticCheckResult(
                "warning",
                "reports_exist_but_inputs_missing",
                "Report artifacts exist under data_dir/reports, but the current derived inputs are insufficient to rebuild the experiments. Treat the bundled reports as stale until you rerun the pipeline.",
            )
        )

    for status in experiment_statuses:
        if {"micro_bybit_score", "micro_bybit_gate"} & set(status.blocks):
            micro_status = block_map.get("micro_bybit_score") or block_map.get("micro_bybit_gate")
            if micro_status is None or not micro_status.exists or micro_status.rows == 0:
                static_checks.append(
                    StaticCheckResult(
                        "warning",
                        "micro_block_missing_for_experiment",
                        f"Experiment `{status.experiment_id}` requests microstructure blocks but no usable micro feature rows were found.",
                    )
                )
        if status.label_count > 0 and status.entry_source_counts and set(status.entry_source_counts.keys()) == {"bar_open"}:
            static_checks.append(
                StaticCheckResult(
                    "warning",
                    "bar_only_entry_fallback",
                    f"Experiment `{status.experiment_id}` labeled all entries from bar_open; quote-aware entry simulation is unavailable for this dataset.",
                )
            )
        if status.label_count > 0 and status.path_source_counts and set(status.path_source_counts.keys()) == {"trade_bars"}:
            static_checks.append(
                StaticCheckResult(
                    "warning",
                    "bar_only_path_fallback",
                    f"Experiment `{status.experiment_id}` labeled all paths from trade_bars; intrabar barrier fidelity is degraded.",
                )
            )

    return VerificationReport(
        static_checks=static_checks,
        block_statuses=block_statuses,
        execution_tape_status=execution_tape_status,
        experiment_statuses=experiment_statuses,
    )


def write_verification_artifacts(report: VerificationReport, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "static_checks": [asdict(item) for item in report.static_checks],
        "block_statuses": [asdict(item) for item in report.block_statuses],
        "execution_tape_status": asdict(report.execution_tape_status) if report.execution_tape_status is not None else None,
        "experiment_statuses": [asdict(item) for item in report.experiment_statuses],
    }
    (output_path / "verification_report.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    lines = [
        "# Project verification report",
        "",
        f"Errors: **{len(report.errors)}**",
        f"Warnings: **{len(report.warnings)}**",
        "",
        "## Static checks",
        "",
    ]
    if not report.static_checks:
        lines.append("No static issues were found.")
    else:
        lines.append("| level | code | message |")
        lines.append("|---|---|---|")
        for item in report.static_checks:
            lines.append(f"| {item.level} | {item.code} | {item.message} |")

    lines.extend(["", "## Block statuses", ""])
    if not report.block_statuses:
        lines.append("No block statuses were produced.")
    else:
        lines.append("| block | exists | rows | monotonic ts | unique ts | parse errors |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for item in report.block_statuses:
            lines.append(f"| {item.block} | {'yes' if item.exists else 'no'} | {item.rows} | {'yes' if item.monotonic_ts else 'no'} | {'yes' if item.unique_ts else 'no'} | {item.parse_errors} |")

    lines.extend(["", "## Execution tape", ""])
    if report.execution_tape_status is None:
        lines.append("No raw execution tape was found.")
    else:
        item = report.execution_tape_status
        lines.append("| exists | rows | gaps | out-of-order | state errors |")
        lines.append("|---:|---:|---:|---:|---:|")
        lines.append(f"| {'yes' if item.exists else 'no'} | {item.rows} | {item.gap_count} | {item.out_of_order_count} | {item.state_error_count} |")

    lines.extend(["", "## Experiment statuses", ""])
    if not report.experiment_statuses:
        lines.append("No experiment statuses were produced.")
    else:
        lines.append("| experiment | blocks | skip reason | rows | candidates | labels | entry sources | path sources |")
        lines.append("|---|---|---|---:|---:|---:|---|---|")
        for item in report.experiment_statuses:
            entry_sources = ", ".join(f"{k}:{v}" for k, v in sorted(item.entry_source_counts.items())) or "-"
            path_sources = ", ".join(f"{k}:{v}" for k, v in sorted(item.path_source_counts.items())) or "-"
            blocks = ", ".join(item.blocks) or "-"
            lines.append(f"| {item.experiment_id} | {blocks} | {item.skip_reason or '-'} | {item.rows} | {item.candidate_count} | {item.label_count} | {entry_sources} | {path_sources} |")

    (output_path / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
