from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from btc_alert_engine.collectors.bybit_public_ws import (
    BybitLiquidationCollector,
    BybitOrderBookCollector,
    BybitTradeCollector,
)
from btc_alert_engine.collectors.bybit_rest import BybitMarketRestPoller
from btc_alert_engine.collectors.deribit_dvol import DeribitVolatilityIndexPoller
from btc_alert_engine.collectors.macro_events import MacroCsvIngestor
from btc_alert_engine.importers import import_bybit_history_orderbook, import_bybit_history_trades
from btc_alert_engine.backfill import backfill_bybit_rest_history, backfill_deribit_dvol_history
from btc_alert_engine.config import RuntimeSettings, default_reports_root
from btc_alert_engine.features.bybit_foundation import (
    build_crowding_features,
    build_micro_features,
    build_regime_features,
    build_trend_features,
    load_micro_buckets,
    load_price_bars,
)
from btc_alert_engine.features.external_context import (
    build_aggregate_derivs_coinglass_features,
    build_macro_veto_features,
    build_onchain_cryptoquant_features,
    build_onchain_glassnode_features,
    build_options_deribit_features,
    build_options_glassnode_features,
)
from btc_alert_engine.logging_config import configure_logging
from btc_alert_engine.materialize.bybit_foundation import materialize_bybit_bars, materialize_micro_buckets
from btc_alert_engine.normalize.replay import deterministic_replay_hash, replay_top_of_book
from btc_alert_engine.research.labeling import label_candidates
from btc_alert_engine.research.walkforward import run_walkforward_experiments
from btc_alert_engine.schemas import (
    CandidateEvent,
    CrowdingFeatureSnapshot,
    MacroVetoFeatureSnapshot,
    MicroFeatureSnapshot,
    RegimeFeatureSnapshot,
    TrendFeatureSnapshot,
)
from btc_alert_engine.storage.partitioned_ndjson import PartitionedNdjsonWriter
from btc_alert_engine.storage.raw_ndjson import RawEventWriter
from btc_alert_engine.strategy.bybit_candidates import (
    build_continuation_candidates,
    build_stress_reversal_candidates,
    load_feature_models,
)
from btc_alert_engine.verification.project import generate_verification_report, write_verification_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc-alert-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect_sub = collect.add_subparsers(dest="collector", required=True)

    def add_common_collect_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbol", default="BTCUSDT")
        p.add_argument("--category", default="linear")
        p.add_argument("--data-dir", default=None)
        p.add_argument("--testnet", action="store_true")

    p = collect_sub.add_parser("bybit-orderbook")
    add_common_collect_args(p)
    p.add_argument("--depth", type=int, default=200)

    p = collect_sub.add_parser("bybit-trades")
    add_common_collect_args(p)

    p = collect_sub.add_parser("bybit-liquidations")
    add_common_collect_args(p)

    p = collect_sub.add_parser("bybit-rest")
    add_common_collect_args(p)

    backfill = subparsers.add_parser("backfill")
    backfill_sub = backfill.add_subparsers(dest="backfill_job", required=True)

    p = backfill_sub.add_parser("bybit-rest-history")
    add_common_collect_args(p)
    p.add_argument("--start", required=True, help="Start timestamp (ms or ISO-8601)")
    p.add_argument("--end", required=True, help="End timestamp (ms or ISO-8601)")
    p.add_argument("--datasets", nargs="+", default=None, help="Optional dataset names, e.g. kline_15 funding_history open_interest account_ratio")
    p.add_argument("--oi-interval", default="5min", choices=["5min", "15min", "30min", "1h", "4h", "1d"])
    p.add_argument("--account-ratio-period", default="5min", choices=["5min", "15min", "30min", "1h", "4h", "1d"])
    p.add_argument("--sleep-s", type=float, default=0.0)

    p = backfill_sub.add_parser("deribit-dvol-history")
    p.add_argument("--currency", default="BTC")
    p.add_argument("--resolution", default="60")
    p.add_argument("--start", required=True, help="Start timestamp (ms or ISO-8601)")
    p.add_argument("--end", required=True, help="End timestamp (ms or ISO-8601)")
    p.add_argument("--sleep-s", type=float, default=0.0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--testnet", action="store_true")

    p = collect_sub.add_parser("deribit-dvol")
    p.add_argument("--currency", default="BTC")
    p.add_argument("--resolution", default="60")
    p.add_argument("--interval-s", type=float, default=60.0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--testnet", action="store_true")

    replay = subparsers.add_parser("replay")
    replay_sub = replay.add_subparsers(dest="mode", required=True)
    p = replay_sub.add_parser("top-of-book")
    p.add_argument("--input", required=True)
    p.add_argument("--symbol", default="BTCUSDT")

    p = subparsers.add_parser("import-macro-csv")
    p.add_argument("--csv", required=True)
    p.add_argument("--data-dir", default=None)


    import_cmd = subparsers.add_parser("import")
    import_sub = import_cmd.add_subparsers(dest="import_job", required=True)

    p = import_sub.add_parser("bybit-history-trades")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--symbol", default=None)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--data-dir", default=None)

    p = import_sub.add_parser("bybit-history-orderbook")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--symbol", default=None)
    p.add_argument("--depth", type=int, default=500)
    p.add_argument("--data-dir", default=None)

    materialize = subparsers.add_parser("materialize")
    materialize_sub = materialize.add_subparsers(dest="materializer", required=True)

    p = materialize_sub.add_parser("bybit-bars")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default=None)

    p = materialize_sub.add_parser("micro-buckets")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default=None)

    features = subparsers.add_parser("features")
    features_sub = features.add_subparsers(dest="feature_job", required=True)

    def add_feature_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbol", default="BTCUSDT")
        p.add_argument("--data-dir", default=None)
        p.add_argument("--raw-input", nargs="+", default=None)

    for job in [
        "bybit-foundation",
        "options-deribit",
        "macro-veto",
        "glassnode-options",
        "glassnode-onchain",
        "cryptoquant-onchain",
        "coinglass-overlay",
    ]:
        p = features_sub.add_parser(job)
        add_feature_args(p)

    signals = subparsers.add_parser("signals")
    signals_sub = signals.add_subparsers(dest="signal_job", required=True)
    p = signals_sub.add_parser("bybit-candidates")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--enable-macro-veto", action="store_true")
    p.add_argument("--sides", nargs="+", choices=["long", "short"], default=["long"])

    research = subparsers.add_parser("research")
    research_sub = research.add_subparsers(dest="research_job", required=True)
    p = research_sub.add_parser("label-candidates")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--latency-ms", type=int, default=0)

    p = research_sub.add_parser("walkforward")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--registry", default="research_registry.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--experiments", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--slippage-bps", type=float, default=1.0)

    verify = subparsers.add_parser("verify")
    verify_sub = verify.add_subparsers(dest="verify_job", required=True)
    p = verify_sub.add_parser("project")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--registry", default="research_registry.yaml")
    p.add_argument("--contracts", default="feature_contracts.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--strict", action="store_true")

    return parser


def _parse_ts_arg(value: str) -> int:
    if value.isdigit():
        return int(value)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _load_foundation_inputs(derived_dir: Path, symbol: str):
    trade_path = derived_dir / "bars" / "bybit" / "trade_15m" / symbol
    index_path = derived_dir / "bars" / "bybit" / "index_price_15m" / symbol
    premium_path = derived_dir / "bars" / "bybit" / "premium_index_15m" / symbol
    micro_path = derived_dir / "micro" / "bybit" / "1s" / symbol

    trade_bars = load_price_bars([trade_path]) if trade_path.exists() else []
    index_bars = load_price_bars([index_path]) if index_path.exists() else []
    premium_bars = load_price_bars([premium_path]) if premium_path.exists() else []
    micro_buckets = load_micro_buckets([micro_path]) if micro_path.exists() else []
    return trade_bars, index_bars, premium_bars, micro_buckets


def _write_feature_block(derived_dir: Path, namespace: str, symbol: str, items: list[object]) -> int:
    with PartitionedNdjsonWriter(derived_dir) as writer:
        for item in items:
            writer.write(namespace=namespace, symbol=symbol, ts_ms=int(getattr(item, "ts")), record=item)
    return len(items)


async def _run_async(args: argparse.Namespace) -> None:
    settings = RuntimeSettings.from_env()
    data_dir = Path(args.data_dir) if getattr(args, "data_dir", None) else settings.data_dir
    raw_dir = data_dir / "raw"

    if args.command == "backfill":
        with RawEventWriter(raw_dir) as writer:
            if args.backfill_job == "bybit-rest-history":
                counts = await backfill_bybit_rest_history(
                    writer=writer,
                    symbol=args.symbol,
                    category=args.category,
                    start_ms=_parse_ts_arg(args.start),
                    end_ms=_parse_ts_arg(args.end),
                    datasets=args.datasets,
                    oi_interval=args.oi_interval,
                    account_ratio_period=args.account_ratio_period,
                    testnet=args.testnet or settings.bybit_testnet,
                    sleep_s=args.sleep_s,
                )
                print(" ".join(f"{name}={count}" for name, count in counts.items()))
                return
            if args.backfill_job == "deribit-dvol-history":
                count = await backfill_deribit_dvol_history(
                    writer=writer,
                    currency=args.currency,
                    resolution=args.resolution,
                    start_ms=_parse_ts_arg(args.start),
                    end_ms=_parse_ts_arg(args.end),
                    testnet=args.testnet or settings.deribit_testnet,
                    sleep_s=args.sleep_s,
                )
                print(f"wrote_deribit_dvol_backfill={count}")
                return

    if args.command == "import":
        with RawEventWriter(raw_dir) as writer:
            if args.import_job == "bybit-history-trades":
                counts = import_bybit_history_trades(
                    args.input,
                    writer=writer,
                    symbol=args.symbol,
                    batch_size=args.batch_size,
                )
                print(" ".join(f"{name}={count}" for name, count in counts.items()))
                return
            if args.import_job == "bybit-history-orderbook":
                counts = import_bybit_history_orderbook(
                    args.input,
                    writer=writer,
                    symbol=args.symbol,
                    depth=args.depth,
                )
                print(" ".join(f"{name}={count}" for name, count in counts.items()))
                return

    if args.command == "materialize":
        derived_dir = data_dir / "derived"
        with PartitionedNdjsonWriter(derived_dir) as writer:
            if args.materializer == "bybit-bars":
                bars_by_kind = materialize_bybit_bars(args.input, symbol=args.symbol)
                count = 0
                for kind, bars in bars_by_kind.items():
                    for bar in bars:
                        writer.write(namespace=f"bars/bybit/{kind}", symbol=args.symbol, ts_ms=bar.ts, record=bar)
                        count += 1
                print(f"wrote_bars={count}")
                return
            if args.materializer == "micro-buckets":
                buckets = materialize_micro_buckets(args.input, symbol=args.symbol)
                for bucket in buckets:
                    writer.write(namespace="micro/bybit/1s", symbol=args.symbol, ts_ms=bucket.ts, record=bucket)
                print(f"wrote_micro_buckets={len(buckets)}")
                return

    if args.command == "features":
        derived_dir = data_dir / "derived"
        symbol = args.symbol
        raw_input = args.raw_input if args.raw_input else [raw_dir]
        trade_bars, index_bars, premium_bars, micro_buckets = _load_foundation_inputs(derived_dir, symbol)

        if args.feature_job == "bybit-foundation":
            trend = build_trend_features(trade_bars, symbol=symbol)
            regime = build_regime_features(trade_bars, index_bars, premium_bars, symbol=symbol)
            crowding = build_crowding_features(raw_input, premium_bars, micro_buckets, symbol=symbol)
            micro = build_micro_features(micro_buckets, symbol=symbol)
            with PartitionedNdjsonWriter(derived_dir) as writer:
                for item in trend:
                    writer.write(namespace="features/trend_bybit", symbol=symbol, ts_ms=item.ts, record=item)
                for item in regime:
                    writer.write(namespace="features/regime_bybit", symbol=symbol, ts_ms=item.ts, record=item)
                for item in crowding:
                    writer.write(namespace="features/crowding_bybit", symbol=symbol, ts_ms=item.ts, record=item)
                for item in micro:
                    writer.write(namespace="features/micro_bybit", symbol=symbol, ts_ms=item.ts, record=item)
            print(f"wrote_trend={len(trend)} wrote_regime={len(regime)} wrote_crowding={len(crowding)} wrote_micro={len(micro)}")
            return

        if args.feature_job == "options-deribit":
            items = build_options_deribit_features(raw_input, trade_bars, symbol=symbol)
            print(f"wrote_options_deribit={_write_feature_block(derived_dir, 'features/options_deribit', symbol, items)}")
            return
        if args.feature_job == "macro-veto":
            items = build_macro_veto_features(raw_input, trade_bars, symbol=symbol)
            print(f"wrote_macro_veto={_write_feature_block(derived_dir, 'features/macro_veto', symbol, items)}")
            return
        if args.feature_job == "glassnode-options":
            items = build_options_glassnode_features(raw_input, trade_bars, symbol=symbol)
            print(f"wrote_options_glassnode={_write_feature_block(derived_dir, 'features/options_glassnode', symbol, items)}")
            return
        if args.feature_job == "glassnode-onchain":
            items = build_onchain_glassnode_features(raw_input, trade_bars, symbol=symbol)
            print(f"wrote_onchain_glassnode={_write_feature_block(derived_dir, 'features/onchain_glassnode', symbol, items)}")
            return
        if args.feature_job == "cryptoquant-onchain":
            items = build_onchain_cryptoquant_features(raw_input, trade_bars, symbol=symbol)
            print(f"wrote_onchain_cryptoquant={_write_feature_block(derived_dir, 'features/onchain_cryptoquant', symbol, items)}")
            return
        if args.feature_job == "coinglass-overlay":
            items = build_aggregate_derivs_coinglass_features(raw_input, trade_bars, symbol=symbol)
            print(f"wrote_aggregate_derivs_coinglass={_write_feature_block(derived_dir, 'features/aggregate_derivs_coinglass', symbol, items)}")
            return

    if args.command == "signals":
        derived_dir = data_dir / "derived"
        symbol = args.symbol
        trade_bars, _, _, _ = _load_foundation_inputs(derived_dir, symbol)
        trend = load_feature_models([derived_dir / "features" / "trend_bybit" / symbol], TrendFeatureSnapshot)
        regime = load_feature_models([derived_dir / "features" / "regime_bybit" / symbol], RegimeFeatureSnapshot)
        crowding = load_feature_models([derived_dir / "features" / "crowding_bybit" / symbol], CrowdingFeatureSnapshot)
        micro = load_feature_models([derived_dir / "features" / "micro_bybit" / symbol], MicroFeatureSnapshot)
        macro_path = derived_dir / "features" / "macro_veto" / symbol
        macro = load_feature_models([macro_path], MacroVetoFeatureSnapshot) if macro_path.exists() else []
        continuation = build_continuation_candidates(
            trade_bars,
            trend,
            regime,
            crowding,
            micro,
            symbol=symbol,
            macro_features=macro,
            require_macro_veto=bool(args.enable_macro_veto and macro),
            sides=args.sides,
        )
        reversal = build_stress_reversal_candidates(
            trade_bars,
            regime,
            crowding,
            micro,
            symbol=symbol,
            macro_features=macro,
            require_macro_veto=bool(args.enable_macro_veto and macro),
            sides=args.sides,
        )
        with PartitionedNdjsonWriter(derived_dir) as writer:
            for item in continuation:
                writer.write(namespace="candidates/continuation_v1", symbol=symbol, ts_ms=item.ts, record=item)
            for item in reversal:
                writer.write(namespace="candidates/stress_reversal_v0", symbol=symbol, ts_ms=item.ts, record=item)
        print(f"wrote_continuation={len(continuation)} wrote_reversal={len(reversal)}")
        return

    if args.command == "research":
        if args.research_job == "label-candidates":
            derived_dir = data_dir / "derived"
            symbol = args.symbol
            trade_bars, _, _, micro_buckets = _load_foundation_inputs(derived_dir, symbol)
            continuation = load_feature_models([derived_dir / "candidates" / "continuation_v1" / symbol], CandidateEvent)
            reversal = load_feature_models([derived_dir / "candidates" / "stress_reversal_v0" / symbol], CandidateEvent)
            raw_paths = [raw_dir] if raw_dir.exists() else None
            labels = label_candidates(
                continuation + reversal,
                trade_bars,
                micro_buckets=micro_buckets,
                raw_paths=raw_paths,
                slippage_bps=args.slippage_bps,
                latency_ms=args.latency_ms,
            )
            with PartitionedNdjsonWriter(derived_dir) as writer:
                for item in labels:
                    writer.write(namespace="labels/bybit", symbol=symbol, ts_ms=item.ts, record=item)
            print(f"wrote_labels={len(labels)}")
            return

        if args.research_job == "walkforward":
            reports_dir = run_walkforward_experiments(
                data_dir=data_dir,
                registry_path=args.registry,
                symbol=args.symbol,
                output_dir=args.output_dir,
                experiment_ids=args.experiments,
                model_keys=args.models,
                skip_missing=args.skip_missing,
                slippage_bps=args.slippage_bps,
            )
            print(f"reports_dir={reports_dir}")
            return

    if args.command == "verify" and args.verify_job == "project":
        output_dir = Path(args.output_dir) if args.output_dir else default_reports_root(data_dir) / "verification"
        report = generate_verification_report(
            data_dir=data_dir,
            registry_path=args.registry,
            contracts_path=args.contracts,
            symbol=args.symbol,
            slippage_bps=args.slippage_bps,
        )
        artifacts_dir = write_verification_artifacts(report, output_dir)
        print(f"verification_dir={artifacts_dir} errors={len(report.errors)} warnings={len(report.warnings)}")
        if args.strict and report.errors:
            raise SystemExit(1)
        return

    with RawEventWriter(raw_dir) as writer:
        if args.command == "collect":
            if args.collector == "bybit-orderbook":
                collector = BybitOrderBookCollector(
                    symbol=args.symbol,
                    depth=args.depth,
                    category=args.category,
                    testnet=args.testnet or settings.bybit_testnet,
                    writer=writer,
                )
            elif args.collector == "bybit-trades":
                collector = BybitTradeCollector(
                    symbol=args.symbol,
                    category=args.category,
                    testnet=args.testnet or settings.bybit_testnet,
                    writer=writer,
                )
            elif args.collector == "bybit-liquidations":
                collector = BybitLiquidationCollector(
                    symbol=args.symbol,
                    category=args.category,
                    testnet=args.testnet or settings.bybit_testnet,
                    writer=writer,
                )
            elif args.collector == "bybit-rest":
                collector = BybitMarketRestPoller(
                    symbol=args.symbol,
                    category=args.category,
                    testnet=args.testnet or settings.bybit_testnet,
                    writer=writer,
                )
            elif args.collector == "deribit-dvol":
                collector = DeribitVolatilityIndexPoller(
                    currency=args.currency,
                    resolution=args.resolution,
                    interval_s=args.interval_s,
                    testnet=args.testnet or settings.deribit_testnet,
                    writer=writer,
                )
            else:  # pragma: no cover - parser should prevent this
                raise ValueError(args.collector)
            await collector.run()
            return

        if args.command == "replay" and args.mode == "top-of-book":
            states = replay_top_of_book([args.input], symbol=args.symbol)
            for state in states[-10:]:
                print(state.model_dump_json())
            digest = deterministic_replay_hash([args.input], symbol=args.symbol)
            print(f"states={len(states)} sha256={digest}")
            return

        if args.command == "import-macro-csv":
            ingestor = MacroCsvIngestor(writer=writer)
            ingestor.ingest(args.csv)
            return


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = RuntimeSettings.from_env()
    configure_logging(settings.log_level)
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
