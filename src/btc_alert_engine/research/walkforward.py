from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from btc_alert_engine.config import default_reports_root, load_research_registry
from btc_alert_engine.provenance import report_provenance
from btc_alert_engine.research.execution import build_raw_execution_tape
from btc_alert_engine.research.experiments import build_experiment_event_frame, feature_columns_for_experiment

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

EPS = 1e-6

RESULT_COLUMNS = [
    "experiment_id",
    "generator",
    "model_key",
    "split_kind",
    "fold_id",
    "budget_per_week",
    "threshold",
    "n_total",
    "n_alerts",
    "alerts_per_week",
    "brier_score",
    "calibration_slope",
    "calibration_intercept",
    "precision_tp_before_sl",
    "expectancy_r_per_alert",
    "hit_rate",
    "profit_factor",
    "max_drawdown_r",
    "worst_20_trade_drawdown_r",
    "median_hold_minutes",
    "mean_probability_alerts",
    "event_n_alerts",
    "event_alerts_per_week",
    "event_precision_tp_before_sl",
    "event_expectancy_r_per_alert",
    "event_hit_rate",
    "event_profit_factor",
    "event_max_drawdown_r",
    "event_worst_20_trade_drawdown_r",
    "event_median_hold_minutes",
    "event_mean_probability_alerts",
    "event_max_concurrent_positions",
    "portfolio_selection_policy",
    "overlap_filtered_alerts",
    "primary_metric",
    "test_start",
    "test_end",
]
SUMMARY_GROUP_COLUMNS = ["experiment_id", "generator", "model_key", "split_kind", "budget_per_week"]
SUMMARY_NUMERIC_COLUMNS = [
    "threshold",
    "n_total",
    "n_alerts",
    "alerts_per_week",
    "brier_score",
    "calibration_slope",
    "calibration_intercept",
    "precision_tp_before_sl",
    "expectancy_r_per_alert",
    "hit_rate",
    "profit_factor",
    "max_drawdown_r",
    "worst_20_trade_drawdown_r",
    "median_hold_minutes",
    "mean_probability_alerts",
    "event_n_alerts",
    "event_alerts_per_week",
    "event_precision_tp_before_sl",
    "event_expectancy_r_per_alert",
    "event_hit_rate",
    "event_profit_factor",
    "event_max_drawdown_r",
    "event_worst_20_trade_drawdown_r",
    "event_median_hold_minutes",
    "event_mean_probability_alerts",
    "event_max_concurrent_positions",
    "overlap_filtered_alerts",
    "primary_metric",
]


@dataclass(slots=True)
class SplitSpec:
    split_kind: str
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibrate_start: pd.Timestamp
    calibrate_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(slots=True)
class TrainedCalibratedModel:
    model_key: str
    feature_columns: list[str]
    estimator: Any
    calibrator_kind: str
    calibrator: Any | None
    constant_probability: float | None = None
    dropped_feature_columns: list[str] | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if len(frame) == 0:
            return np.array([], dtype=float)
        if self.constant_probability is not None:
            return np.full(len(frame), self.constant_probability, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*encountered in matmul.*",
                category=RuntimeWarning,
                module=r"sklearn\..*",
            )
            raw = self.estimator.predict_proba(frame[self.feature_columns])[:, 1]
        if not np.isfinite(raw).all():
            raise ValueError(f"Non-finite raw predicted probabilities for model {self.model_key}")
        raw = np.clip(raw, EPS, 1.0 - EPS)
        if self.calibrator is None:
            return raw
        if self.calibrator_kind == "isotonic":
            return np.clip(np.asarray(self.calibrator.predict(raw), dtype=float), EPS, 1.0 - EPS)
        if self.calibrator_kind == "sigmoid":
            logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*encountered in matmul.*",
                    category=RuntimeWarning,
                    module=r"sklearn\..*",
                )
                calibrated = self.calibrator.predict_proba(logits)[:, 1]
            if not np.isfinite(calibrated).all():
                raise ValueError(f"Non-finite calibrated probabilities for model {self.model_key}")
            return np.clip(calibrated, EPS, 1.0 - EPS)
        return raw


@dataclass(slots=True)
class EvaluationResult:
    experiment_id: str
    generator: str
    model_key: str
    split_kind: str
    fold_id: str
    budget_per_week: int
    threshold: float
    n_total: int
    n_alerts: int
    alerts_per_week: float
    brier_score: float | None
    calibration_slope: float | None
    calibration_intercept: float | None
    precision_tp_before_sl: float | None
    expectancy_r_per_alert: float | None
    hit_rate: float | None
    profit_factor: float | None
    max_drawdown_r: float | None
    worst_20_trade_drawdown_r: float | None
    median_hold_minutes: float | None
    mean_probability_alerts: float | None
    event_n_alerts: int
    event_alerts_per_week: float
    event_precision_tp_before_sl: float | None
    event_expectancy_r_per_alert: float | None
    event_hit_rate: float | None
    event_profit_factor: float | None
    event_max_drawdown_r: float | None
    event_worst_20_trade_drawdown_r: float | None
    event_median_hold_minutes: float | None
    event_mean_probability_alerts: float | None
    event_max_concurrent_positions: int
    portfolio_selection_policy: str
    overlap_filtered_alerts: int
    primary_metric: float | None
    test_start: str
    test_end: str


@dataclass(slots=True)
class ExperimentRunBundle:
    experiment_id: str
    generator: str
    blocks: list[str]
    dataset_rows: int
    skip_reason: str | None
    outer_results: list[EvaluationResult]
    final_results: list[EvaluationResult]
    selected_model: str | None
    candidate_side_counts: dict[str, int] | None = None
    label_side_counts: dict[str, int] | None = None


def _ensure_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _calendar_days(days: int) -> pd.Timedelta:
    return pd.Timedelta(days=int(days))


def make_outer_splits(frame: pd.DataFrame, validation: dict[str, Any]) -> list[SplitSpec]:
    if frame.empty:
        return []
    warmup_end = _ensure_utc(validation["warmup_only"]["end"]) + pd.Timedelta(days=1)
    train_days = int(validation["outer_walk_forward"]["train_days"])
    calibrate_days = int(validation["outer_walk_forward"]["calibrate_days"])
    test_days = int(validation["outer_walk_forward"]["test_days"])
    purge = pd.Timedelta(hours=int(validation["outer_walk_forward"]["purge_hours"]))
    embargo = pd.Timedelta(hours=int(validation["outer_walk_forward"]["embargo_hours"]))

    available_start = frame["ts_dt"].min().floor("D")
    available_end = frame["ts_dt"].max().ceil("D")
    first_test_start = max(available_start, warmup_end) + _calendar_days(train_days) + purge + _calendar_days(calibrate_days) + embargo

    splits: list[SplitSpec] = []
    cursor = first_test_start
    fold_idx = 1
    while cursor + _calendar_days(test_days) <= available_end + pd.Timedelta(seconds=1):
        test_start = cursor
        test_end = cursor + _calendar_days(test_days)
        calibrate_end = test_start - embargo
        calibrate_start = calibrate_end - _calendar_days(calibrate_days)
        train_end = calibrate_start - purge
        train_start = train_end - _calendar_days(train_days)
        splits.append(
            SplitSpec(
                split_kind="outer",
                fold_id=f"outer_{fold_idx:02d}",
                train_start=train_start,
                train_end=train_end,
                calibrate_start=calibrate_start,
                calibrate_end=calibrate_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor = cursor + _calendar_days(test_days)
        fold_idx += 1
    return splits


def make_final_split(validation: dict[str, Any]) -> SplitSpec:
    freeze_date = _ensure_utc(validation["final_frozen_test"]["freeze_date"]) + pd.Timedelta(days=1)
    calibrate_start = _ensure_utc(validation["final_frozen_test"]["calibrate_window"]["start"])
    calibrate_end = _ensure_utc(validation["final_frozen_test"]["calibrate_window"]["end"]) + pd.Timedelta(days=1)
    test_start = _ensure_utc(validation["final_frozen_test"]["untouched_test"]["start"])
    test_end = _ensure_utc(validation["final_frozen_test"]["untouched_test"]["end"]) + pd.Timedelta(days=1)
    train_end = min(freeze_date, calibrate_start)
    train_start = _ensure_utc(validation["warmup_only"]["start"])
    return SplitSpec(
        split_kind="final",
        fold_id="final_frozen",
        train_start=train_start,
        train_end=train_end,
        calibrate_start=calibrate_start,
        calibrate_end=calibrate_end,
        test_start=test_start,
        test_end=test_end,
    )


def _slice_split(frame: pd.DataFrame, split: SplitSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[(frame["ts_dt"] >= split.train_start) & (frame["ts_dt"] < split.train_end)].copy()
    calibrate = frame[(frame["ts_dt"] >= split.calibrate_start) & (frame["ts_dt"] < split.calibrate_end)].copy()
    test = frame[(frame["ts_dt"] >= split.test_start) & (frame["ts_dt"] < split.test_end)].copy()
    return train, calibrate, test


def _build_estimator(model_key: str) -> Any:
    if model_key == "baseline":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear")),
            ]
        )
    if model_key == "challenger":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        num_leaves=31,
                        max_depth=-1,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
    raise ValueError(model_key)


def _fit_sigmoid_calibrator(raw_proba: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y)) < 2:
        return None
    clipped = np.clip(raw_proba, EPS, 1.0 - EPS)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul.*",
            category=RuntimeWarning,
            module=r"sklearn\..*",
        )
        model.fit(logits, y)
    return model


def _fit_calibrator(raw_proba: np.ndarray, y: np.ndarray, order: Iterable[str]) -> tuple[str, Any | None]:
    unique_y = np.unique(y)
    if len(unique_y) < 2:
        return ("none", None)
    unique_scores = np.unique(np.round(raw_proba, 8))
    for method in order:
        if method == "isotonic":
            if len(unique_scores) < 8:
                continue
            try:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(raw_proba, y)
                return ("isotonic", model)
            except Exception:
                continue
        if method == "sigmoid":
            model = _fit_sigmoid_calibrator(raw_proba, y)
            if model is not None:
                return ("sigmoid", model)
    return ("none", None)


def _prune_feature_columns(train: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for column in feature_columns:
        if column not in train.columns:
            dropped.append(column)
            continue
        non_null = train[column].dropna()
        if non_null.empty:
            dropped.append(column)
            continue
        if non_null.nunique(dropna=True) <= 1:
            dropped.append(column)
            continue
        kept.append(column)
    return kept, dropped


def train_calibrated_model(
    train: pd.DataFrame,
    calibrate: pd.DataFrame,
    feature_columns: list[str],
    *,
    model_key: str,
    calibration_order: Iterable[str],
) -> TrainedCalibratedModel:
    kept_features, dropped_features = _prune_feature_columns(train, feature_columns)
    if not kept_features:
        base_rate = float(train["y"].mean()) if len(train) else 0.5
        return TrainedCalibratedModel(
            model_key,
            kept_features,
            None,
            "none",
            None,
            constant_probability=base_rate,
            dropped_feature_columns=dropped_features,
        )

    y_train = train["y"].astype(int).to_numpy()
    y_cal = calibrate["y"].astype(int).to_numpy()
    if len(train) < 20 or len(np.unique(y_train)) < 2:
        base_rate = float(y_train.mean()) if len(y_train) else 0.5
        return TrainedCalibratedModel(
            model_key,
            kept_features,
            None,
            "none",
            None,
            constant_probability=base_rate,
            dropped_feature_columns=dropped_features,
        )

    estimator = _build_estimator(model_key)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul.*",
            category=RuntimeWarning,
            module=r"sklearn\..*",
        )
        estimator.fit(train[kept_features], y_train)
    if len(calibrate) == 0:
        return TrainedCalibratedModel(
            model_key,
            kept_features,
            estimator,
            "none",
            None,
            dropped_feature_columns=dropped_features,
        )

    raw = estimator.predict_proba(calibrate[kept_features])[:, 1]
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite calibration probabilities for model {model_key}")
    raw = np.clip(raw, EPS, 1.0 - EPS)
    calibrator_kind, calibrator = _fit_calibrator(raw, y_cal, calibration_order)
    return TrainedCalibratedModel(
        model_key,
        kept_features,
        estimator,
        calibrator_kind,
        calibrator,
        dropped_feature_columns=dropped_features,
    )


def _window_weeks(start: pd.Timestamp, end: pd.Timestamp) -> float:
    span_seconds = max((end - start).total_seconds(), 86_400.0)
    return max(span_seconds / 604_800.0, 1.0 / 7.0)


def _coalesce_entry_ts(frame: pd.DataFrame) -> pd.Series:
    if "entry_ts" in frame.columns:
        return frame["entry_ts"].fillna(frame.get("ts"))
    return frame["ts"]


def _coalesce_exit_ts(frame: pd.DataFrame) -> pd.Series:
    if "exit_ts" in frame.columns:
        return frame["exit_ts"].fillna(_coalesce_entry_ts(frame))
    return _coalesce_entry_ts(frame)


def _max_concurrent_positions(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "entry_ts" not in frame.columns or "exit_ts" not in frame.columns:
        return 1 if len(frame) else 0
    events: list[tuple[int, int]] = []
    for _, row in frame.iterrows():
        entry_ts = int(row["entry_ts"] if pd.notna(row.get("entry_ts")) else row["ts"])
        exit_ts = int(row["exit_ts"] if pd.notna(row.get("exit_ts")) else entry_ts)
        exit_ts = max(exit_ts, entry_ts)
        events.append((entry_ts, 1))
        events.append((exit_ts, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    concurrent = 0
    peak = 0
    for _, delta in events:
        concurrent += delta
        peak = max(peak, concurrent)
    return peak


def _apply_portfolio_policy(
    selected: pd.DataFrame,
    *,
    mode: str,
    selection_policy: str,
    cooldown_minutes: int,
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    if mode == "event_level":
        return selected.sort_values(["ts", "p"], ascending=[True, False]).reset_index(drop=True)
    if "entry_ts" not in selected.columns or "exit_ts" not in selected.columns:
        return selected.sort_values(["ts", "p"], ascending=[True, False]).reset_index(drop=True)

    cooldown_ms = max(int(cooldown_minutes), 0) * 60_000
    working = selected.copy()
    working["entry_ts"] = _coalesce_entry_ts(working).astype(int)
    working["exit_ts"] = _coalesce_exit_ts(working).astype(int)
    working["exit_ts"] = working[["entry_ts", "exit_ts"]].max(axis=1)

    if selection_policy == "highest_probability":
        ordered = working.sort_values(["entry_ts", "ts", "p"], ascending=[True, True, False]).reset_index(drop=True)
        chosen_rows: list[pd.Series] = []
        remaining = ordered
        while not remaining.empty:
            seed = remaining.iloc[0]
            seed_end = int(seed["exit_ts"]) + cooldown_ms
            cluster_frame = remaining[remaining["entry_ts"] < seed_end].copy()
            chosen = cluster_frame.sort_values(["p", "entry_ts", "ts"], ascending=[False, True, True]).iloc[0]
            chosen_rows.append(chosen)
            next_free_ts = int(chosen["exit_ts"]) + cooldown_ms
            remaining = remaining[remaining["entry_ts"] >= next_free_ts].reset_index(drop=True)
        return pd.DataFrame(chosen_rows).sort_values(["ts", "p"], ascending=[True, False]).reset_index(drop=True)

    ordered = working.sort_values(["entry_ts", "ts", "p"], ascending=[True, True, False]).reset_index(drop=True)
    taken: list[pd.Series] = []
    next_free_ts: int | None = None
    for _, row in ordered.iterrows():
        entry_ts = int(row["entry_ts"])
        if next_free_ts is not None and entry_ts < next_free_ts:
            continue
        taken.append(row)
        next_free_ts = int(row["exit_ts"]) + cooldown_ms
    return pd.DataFrame(taken).sort_values(["ts", "p"], ascending=[True, False]).reset_index(drop=True)


def _selection_metrics(selected: pd.DataFrame, *, weeks: float) -> dict[str, float | int | None]:
    if selected.empty:
        return {
            "n_alerts": 0,
            "alerts_per_week": 0.0,
            "precision_tp_before_sl": None,
            "expectancy_r_per_alert": None,
            "hit_rate": None,
            "profit_factor": None,
            "max_drawdown_r": None,
            "worst_20_trade_drawdown_r": None,
            "median_hold_minutes": None,
            "mean_probability_alerts": None,
        }
    y_selected = selected["y"].astype(int).to_numpy()
    return {
        "n_alerts": int(len(selected)),
        "alerts_per_week": float(len(selected) / weeks) if weeks > 0 else 0.0,
        "precision_tp_before_sl": float(y_selected.mean()) if len(y_selected) else None,
        "expectancy_r_per_alert": float(selected["realized_r"].mean()),
        "hit_rate": float((selected["outcome"] == "tp").mean()),
        "profit_factor": _profit_factor(selected["realized_r"]),
        "max_drawdown_r": _max_drawdown(selected["realized_r"]),
        "worst_20_trade_drawdown_r": _worst_20_trade_drawdown(selected["realized_r"]),
        "median_hold_minutes": float(selected["holding_minutes"].median()) if "holding_minutes" in selected.columns else None,
        "mean_probability_alerts": float(selected["p"].mean()) if "p" in selected.columns else None,
    }


def threshold_for_budget(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    budget_per_week: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    portfolio_mode: str,
    selection_policy: str,
    cooldown_minutes: int,
) -> float:
    if len(probabilities) == 0:
        return 1.0
    working = frame.copy()
    working["p"] = np.asarray(probabilities, dtype=float)
    weeks = _window_weeks(window_start, window_end)
    target = max(1, int(round(weeks * budget_per_week)))
    thresholds = sorted({float(value) for value in np.asarray(probabilities, dtype=float)}, reverse=True)
    best_threshold = thresholds[0]
    best_key: tuple[float, int, int, float] | None = None
    for threshold in thresholds:
        selected = working[working["p"] >= threshold].sort_values("ts")
        portfolio_selected = _apply_portfolio_policy(
            selected,
            mode=portfolio_mode,
            selection_policy=selection_policy,
            cooldown_minutes=cooldown_minutes,
        )
        count = int(len(portfolio_selected))
        gap = abs(count - target)
        under_penalty = 0 if count <= target else 1
        size_penalty = -count if count <= target else count
        key = (gap, under_penalty, size_penalty, -threshold)
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def _profit_factor(realized_r: pd.Series) -> float | None:
    if realized_r.empty:
        return None
    gross_profit = realized_r[realized_r > 0].sum()
    gross_loss = -realized_r[realized_r < 0].sum()
    if gross_loss <= 0:
        return None if gross_profit <= 0 else float("inf")
    return float(gross_profit / gross_loss)


def _max_drawdown(realized_r: pd.Series) -> float | None:
    if realized_r.empty:
        return None
    equity = realized_r.cumsum()
    peak = equity.cummax()
    drawdown = peak - equity
    return float(drawdown.max())


def _worst_20_trade_drawdown(realized_r: pd.Series) -> float | None:
    if realized_r.empty:
        return None
    if len(realized_r) <= 20:
        return _max_drawdown(realized_r)
    worst = 0.0
    for start in range(0, len(realized_r) - 19):
        window = realized_r.iloc[start : start + 20].reset_index(drop=True)
        dd = _max_drawdown(window)
        if dd is not None:
            worst = max(worst, dd)
    return float(worst)


def _calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None, None
    clipped = np.clip(p, EPS, 1.0 - EPS)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*encountered in matmul.*",
                category=RuntimeWarning,
                module=r"sklearn\..*",
            )
            model.fit(logits, y)
        return float(model.coef_[0][0]), float(model.intercept_[0])
    except Exception:
        return None, None


def evaluate_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
    budget_per_week: int,
    experiment_id: str,
    generator: str,
    model_key: str,
    split: SplitSpec,
    portfolio_mode: str,
    selection_policy: str,
    cooldown_minutes: int,
) -> EvaluationResult:
    working = frame.copy()
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError(f"Non-finite predicted probabilities for experiment {experiment_id} model {model_key}")
    working["p"] = p
    event_selected = working[working["p"] >= threshold].sort_values("ts")
    portfolio_selected = _apply_portfolio_policy(
        event_selected,
        mode=portfolio_mode,
        selection_policy=selection_policy,
        cooldown_minutes=cooldown_minutes,
    )

    y_all = working["y"].astype(int).to_numpy()
    p_all = working["p"].to_numpy(dtype=float)
    slope, intercept = _calibration_slope_intercept(y_all, p_all)
    brier = brier_score_loss(y_all, p_all) if len(working) else None
    weeks = _window_weeks(split.test_start, split.test_end)

    event_metrics = _selection_metrics(event_selected, weeks=weeks)
    portfolio_metrics = _selection_metrics(portfolio_selected, weeks=weeks)
    event_max_concurrent = _max_concurrent_positions(event_selected)

    return EvaluationResult(
        experiment_id=experiment_id,
        generator=generator,
        model_key=model_key,
        split_kind=split.split_kind,
        fold_id=split.fold_id,
        budget_per_week=budget_per_week,
        threshold=float(threshold),
        n_total=int(len(working)),
        n_alerts=int(portfolio_metrics["n_alerts"]),
        alerts_per_week=float(portfolio_metrics["alerts_per_week"]),
        brier_score=None if brier is None or math.isnan(brier) else float(brier),
        calibration_slope=slope,
        calibration_intercept=intercept,
        precision_tp_before_sl=portfolio_metrics["precision_tp_before_sl"],
        expectancy_r_per_alert=portfolio_metrics["expectancy_r_per_alert"],
        hit_rate=portfolio_metrics["hit_rate"],
        profit_factor=portfolio_metrics["profit_factor"],
        max_drawdown_r=portfolio_metrics["max_drawdown_r"],
        worst_20_trade_drawdown_r=portfolio_metrics["worst_20_trade_drawdown_r"],
        median_hold_minutes=portfolio_metrics["median_hold_minutes"],
        mean_probability_alerts=portfolio_metrics["mean_probability_alerts"],
        event_n_alerts=int(event_metrics["n_alerts"]),
        event_alerts_per_week=float(event_metrics["alerts_per_week"]),
        event_precision_tp_before_sl=event_metrics["precision_tp_before_sl"],
        event_expectancy_r_per_alert=event_metrics["expectancy_r_per_alert"],
        event_hit_rate=event_metrics["hit_rate"],
        event_profit_factor=event_metrics["profit_factor"],
        event_max_drawdown_r=event_metrics["max_drawdown_r"],
        event_worst_20_trade_drawdown_r=event_metrics["worst_20_trade_drawdown_r"],
        event_median_hold_minutes=event_metrics["median_hold_minutes"],
        event_mean_probability_alerts=event_metrics["mean_probability_alerts"],
        event_max_concurrent_positions=event_max_concurrent,
        portfolio_selection_policy=selection_policy,
        overlap_filtered_alerts=int(event_metrics["n_alerts"] - portfolio_metrics["n_alerts"]),
        primary_metric=portfolio_metrics["expectancy_r_per_alert"],
        test_start=split.test_start.isoformat(),
        test_end=split.test_end.isoformat(),
    )


def run_split(
    frame: pd.DataFrame,
    split: SplitSpec,
    *,
    feature_columns: list[str],
    model_key: str,
    calibration_order: Iterable[str],
    budgets: list[int],
    experiment_id: str,
    generator: str,
    portfolio_mode: str,
    selection_policy: str,
    cooldown_minutes: int,
) -> list[EvaluationResult]:
    train, calibrate, test = _slice_split(frame, split)
    if len(train) < 20 or len(calibrate) < 5 or len(test) < 5:
        return []
    model = train_calibrated_model(train, calibrate, feature_columns, model_key=model_key, calibration_order=calibration_order)
    p_cal = model.predict_proba(calibrate)
    p_test = model.predict_proba(test)
    results: list[EvaluationResult] = []
    for budget in budgets:
        threshold = threshold_for_budget(
            calibrate,
            p_cal,
            budget_per_week=budget,
            window_start=split.calibrate_start,
            window_end=split.calibrate_end,
            portfolio_mode=portfolio_mode,
            selection_policy=selection_policy,
            cooldown_minutes=cooldown_minutes,
        )
        results.append(
            evaluate_predictions(
                test,
                p_test,
                threshold=threshold,
                budget_per_week=budget,
                experiment_id=experiment_id,
                generator=generator,
                model_key=model_key,
                split=split,
                portfolio_mode=portfolio_mode,
                selection_policy=selection_policy,
                cooldown_minutes=cooldown_minutes,
            )
        )
    return results


def _result_frame(results: list[EvaluationResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame([asdict(item) for item in results])


def _summarize_results(results: list[EvaluationResult]) -> pd.DataFrame:
    frame = _result_frame(results)
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_GROUP_COLUMNS + SUMMARY_NUMERIC_COLUMNS)
    summary = (
        frame.groupby(SUMMARY_GROUP_COLUMNS, dropna=False)[SUMMARY_NUMERIC_COLUMNS]
        .mean(numeric_only=True)
        .reset_index()
    )
    return summary


def _primary_budget(goal: dict[str, Any], promotion_rules: dict[str, Any] | None) -> int:
    if promotion_rules and "primary_budget_per_week" in promotion_rules:
        return int(promotion_rules["primary_budget_per_week"])
    return int(goal.get("production_budget_per_week", 3))


def _portfolio_settings(goal: dict[str, Any]) -> tuple[str, str, int]:
    config = goal.get("portfolio_evaluation", {}) or {}
    mode = str(config.get("mode", "one_position_at_a_time"))
    selection_policy = str(config.get("selection_policy", "first_signal"))
    cooldown_minutes = int(config.get("cooldown_minutes", 0))
    if mode not in {"one_position_at_a_time", "event_level"}:
        mode = "one_position_at_a_time"
    if selection_policy not in {"first_signal", "highest_probability"}:
        selection_policy = "first_signal"
    return mode, selection_policy, cooldown_minutes


def _best_model_for_experiment(summary: pd.DataFrame, experiment_id: str, primary_budget: int) -> str | None:
    required = {"experiment_id", "split_kind", "budget_per_week", "primary_metric", "max_drawdown_r", "model_key"}
    if summary.empty or not required.issubset(summary.columns):
        return None
    exp = summary[
        (summary["experiment_id"] == experiment_id)
        & (summary["split_kind"] == "outer")
        & (summary["budget_per_week"] == primary_budget)
    ]
    if exp.empty:
        return None
    exp = exp.sort_values(["primary_metric", "max_drawdown_r"], ascending=[False, True], na_position="last")
    return str(exp.iloc[0]["model_key"])


def _find_parent(experiments: list[dict[str, Any]], experiment_id: str) -> str | None:
    ordered = [dict(item) for item in experiments]
    current_index = next((i for i, item in enumerate(ordered) if item["id"] == experiment_id), None)
    if current_index is None:
        return None
    current = ordered[current_index]
    current_blocks = set(current.get("blocks", []))
    current_generator = current.get("generator")
    candidates: list[tuple[int, str]] = []
    for item in ordered[:current_index]:
        if item.get("generator") != current_generator:
            continue
        blocks = set(item.get("blocks", []))
        if blocks < current_blocks:
            candidates.append((len(blocks), str(item["id"])))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][1]


def _fold_improvement_fraction(child: pd.DataFrame, parent: pd.DataFrame) -> float | None:
    if child.empty or parent.empty:
        return None
    merged = child.merge(parent, on=["fold_id", "budget_per_week"], suffixes=("_child", "_parent"))
    if merged.empty:
        return None
    improved = (merged["expectancy_r_per_alert_child"] > merged["expectancy_r_per_alert_parent"]).mean()
    return float(improved)


def _break_segment(ts_start: str) -> str:
    ts = _ensure_utc(ts_start)
    return "pre_2023_10_23" if ts < pd.Timestamp("2023-10-23", tz="UTC") else "post_2023_10_23"


def _promotion_report(
    registry: Any,
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    selected_models: dict[str, str | None],
) -> pd.DataFrame:
    primary_budget = _primary_budget(registry.goal, registry.__dict__.get("promotion_rules") or registry.model_dump().get("promotion_rules"))
    promotion_rules = registry.model_dump().get("promotion_rules", {})
    rows: list[dict[str, Any]] = []
    for experiment in registry.experiments:
        experiment_id = str(experiment["id"])
        selected_model = selected_models.get(experiment_id)
        if not selected_model:
            rows.append({"experiment_id": experiment_id, "selected_model": None, "parent_experiment": None, "promote": False, "reason": "no_model_selected"})
            continue
        if summary.empty:
            rows.append({"experiment_id": experiment_id, "selected_model": selected_model, "parent_experiment": None, "promote": False, "reason": "missing_final_metrics"})
            continue
        final_row = summary[
            (summary["experiment_id"] == experiment_id)
            & (summary["split_kind"] == "final")
            & (summary["budget_per_week"] == primary_budget)
            & (summary["model_key"] == selected_model)
        ]
        if final_row.empty:
            rows.append({"experiment_id": experiment_id, "selected_model": selected_model, "parent_experiment": None, "promote": False, "reason": "missing_final_metrics"})
            continue
        final_metric = float(final_row.iloc[0]["expectancy_r_per_alert"]) if pd.notna(final_row.iloc[0]["expectancy_r_per_alert"]) else None
        final_dd = float(final_row.iloc[0]["worst_20_trade_drawdown_r"]) if pd.notna(final_row.iloc[0]["worst_20_trade_drawdown_r"]) else None
        parent_id = _find_parent(registry.experiments, experiment_id)
        if parent_id is None:
            rows.append({
                "experiment_id": experiment_id,
                "selected_model": selected_model,
                "parent_experiment": None,
                "promote": True,
                "reason": "root_experiment",
                "final_expectancy_r_per_alert": final_metric,
                "final_worst_20_trade_drawdown_r": final_dd,
            })
            continue
        parent_model = selected_models.get(parent_id)
        parent_final = summary[
            (summary["experiment_id"] == parent_id)
            & (summary["split_kind"] == "final")
            & (summary["budget_per_week"] == primary_budget)
            & (summary["model_key"] == parent_model)
        ]
        if parent_final.empty:
            rows.append({"experiment_id": experiment_id, "selected_model": selected_model, "parent_experiment": parent_id, "promote": False, "reason": "missing_parent_final_metrics"})
            continue
        parent_metric = float(parent_final.iloc[0]["expectancy_r_per_alert"]) if pd.notna(parent_final.iloc[0]["expectancy_r_per_alert"]) else None
        parent_dd = float(parent_final.iloc[0]["worst_20_trade_drawdown_r"]) if pd.notna(parent_final.iloc[0]["worst_20_trade_drawdown_r"]) else None

        child_outer = fold_metrics[
            (fold_metrics["experiment_id"] == experiment_id)
            & (fold_metrics["split_kind"] == "outer")
            & (fold_metrics["budget_per_week"] == primary_budget)
            & (fold_metrics["model_key"] == selected_model)
        ]
        parent_outer = fold_metrics[
            (fold_metrics["experiment_id"] == parent_id)
            & (fold_metrics["split_kind"] == "outer")
            & (fold_metrics["budget_per_week"] == primary_budget)
            & (fold_metrics["model_key"] == parent_model)
        ]
        positive_sign_fraction = _fold_improvement_fraction(child_outer, parent_outer)

        improve_req = float(promotion_rules.get("improve_parent_expectancy_by_at_least_r", 0.10))
        near_equal_tol = float(promotion_rules.get("near_equal_expectancy_tolerance_r", 0.02))
        dd_improve_req = float(promotion_rules.get("drawdown_improvement_required_percent", 20.0))
        sign_req = float(promotion_rules.get("positive_sign_required_outer_folds_percent", 70.0)) / 100.0
        risk_tail_req = float(promotion_rules.get("risk_block_promotion", {}).get("keep_if_expectancy_flat_and_tail_risk_improves_percentile_mae_or_w20dd_by", 25.0))

        expectancy_delta = None if final_metric is None or parent_metric is None else final_metric - parent_metric
        dd_improvement_pct = None
        if final_dd is not None and parent_dd not in (None, 0):
            dd_improvement_pct = (parent_dd - final_dd) / parent_dd * 100.0

        child_segments = child_outer.copy()
        if not child_segments.empty:
            child_segments["segment"] = child_segments["test_start"].map(_break_segment)
            seg_means = child_segments.groupby("segment")["expectancy_r_per_alert"].mean().to_dict()
            break_consistent = not (
                seg_means.get("pre_2023_10_23") is not None
                and seg_means.get("post_2023_10_23") is not None
                and seg_means["pre_2023_10_23"] > 0
                and seg_means["post_2023_10_23"] < 0
            )
        else:
            break_consistent = True

        added_blocks = set(experiment.get("blocks", [])) - set(next(item for item in registry.experiments if item["id"] == parent_id).get("blocks", []))
        is_risk_block_child = bool(added_blocks & {"options_deribit", "options_glassnode", "macro_veto"})

        promote = False
        reason = "did_not_meet_promotion_rules"
        if expectancy_delta is not None and expectancy_delta >= improve_req and (positive_sign_fraction or 0.0) >= sign_req and break_consistent:
            promote = True
            reason = "expectancy_improved"
        elif (
            expectancy_delta is not None
            and abs(expectancy_delta) <= near_equal_tol
            and dd_improvement_pct is not None
            and dd_improvement_pct >= dd_improve_req
            and (positive_sign_fraction or 0.0) >= sign_req
            and break_consistent
        ):
            promote = True
            reason = "drawdown_improved_with_flat_expectancy"
        elif (
            is_risk_block_child
            and expectancy_delta is not None
            and abs(expectancy_delta) <= near_equal_tol
            and dd_improvement_pct is not None
            and dd_improvement_pct >= risk_tail_req
            and break_consistent
        ):
            promote = True
            reason = "risk_block_tail_improvement"

        rows.append(
            {
                "experiment_id": experiment_id,
                "selected_model": selected_model,
                "parent_experiment": parent_id,
                "final_expectancy_r_per_alert": final_metric,
                "parent_final_expectancy_r_per_alert": parent_metric,
                "expectancy_delta_r": expectancy_delta,
                "final_worst_20_trade_drawdown_r": final_dd,
                "parent_final_worst_20_trade_drawdown_r": parent_dd,
                "drawdown_improvement_pct": dd_improvement_pct,
                "positive_sign_fraction": positive_sign_fraction,
                "break_consistent": break_consistent,
                "promote": promote,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _markdown_report(
    registry: Any,
    summary: pd.DataFrame,
    promotion: pd.DataFrame,
    selected_models: dict[str, str | None],
) -> str:
    primary_budget = _primary_budget(registry.goal, registry.model_dump().get("promotion_rules", {}))
    portfolio_mode, selection_policy, cooldown_minutes = _portfolio_settings(registry.goal)
    lines = [
        f"# Walk-forward promotion report: {registry.program['id']}",
        "",
        f"Primary budget: **{primary_budget} alerts/week**",
        f"Portfolio evaluation: **{portfolio_mode} / {selection_policy}**, cooldown **{cooldown_minutes} min**",
        "",
        "## Selected models",
        "",
    ]
    for experiment in registry.experiments:
        exp_id = str(experiment["id"])
        lines.append(f"- `{exp_id}` → `{selected_models.get(exp_id)}`")
    lines.extend(["", "## Final split summary", ""])
    if summary.empty or not {"split_kind", "budget_per_week", "experiment_id", "model_key"}.issubset(summary.columns):
        final_rows = pd.DataFrame()
    else:
        final_rows = summary[(summary["split_kind"] == "final") & (summary["budget_per_week"] == primary_budget)].sort_values(["experiment_id", "model_key"])
    if final_rows.empty:
        lines.append("No final metrics were produced.")
    else:
        lines.append("| experiment | model | portfolio expectancy R/trade | portfolio alerts/week | portfolio precision | event alerts/week | event precision | worst 20-trade DD |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in final_rows.iterrows():
            lines.append(
                "| {experiment_id} | {model_key} | {expectancy:.4f} | {apw:.2f} | {precision:.3f} | {event_apw:.2f} | {event_precision:.3f} | {w20:.3f} |".format(
                    experiment_id=row["experiment_id"],
                    model_key=row["model_key"],
                    expectancy=row["expectancy_r_per_alert"] if pd.notna(row["expectancy_r_per_alert"]) else float("nan"),
                    apw=row["alerts_per_week"] if pd.notna(row["alerts_per_week"]) else 0.0,
                    precision=row["precision_tp_before_sl"] if pd.notna(row["precision_tp_before_sl"]) else float("nan"),
                    event_apw=row.get("event_alerts_per_week") if pd.notna(row.get("event_alerts_per_week")) else 0.0,
                    event_precision=row.get("event_precision_tp_before_sl") if pd.notna(row.get("event_precision_tp_before_sl")) else float("nan"),
                    w20=row["worst_20_trade_drawdown_r"] if pd.notna(row["worst_20_trade_drawdown_r"]) else float("nan"),
                )
            )
    lines.extend(["", "## Promotion decisions", ""])
    if promotion.empty:
        lines.append("No promotion decisions were produced.")
    else:
        lines.append("| experiment | selected model | parent | promote | reason | Δ expectancy | DD improvement % |")
        lines.append("|---|---|---|---:|---|---:|---:|")
        for _, row in promotion.iterrows():
            lines.append(
                "| {experiment_id} | {selected_model} | {parent} | {promote} | {reason} | {delta:.4f} | {ddi:.1f} |".format(
                    experiment_id=row["experiment_id"],
                    selected_model=row.get("selected_model") or "-",
                    parent=row.get("parent_experiment") or "-",
                    promote="yes" if bool(row.get("promote", False)) else "no",
                    reason=row.get("reason") or "-",
                    delta=row.get("expectancy_delta_r") if pd.notna(row.get("expectancy_delta_r")) else 0.0,
                    ddi=row.get("drawdown_improvement_pct") if pd.notna(row.get("drawdown_improvement_pct")) else 0.0,
                )
            )
    return "\n".join(lines) + "\n"


def run_walkforward_experiments(
    *,
    data_dir: str | Path,
    registry_path: str | Path,
    symbol: str = "BTCUSDT",
    output_dir: str | Path | None = None,
    experiment_ids: list[str] | None = None,
    model_keys: list[str] | None = None,
    skip_missing: bool = True,
    slippage_bps: float = 1.0,
) -> Path:
    registry = load_research_registry(registry_path)
    data_root = Path(data_dir)
    derived_dir = data_root / "derived"
    raw_dir = data_root / "raw"
    raw_paths = [raw_dir] if raw_dir.exists() else None
    raw_tape = None
    if raw_paths:
        raw_tape, _ = build_raw_execution_tape(raw_paths, symbol=symbol, tolerate_gaps=True)
    latency_ms = int(registry.labeling.get("latency_ms", 0))
    reports_root = Path(output_dir) if output_dir is not None else default_reports_root(data_root) / "walkforward"
    reports_root.mkdir(parents=True, exist_ok=True)

    requested_models = model_keys or ["baseline", "challenger"]
    available_models: list[str] = []
    for key in requested_models:
        if key == "challenger" and LGBMClassifier is None:
            continue
        available_models.append(key)
    if not available_models:
        available_models = ["baseline"]

    budgets = list(registry.goal.get("alert_budgets_per_week", [1, 3, 7]))
    calibration_order = registry.models.get("calibration", {}).get("order", ["isotonic", "sigmoid"])
    portfolio_mode, selection_policy, cooldown_minutes = _portfolio_settings(registry.goal)

    selected_experiments = [exp for exp in registry.experiments if experiment_ids is None or exp["id"] in experiment_ids]

    bundles: list[ExperimentRunBundle] = []
    all_results: list[EvaluationResult] = []

    for experiment in selected_experiments:
        dataset = build_experiment_event_frame(
            derived_dir,
            symbol,
            experiment,
            raw_paths=raw_paths,
            raw_tape=raw_tape,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            skip_missing=skip_missing,
        )
        if dataset.skip_reason or dataset.frame.empty:
            bundles.append(ExperimentRunBundle(dataset.experiment_id, dataset.generator, dataset.blocks, len(dataset.frame), dataset.skip_reason or "empty_frame", [], [], None, {}, {}))
            continue
        feature_columns = feature_columns_for_experiment(dataset.frame, dataset.blocks)
        outer_splits = make_outer_splits(dataset.frame, registry.validation)
        final_split = make_final_split(registry.validation)

        outer_results: list[EvaluationResult] = []
        final_results: list[EvaluationResult] = []
        for model_key in available_models:
            for split in outer_splits:
                outer_results.extend(
                    run_split(
                        dataset.frame,
                        split,
                        feature_columns=feature_columns,
                        model_key=model_key,
                        calibration_order=calibration_order,
                        budgets=budgets,
                        experiment_id=dataset.experiment_id,
                        generator=dataset.generator,
                        portfolio_mode=portfolio_mode,
                        selection_policy=selection_policy,
                        cooldown_minutes=cooldown_minutes,
                    )
                )
            final_results.extend(
                run_split(
                    dataset.frame,
                    final_split,
                    feature_columns=feature_columns,
                    model_key=model_key,
                    calibration_order=calibration_order,
                    budgets=budgets,
                    experiment_id=dataset.experiment_id,
                    generator=dataset.generator,
                    portfolio_mode=portfolio_mode,
                    selection_policy=selection_policy,
                    cooldown_minutes=cooldown_minutes,
                )
            )
        all_results.extend(outer_results)
        all_results.extend(final_results)
        candidate_side_counts = (
            {str(key): int(value) for key, value in pd.Series([c.side for c in dataset.candidates]).value_counts().sort_index().items()}
            if dataset.candidates
            else {}
        )
        label_side_counts = (
            {str(key): int(value) for key, value in dataset.labels["side"].value_counts().sort_index().items()}
            if not dataset.labels.empty and "side" in dataset.labels.columns
            else {}
        )
        bundles.append(ExperimentRunBundle(dataset.experiment_id, dataset.generator, dataset.blocks, len(dataset.frame), None, outer_results, final_results, None, candidate_side_counts, label_side_counts))

    results_frame = _result_frame(all_results)
    summary_frame = _summarize_results(all_results)

    primary_budget = _primary_budget(registry.goal, registry.model_dump().get("promotion_rules", {}))
    selected_models: dict[str, str | None] = {}
    for bundle in bundles:
        if bundle.skip_reason:
            selected_models[bundle.experiment_id] = None
        else:
            selected_models[bundle.experiment_id] = _best_model_for_experiment(summary_frame, bundle.experiment_id, primary_budget)

    promotion_frame = _promotion_report(registry, summary_frame, results_frame, selected_models)

    project_root = Path(registry_path).resolve().parent
    contracts_path = project_root / "feature_contracts.yaml"
    provenance = report_provenance(
        project_root=project_root,
        data_dir=data_root,
        registry_path=registry_path,
        contracts_path=contracts_path if contracts_path.exists() else None,
    )

    manifest = {
        "provenance": provenance,
        "program_id": registry.program["id"],
        "symbol": symbol,
        "available_models": available_models,
        "portfolio_evaluation": {
            "mode": portfolio_mode,
            "selection_policy": selection_policy,
            "cooldown_minutes": cooldown_minutes,
        },
        "experiments": [
            {
                "experiment_id": bundle.experiment_id,
                "generator": bundle.generator,
                "blocks": bundle.blocks,
                "dataset_rows": bundle.dataset_rows,
                "skip_reason": bundle.skip_reason,
                "selected_model": selected_models.get(bundle.experiment_id),
                "candidate_side_counts": bundle.candidate_side_counts or {},
                "label_side_counts": bundle.label_side_counts or {},
            }
            for bundle in bundles
        ],
    }

    manifest_path = reports_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_frame.to_csv(reports_root / "fold_metrics.csv", index=False)
    summary_frame.to_csv(reports_root / "summary_metrics.csv", index=False)
    promotion_frame.to_csv(reports_root / "promotion_decisions.csv", index=False)
    (reports_root / "promotion_report.md").write_text(
        _markdown_report(registry, summary_frame, promotion_frame, selected_models),
        encoding="utf-8",
    )
    return reports_root
