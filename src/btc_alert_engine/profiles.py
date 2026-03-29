from __future__ import annotations

from dataclasses import dataclass


_INTERVAL_CODE_TO_MINUTES: dict[str, int] = {
    "1": 1,
    "3": 3,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
    "120": 120,
    "240": 240,
    "360": 360,
    "720": 720,
    "D": 24 * 60,
}


def interval_code_to_label(interval_code: str) -> str:
    if interval_code == "60":
        return "1h"
    if interval_code == "120":
        return "2h"
    if interval_code == "240":
        return "4h"
    if interval_code == "360":
        return "6h"
    if interval_code == "720":
        return "12h"
    if interval_code == "D":
        return "1d"
    return f"{interval_code}m"


@dataclass(frozen=True, slots=True)
class StrategyProfile:
    id: str
    continuation_module: str
    stress_reversal_module: str
    feature_block_suffix: str
    trigger_interval_code: str
    setup_resample_rule: str
    regime_resample_rule: str
    setup_interval_label: str
    regime_interval_label: str
    setup_breakout_lookback: int = 20
    setup_active_bars: int = 10
    trigger_pivot_lookback: int = 4
    regime_ema_fast: int = 50
    regime_ema_slow: int = 200
    regime_slope_lookback: int = 5
    adx_period: int = 14
    atr_period: int = 14
    atr_percentile_days: int = 90
    continuation_timeout_bars: int = 96
    stress_timeout_bars: int = 32

    @property
    def trigger_interval_label(self) -> str:
        return interval_code_to_label(self.trigger_interval_code)

    @property
    def trigger_resample_rule(self) -> str:
        minutes = _INTERVAL_CODE_TO_MINUTES[self.trigger_interval_code]
        return f"{minutes}min"

    @property
    def trigger_minutes(self) -> int:
        return _INTERVAL_CODE_TO_MINUTES[self.trigger_interval_code]

    @property
    def regime_minutes(self) -> int:
        rule = self.regime_resample_rule.lower().replace("min", "")
        if rule.endswith("h"):
            return int(rule[:-1]) * 60
        return int(rule)

    @property
    def regime_bars_per_day(self) -> int:
        return max((24 * 60) // self.regime_minutes, 1)

    @property
    def feature_block_trend(self) -> str:
        return "trend_bybit" if self.feature_block_suffix == "" else f"trend_bybit_{self.feature_block_suffix}"

    @property
    def feature_block_regime(self) -> str:
        return "regime_bybit" if self.feature_block_suffix == "" else f"regime_bybit_{self.feature_block_suffix}"


CORE_PROFILE = StrategyProfile(
    id="core",
    continuation_module="continuation_v1",
    stress_reversal_module="stress_reversal_v0",
    feature_block_suffix="",
    trigger_interval_code="15",
    setup_resample_rule="1h",
    regime_resample_rule="4h",
    setup_interval_label="1h",
    regime_interval_label="4h",
    continuation_timeout_bars=96,
    stress_timeout_bars=32,
)

FAST_PROFILE = StrategyProfile(
    id="fast",
    continuation_module="continuation_v1_fast",
    stress_reversal_module="stress_reversal_v0_fast",
    feature_block_suffix="fast",
    trigger_interval_code="5",
    setup_resample_rule="15min",
    regime_resample_rule="1h",
    setup_interval_label="15m",
    regime_interval_label="1h",
    continuation_timeout_bars=288,
    stress_timeout_bars=96,
)


_PROFILES_BY_ID = {
    CORE_PROFILE.id: CORE_PROFILE,
    FAST_PROFILE.id: FAST_PROFILE,
}

_PROFILES_BY_GENERATOR = {
    CORE_PROFILE.continuation_module: CORE_PROFILE,
    CORE_PROFILE.stress_reversal_module: CORE_PROFILE,
    FAST_PROFILE.continuation_module: FAST_PROFILE,
    FAST_PROFILE.stress_reversal_module: FAST_PROFILE,
}


def get_profile(profile_id: str | StrategyProfile) -> StrategyProfile:
    if isinstance(profile_id, StrategyProfile):
        return profile_id
    key = str(profile_id).lower()
    if key not in _PROFILES_BY_ID:
        raise KeyError(f"Unknown strategy profile: {profile_id}")
    return _PROFILES_BY_ID[key]


def profile_for_generator(generator: str) -> StrategyProfile:
    try:
        return _PROFILES_BY_GENERATOR[str(generator)]
    except KeyError as exc:
        raise KeyError(f"Unknown generator/profile mapping: {generator}") from exc


def all_profiles() -> tuple[StrategyProfile, ...]:
    return (CORE_PROFILE, FAST_PROFILE)
