import json
from pathlib import Path

from btc_alert_engine.normalize.replay import deterministic_replay_hash, replay_top_of_book


def test_replay_is_deterministic_and_stateful(tmp_path: Path) -> None:
    raw_dir = Path(__file__).parent / "data" / "raw_sample"
    first = deterministic_replay_hash([raw_dir], symbol="BTCUSDT")
    second = deterministic_replay_hash([raw_dir], symbol="BTCUSDT")
    assert first == second
    assert len(first) == 64

    states = replay_top_of_book([raw_dir], symbol="BTCUSDT")
    assert len(states) == 2
    assert str(states[-1].best_bid_price) == "16494.00"
    assert str(states[-1].best_ask_price) == "16610.80"

    mutated_dir = tmp_path / "raw_mutated"
    mutated_dir.mkdir(parents=True)
    sample_path = raw_dir / "sample.ndjson"
    lines = sample_path.read_text(encoding="utf-8").strip().splitlines()
    mutated = json.loads(lines[-1])
    mutated["payload"]["data"]["b"][0][1] = "0.051"
    (mutated_dir / "sample.ndjson").write_text("\n".join([*lines[:-1], json.dumps(mutated)]) + "\n", encoding="utf-8")

    mutated_hash = deterministic_replay_hash([mutated_dir], symbol="BTCUSDT")
    assert mutated_hash != first
