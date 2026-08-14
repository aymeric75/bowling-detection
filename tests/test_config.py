import json
from pathlib import Path

from bowling_detection.config import load_config


def test_loads_original_lane_geometry(tmp_path: Path):
    source = json.loads(Path("config/bowling.json").read_text())
    path = tmp_path / "bowling.json"
    path.write_text(json.dumps(source))

    config = load_config(path)

    assert config.left.pins_roi.x2 - config.left.pins_roi.x1 == 600
    assert config.right.pins_roi.x2 - config.right.pins_roi.x1 == 600
    assert config.settle_frames == 100
    assert config.cooldown_frames == 450
