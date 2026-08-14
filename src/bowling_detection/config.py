from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Roi:
    x1: int
    y1: int
    x2: int
    y2: int

    def crop(self, frame):
        return frame[self.y1 : self.y2, self.x1 : self.x2]


@dataclass(frozen=True)
class LaneConfig:
    motion_roi: Roi
    pins_roi: Roi


@dataclass(frozen=True)
class AppConfig:
    frame_width: int
    frame_height: int
    frame_rate: int
    process_every_n_frames: int
    motion_threshold: int
    minimum_motion_area: float
    settle_frames: int
    cooldown_frames: int
    inference_samples: int
    confidence_threshold: float
    model_path: Path
    labels_path: Path
    output_directory: Path
    left: LaneConfig
    right: LaneConfig
    report_url: str | None = None


def _roi(values: list[int]) -> Roi:
    if len(values) != 4:
        raise ValueError("An ROI must contain [x1, y1, x2, y2]")
    return Roi(*map(int, values))


def load_config(path: Path) -> AppConfig:
    raw = json.loads(path.read_text())
    base = path.parent

    def lane(name: str) -> LaneConfig:
        values = raw["lanes"][name]
        return LaneConfig(_roi(values["motion_roi"]), _roi(values["pins_roi"]))

    return AppConfig(
        frame_width=raw["camera"]["width"],
        frame_height=raw["camera"]["height"],
        frame_rate=raw["camera"]["fps"],
        process_every_n_frames=raw["processing"]["every_n_frames"],
        motion_threshold=raw["motion"]["pixel_threshold"],
        minimum_motion_area=raw["motion"]["minimum_contour_area"],
        settle_frames=raw["timing"]["settle_frames"],
        cooldown_frames=raw["timing"]["cooldown_frames"],
        inference_samples=raw["inference"]["samples"],
        confidence_threshold=raw["inference"]["confidence_threshold"],
        model_path=(base / raw["inference"]["model"]).resolve(),
        labels_path=(base / raw["inference"]["labels"]).resolve(),
        output_directory=(base / raw["output"]["directory"]).resolve(),
        report_url=raw.get("reporting", {}).get("url"),
        left=lane("left"),
        right=lane("right"),
    )
