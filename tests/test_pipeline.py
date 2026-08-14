from dataclasses import replace

import numpy as np

from bowling_detection.config import AppConfig, LaneConfig, Roi
from bowling_detection.detector import Detection
from bowling_detection.pipeline import BowlingPipeline, LaneState


class FakeDetector:
    def __init__(self):
        self.counts = iter((3, 7, 5))

    def detect(self, image):
        return [Detection("topPin", 0.9, (1, 1, 5, 8))] * next(self.counts)


class FakePublisher:
    def __init__(self):
        self.results = []

    def publish(self, result):
        self.results.append(result)


class TriggerOnce:
    def __init__(self):
        self.triggered = False

    def update(self, image):
        if not self.triggered:
            self.triggered = True
            return True
        return False


def configuration():
    lane = LaneConfig(Roi(0, 0, 20, 20), Roi(0, 0, 20, 20))
    return AppConfig(
        frame_width=40,
        frame_height=20,
        frame_rate=30,
        process_every_n_frames=1,
        motion_threshold=75,
        minimum_motion_area=500,
        settle_frames=2,
        cooldown_frames=20,
        inference_samples=3,
        confidence_threshold=0.2,
        model_path=None,
        labels_path=None,
        output_directory=None,
        left=lane,
        right=replace(lane, motion_roi=Roi(20, 0, 40, 20)),
    )


def test_lane_waits_samples_and_keeps_highest_count():
    publisher = FakePublisher()
    pipeline = BowlingPipeline(configuration(), FakeDetector(), publisher)
    left, right = pipeline.lanes
    left.motion = TriggerOnce()
    right.state = LaneState.COOLDOWN
    right.cooldown_until = 999
    frame = np.zeros((20, 40, 3), dtype=np.uint8)

    for frame_index in range(5):
        pipeline.process(frame_index, frame)

    assert len(publisher.results) == 1
    assert publisher.results[0].lane == "left"
    assert publisher.results[0].count == 7
    assert left.state is LaneState.COOLDOWN
