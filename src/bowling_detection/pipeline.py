from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

import cv2
import numpy as np

from .config import AppConfig, LaneConfig
from .detector import Detection
from .motion import MotionDetector
from .output import LaneResult, ResultPublisher


class Detector(Protocol):
    def detect(self, bgr_image: np.ndarray) -> list[Detection]: ...


class LaneState(Enum):
    ARMED = auto()
    SETTLING = auto()
    SAMPLING = auto()
    COOLDOWN = auto()


@dataclass
class LaneRuntime:
    name: str
    config: LaneConfig
    motion: MotionDetector
    state: LaneState = LaneState.ARMED
    triggered_at: int = 0
    cooldown_until: int = 0
    samples: list[tuple[list[Detection], np.ndarray]] = field(default_factory=list)


class BowlingPipeline:
    """Independent motion-to-result state machines for the two lanes."""

    def __init__(
        self, config: AppConfig, detector: Detector, publisher: ResultPublisher
    ) -> None:
        self.config = config
        self.detector = detector
        self.publisher = publisher
        self.lanes = [
            LaneRuntime(
                name,
                getattr(config, name),
                MotionDetector(config.motion_threshold, config.minimum_motion_area),
            )
            for name in ("left", "right")
        ]

    def process(self, frame_index: int, bgr_frame: np.ndarray) -> None:
        for lane in self.lanes:
            self._process_lane(lane, frame_index, bgr_frame)

    def _process_lane(
        self, lane: LaneRuntime, frame_index: int, frame: np.ndarray
    ) -> None:
        motion_crop = lane.config.motion_roi.crop(frame)

        if lane.state is LaneState.ARMED:
            if lane.motion.update(motion_crop):
                lane.triggered_at = frame_index
                lane.state = LaneState.SETTLING
            return

        if lane.state is LaneState.SETTLING:
            if frame_index < lane.triggered_at + self.config.settle_frames:
                return
            lane.state = LaneState.SAMPLING

        if lane.state is LaneState.SAMPLING:
            pin_crop = lane.config.pins_roi.crop(frame).copy()
            detections = self.detector.detect(pin_crop)
            lane.samples.append((detections, pin_crop))
            if len(lane.samples) >= self.config.inference_samples:
                detections, image = max(lane.samples, key=lambda sample: len(sample[0]))
                annotated = self._draw(image, detections)
                self.publisher.publish(
                    LaneResult(lane.name, len(detections), detections, annotated)
                )
                lane.samples.clear()
                lane.cooldown_until = lane.triggered_at + self.config.cooldown_frames
                lane.state = LaneState.COOLDOWN
            return

        if lane.state is LaneState.COOLDOWN and frame_index >= lane.cooldown_until:
            lane.motion.update(motion_crop)
            lane.state = LaneState.ARMED

    @staticmethod
    def _draw(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(image, (x1, y1), (x2, y2), (105, 240, 152), 2)
            cv2.putText(
                image,
                f"{detection.label} {detection.score:.0%}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (105, 240, 152),
                1,
                cv2.LINE_AA,
            )
        return image
