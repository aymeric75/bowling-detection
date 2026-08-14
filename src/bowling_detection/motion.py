from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """Frame-difference detector used to spot a ball crossing a lane ROI."""

    def __init__(self, pixel_threshold: int, minimum_contour_area: float) -> None:
        self.pixel_threshold = pixel_threshold
        self.minimum_contour_area = minimum_contour_area
        self._previous: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> bool:
        if self._previous is None:
            self._previous = frame.copy()
            return False

        difference = cv2.absdiff(frame, self._previous)
        self._previous = frame.copy()
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(
            blurred, self.pixel_threshold, 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return any(
            cv2.contourArea(contour) > self.minimum_contour_area
            for contour in contours
        )
