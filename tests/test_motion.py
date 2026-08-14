import cv2
import numpy as np

from bowling_detection.motion import MotionDetector


def test_motion_requires_a_large_enough_contour():
    detector = MotionDetector(pixel_threshold=20, minimum_contour_area=100)
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    small_change = background.copy()
    cv2.rectangle(small_change, (5, 5), (8, 8), (255, 255, 255), -1)
    ball_change = background.copy()
    cv2.rectangle(ball_change, (25, 25), (55, 55), (255, 255, 255), -1)

    assert detector.update(background) is False
    assert detector.update(small_change) is False
    assert detector.update(ball_change) is True
