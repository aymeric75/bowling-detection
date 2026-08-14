from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]


class CoralPinDetector:
    """Single-class TFLite detector accelerated by a Coral USB Edge TPU."""

    def __init__(self, model_path: Path, labels_path: Path, threshold: float) -> None:
        from pycoral.adapters import common, detect
        from pycoral.utils.dataset import read_label_file
        from pycoral.utils.edgetpu import make_interpreter

        self._common = common
        self._detect = detect
        self._labels = read_label_file(str(labels_path))
        self._threshold = threshold
        self._interpreter = make_interpreter(str(model_path))
        self._interpreter.allocate_tensors()

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        from PIL import Image

        rgb = bgr_image[:, :, ::-1]
        image = Image.fromarray(rgb)
        resized, scale = self._common.set_resized_input(
            self._interpreter,
            image.size,
            lambda size: image.resize(size, Image.Resampling.LANCZOS),
        )
        self._interpreter.invoke()
        objects = self._detect.get_objects(
            self._interpreter, self._threshold, scale
        )
        return [
            Detection(
                label=self._labels.get(obj.id, str(obj.id)),
                score=float(obj.score),
                box=(obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax),
            )
            for obj in objects
        ]
