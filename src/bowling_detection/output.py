from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .detector import Detection


@dataclass(frozen=True)
class LaneResult:
    lane: str
    count: int
    detections: list[Detection]
    image: np.ndarray


class ResultPublisher:
    def __init__(self, output_directory: Path, report_url: str | None) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.report_url = report_url

    def publish(self, result: LaneResult) -> None:
        image_path = self.output_directory / f"latest-{result.lane}.jpg"
        json_path = self.output_directory / f"latest-{result.lane}.json"
        cv2.imwrite(str(image_path), result.image)
        payload = {
            "lane": result.lane,
            "standing_pins": result.count,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "detections": [
                {"label": item.label, "score": item.score, "box": item.box}
                for item in result.detections
            ],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        if self.report_url:
            self._post(result, image_path)

    def _post(self, result: LaneResult, image_path: Path) -> None:
        import requests

        token = os.environ.get("BOWLING_API_TOKEN")
        if not token:
            raise RuntimeError("BOWLING_API_TOKEN is required when reporting is enabled")
        with image_path.open("rb") as image:
            response = requests.post(
                self.report_url,
                data={
                    "api_token": token,
                    "mac_address": ":".join(
                        f"{(uuid.getnode() >> shift) & 0xff:02x}"
                        for shift in range(40, -1, -8)
                    ),
                    "lane": result.lane,
                    "nbr_pins_up": result.count,
                },
                files={"img_frame": image},
                timeout=10,
            )
        response.raise_for_status()
