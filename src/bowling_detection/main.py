from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import load_config
from .detector import CoralPinDetector
from .output import ResultPublisher
from .pipeline import BowlingPipeline


def run(config_path: Path) -> None:
    from picamera2 import Picamera2

    config = load_config(config_path)
    detector = CoralPinDetector(
        config.model_path, config.labels_path, config.confidence_threshold
    )
    publisher = ResultPublisher(config.output_directory, config.report_url)
    pipeline = BowlingPipeline(config, detector, publisher)

    camera = Picamera2()
    camera.configure(
        camera.create_video_configuration(
            main={
                "size": (config.frame_width, config.frame_height),
                "format": "RGB888",
            },
            controls={"FrameRate": config.frame_rate},
        )
    )
    camera.start()

    try:
        frame_index = 0
        while True:
            rgb_frame = camera.capture_array("main")
            if frame_index % config.process_every_n_frames == 0:
                pipeline.process(
                    frame_index, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                )
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-lane bowling pin detection on Raspberry Pi + Coral"
    )
    parser.add_argument("--config", type=Path, default=Path("config/bowling.json"))
    arguments = parser.parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()
