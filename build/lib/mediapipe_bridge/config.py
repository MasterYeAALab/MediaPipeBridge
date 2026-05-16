from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InputMode = Literal["camera", "ndi"]


@dataclass
class InputConfig:
    mode: InputMode = "camera"
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    ndi_source_name: str = ""


@dataclass
class FeatureConfig:
    pose: bool = True
    hands: bool = True
    face_mesh: bool = False
    face_detection: bool = False
    selfie_segmentation: bool = False
    draw_overlays: bool = True
    gesture_recognition: bool = False
    custom_gesture_recognition: bool = False
    use_gpu: bool = False
    model_complexity: int = 0  # 0=fast, 1=balanced
    process_scale: float = 0.5
    max_hands: int = 2
    max_faces: int = 1

    def enabled_key(self) -> tuple:
        return (
            self.pose,
            self.hands,
            self.face_mesh,
            self.face_detection,
            self.selfie_segmentation,
            self.max_hands,
            self.max_faces,
        )


@dataclass
class OscConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 9000


@dataclass
class NdiOutputConfig:
    send_camera_input: bool = False
    camera_input_name: str = "MediaPipe Bridge Input"
    send_processed: bool = False
    processed_name: str = "MediaPipe Bridge Processed"
    output_scale: float = 1.0  # 1.0, 0.5, 0.25


@dataclass
class AppConfig:
    input: InputConfig
    features: FeatureConfig
    osc: OscConfig
    ndi: NdiOutputConfig

    @classmethod
    def defaults(cls) -> AppConfig:
        return cls(
            input=InputConfig(),
            features=FeatureConfig(),
            osc=OscConfig(),
            ndi=NdiOutputConfig(),
        )
