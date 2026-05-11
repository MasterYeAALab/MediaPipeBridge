from __future__ import annotations

import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .frames import FramePacket


@dataclass
class CameraDevice:
    index: int
    label: str
    width: int = 0
    height: int = 0


def _camera_backends(cv2) -> list[int]:
    backends: list[int] = []
    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    if platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_ANY"):
        backends.append(cv2.CAP_ANY)
    backends.append(0)
    return list(dict.fromkeys(backends))


def _open_capture(cv2, camera_index: int):
    last_capture = None
    for backend in _camera_backends(cv2):
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            return capture
        capture.release()
        last_capture = capture
    return last_capture


def list_camera_devices(max_index: int = 10) -> list[CameraDevice]:
    import cv2

    devices: list[CameraDevice] = []
    for index in range(max_index):
        capture = _open_capture(cv2, index)
        if capture is None or not capture.isOpened():
            continue

        ok, frame = capture.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            label = f"Camera {index} ({width}x{height})"
            devices.append(CameraDevice(index=index, label=label, width=width, height=height))
        capture.release()
    return devices


class VideoSource(ABC):
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> FramePacket | None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class CameraSource(VideoSource):
    def __init__(self, camera_index: int, width: int, height: int, fps: int) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._capture = None

    def open(self) -> None:
        import cv2

        capture = _open_capture(cv2, self.camera_index)
        if capture is None or not capture.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)

        ok, _ = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(
                f"Camera {self.camera_index} opened but returned no frames. "
                "Check camera permission, whether another app is using it, and macOS "
                "Privacy & Security > Camera access for the app/terminal."
            )
        self._capture = capture

    def read(self) -> FramePacket | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        return FramePacket(
            frame_bgr=frame,
            timestamp_ms=int(time.time() * 1000),
            source_name=f"camera:{self.camera_index}",
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
