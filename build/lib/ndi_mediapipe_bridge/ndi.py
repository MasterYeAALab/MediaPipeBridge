from __future__ import annotations

import time
from fractions import Fraction
from typing import Any

import numpy as np

from .frames import FramePacket


class NdiUnavailableError(RuntimeError):
    pass


def _import_cyndilib() -> dict[str, Any]:
    try:
        from cyndilib import Finder, FourCC, Receiver, Sender, VideoFrameSync, VideoSendFrame
        from cyndilib.wrapper.ndi_recv import RecvBandwidth, RecvColorFormat
    except Exception as exc:  # pragma: no cover - depends on local NDI runtime
        raise NdiUnavailableError(
            "NDI backend unavailable. Install the optional dependency with "
            "`pip install .[ndi]` and ensure the NDI runtime/SDK license requirements are met."
        ) from exc

    return {
        "Finder": Finder,
        "FourCC": FourCC,
        "Receiver": Receiver,
        "Sender": Sender,
        "VideoFrameSync": VideoFrameSync,
        "VideoSendFrame": VideoSendFrame,
        "RecvBandwidth": RecvBandwidth,
        "RecvColorFormat": RecvColorFormat,
    }


def list_ndi_sources(timeout_s: float = 1.0) -> list[str]:
    ndi = _import_cyndilib()
    Finder = ndi["Finder"]

    finder = Finder()
    finder.open()
    try:
        finder.wait_for_sources(timeout=timeout_s)
        return list(finder.get_source_names())
    finally:
        finder.close()


class CyndilibNdiSender:
    def __init__(self, name: str, fps: int = 30) -> None:
        self.name = name.strip() or "MediaPipe Bridge"
        self.fps = max(1, fps)
        self._ndi = None
        self._sender = None
        self._video_frame = None
        self._shape: tuple[int, int] | None = None
        self._alpha_mode: bool = False  # False=BGRX, True=BGRA

    def open(self, width: int, height: int, alpha: bool = False) -> None:
        self._ndi = _import_cyndilib()
        Sender = self._ndi["Sender"]
        VideoSendFrame = self._ndi["VideoSendFrame"]
        FourCC = self._ndi["FourCC"]

        sender = Sender(self.name)
        video_frame = VideoSendFrame()
        video_frame.set_resolution(width, height)
        video_frame.set_frame_rate(Fraction(self.fps, 1))
        # BGRA preserves the alpha channel; BGRX tells receivers to ignore it
        video_frame.set_fourcc(FourCC.BGRA if alpha else FourCC.BGRX)
        sender.set_video_frame(video_frame)
        sender.open()

        self._sender = sender
        self._video_frame = video_frame
        self._shape = (height, width)
        self._alpha_mode = alpha

    def send_bgr(self, frame_bgr: np.ndarray) -> None:
        height, width = frame_bgr.shape[:2]
        if self._sender is None or self._shape != (height, width) or self._alpha_mode:
            self.close()
            self.open(width, height, alpha=False)

        bgrx = np.empty((height, width, 4), dtype=np.uint8)
        bgrx[:, :, :3] = frame_bgr[:, :, :3]
        bgrx[:, :, 3] = 255
        self._sender.write_video_async(memoryview(np.ascontiguousarray(bgrx).ravel()))

    def send_bgra(self, frame_bgra: np.ndarray) -> None:
        """Send a 4-channel BGRA frame preserving the alpha channel."""
        height, width = frame_bgra.shape[:2]
        if self._sender is None or self._shape != (height, width) or not self._alpha_mode:
            self.close()
            self.open(width, height, alpha=True)

        self._sender.write_video_async(
            memoryview(np.ascontiguousarray(frame_bgra).ravel())
        )

    def close(self) -> None:
        if self._sender is not None:
            self._sender.close()
        self._sender = None
        self._video_frame = None
        self._shape = None


class CyndilibNdiReceiver:
    def __init__(self, source_name: str) -> None:
        self.source_name = source_name.strip()
        self._ndi = None
        self._finder = None
        self._receiver = None
        self._video_frame = None

    def open(self) -> None:
        self._ndi = _import_cyndilib()
        Finder = self._ndi["Finder"]
        Receiver = self._ndi["Receiver"]
        VideoFrameSync = self._ndi["VideoFrameSync"]
        RecvBandwidth = self._ndi["RecvBandwidth"]
        RecvColorFormat = self._ndi["RecvColorFormat"]

        finder = Finder()
        finder.open()
        finder.wait_for_sources(timeout=2)
        source_names = list(finder.get_source_names())
        if not source_names:
            finder.close()
            raise RuntimeError("No NDI sources found on the network")

        requested = self.source_name or source_names[0]
        source = finder.get_source(requested)
        if source is None:
            for candidate in finder:
                if candidate.name == requested or candidate.stream_name == requested:
                    source = candidate
                    break
        if source is None:
            finder.close()
            raise RuntimeError(f'NDI source "{requested}" was not found')

        receiver = Receiver(
            color_format=RecvColorFormat.BGRX_BGRA,
            bandwidth=RecvBandwidth.highest,
            recv_name="MediaPipe Bridge Receiver",
        )
        video_frame = VideoFrameSync()
        receiver.frame_sync.set_video_frame(video_frame)
        receiver.set_source(source)

        deadline = time.monotonic() + 5
        while not receiver.is_connected() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not receiver.is_connected():
            receiver.close()
            finder.close()
            raise RuntimeError(f'Timed out connecting to NDI source "{requested}"')

        self.source_name = requested
        self._finder = finder
        self._receiver = receiver
        self._video_frame = video_frame

    def read(self) -> FramePacket | None:
        if self._receiver is None or self._video_frame is None:
            return None

        self._receiver.frame_sync.capture_video()
        width, height = self._video_frame.get_resolution()
        if width <= 0 or height <= 0 or self._video_frame.get_data_size() <= 0:
            return None

        raw = np.frombuffer(memoryview(self._video_frame), dtype=np.uint8)
        expected = width * height * 4
        if raw.size < expected:
            return None
        bgrx = raw[:expected].reshape((height, width, 4))
        frame_bgr = bgrx[:, :, :3].copy()
        return FramePacket(
            frame_bgr=frame_bgr,
            timestamp_ms=int(time.time() * 1000),
            source_name=f"ndi:{self.source_name}",
        )

    def close(self) -> None:
        if self._receiver is not None:
            self._receiver.close()
        if self._finder is not None:
            self._finder.close()
        self._receiver = None
        self._finder = None
        self._video_frame = None
