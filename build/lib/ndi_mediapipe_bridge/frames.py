from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FramePacket:
    frame_bgr: np.ndarray
    timestamp_ms: int
    source_name: str
