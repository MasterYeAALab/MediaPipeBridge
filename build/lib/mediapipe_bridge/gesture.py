"""Gesture recognition utilities.

Two recognition modes:
1. Canned gestures  – rule-based classifier over 21 MediaPipe hand landmarks.
   Matches the same names as the MediaPipe GestureRecognizer Tasks API:
   Thumb_Up, Thumb_Down, Pointing_Up, Victory, ILoveYou, Open_Palm,
   Closed_Fist, None.

2. Custom gestures  – record-and-match via normalised landmark feature
   vectors.  Cosine similarity against a saved library of poses.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# MediaPipe hand landmark indices (from the official 21-point model)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(landmarks: list[dict], idx: int, axis: str, default: float = 0.0) -> float:
    """Safely get one coordinate from a landmarks list."""
    try:
        return float(landmarks[idx].get(axis, default))
    except (IndexError, TypeError):
        return default


def _is_finger_extended(landmarks: list[dict], tip: int, pip: int, mcp: int, *, flip_y: bool) -> bool:
    """Return True when the finger is extended (tip is farther from wrist than pip).

    ``flip_y`` should be True when the hand is upside-down (wrist above MCP).
    In screen coordinates y increases downward, so an extended finger pointing
    *up* has tip.y < pip.y.  When flip_y is True the hand points downward so we
    invert the comparison.
    """
    tip_y = _get(landmarks, tip, "y")
    pip_y = _get(landmarks, pip, "y")
    if flip_y:
        return tip_y > pip_y
    return tip_y < pip_y


def _thumb_extended_right(landmarks: list[dict], label: str) -> bool:
    """Rough check: thumb tip is significantly to the right/left of IP joint,
    accounting for handedness (mirrored webcam)."""
    tip_x = _get(landmarks, THUMB_TIP, "x")
    ip_x = _get(landmarks, THUMB_IP, "x")
    mcp_x = _get(landmarks, THUMB_MCP, "x")
    # For Right hand (in mirrored frame = user's right), thumb extends toward +x
    # For Left hand it extends toward -x. Use label from MediaPipe.
    delta = tip_x - ip_x
    if label == "Left":
        return delta < -0.04
    return delta > 0.04


def _wrist_above_mcps(landmarks: list[dict]) -> bool:
    """Return True when wrist y is *above* (smaller y) the average MCP y,
    meaning the hand is held upside-down (fingers pointing down)."""
    wrist_y = _get(landmarks, WRIST, "y")
    avg_mcp_y = sum(
        _get(landmarks, idx, "y") for idx in [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    ) / 4.0
    return wrist_y < avg_mcp_y


# ---------------------------------------------------------------------------
# Canned gesture classifier
# ---------------------------------------------------------------------------

CANNED_GESTURES = [
    "Thumb_Up",
    "Thumb_Down",
    "Pointing_Up",
    "Victory",
    "ILoveYou",
    "Open_Palm",
    "Closed_Fist",
    "None",
]


def classify_canned_gesture(landmarks: list[dict], label: str = "Right") -> str:
    """Classify one hand's landmarks into a canned gesture name.

    Parameters
    ----------
    landmarks:
        List of 21 dicts with at least ``x`` and ``y`` keys (normalised 0-1).
    label:
        ``"Left"`` or ``"Right"`` as reported by MediaPipe handedness.

    Returns
    -------
    str
        One of the CANNED_GESTURES names.
    """
    if not landmarks or len(landmarks) < 21:
        return "None"

    flip = _wrist_above_mcps(landmarks)

    index_ext = _is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP, flip_y=flip)
    middle_ext = _is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, flip_y=flip)
    ring_ext = _is_finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP, flip_y=flip)
    pinky_ext = _is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP, flip_y=flip)
    thumb_ext = _thumb_extended_right(landmarks, label)

    # --- Closed fist ---
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "Closed_Fist"

    # --- Open palm: all four fingers extended ---
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "Open_Palm"

    # --- Pointing up: only index extended ---
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "Pointing_Up"

    # --- Victory / Peace: index + middle ---
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "Victory"

    # --- ILoveYou: thumb + index + pinky ---
    if thumb_ext and index_ext and not middle_ext and not ring_ext and pinky_ext:
        return "ILoveYou"

    # --- Thumb gestures (need thumb extended + all fingers curled) ---
    if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        # Distinguish up / down by wrist orientation
        wrist_y = _get(landmarks, WRIST, "y")
        thumb_tip_y = _get(landmarks, THUMB_TIP, "y")
        if thumb_tip_y < wrist_y:
            return "Thumb_Up"
        else:
            return "Thumb_Down"

    return "None"


# ---------------------------------------------------------------------------
# Custom gesture library
# ---------------------------------------------------------------------------

def _normalise_landmarks(landmarks: list[dict]) -> list[float] | None:
    """Convert 21 landmarks into a scale- and translation-invariant feature vector.

    Strategy:
    1. Subtract wrist position (translation invariance).
    2. Divide by the distance from wrist to middle MCP (scale invariance).
    3. Flatten (x, y, z) for all 21 points → 63-element vector.

    Returns None if landmarks are invalid.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    # Wrist as origin
    wx = _get(landmarks, WRIST, "x")
    wy = _get(landmarks, WRIST, "y")
    wz = _get(landmarks, WRIST, "z")

    # Scale reference: wrist → middle MCP distance
    mx = _get(landmarks, MIDDLE_MCP, "x") - wx
    my = _get(landmarks, MIDDLE_MCP, "y") - wy
    mz = _get(landmarks, MIDDLE_MCP, "z") - wz
    ref = math.sqrt(mx * mx + my * my + mz * mz)
    if ref < 1e-6:
        return None

    vector: list[float] = []
    for lm in landmarks:
        x = (float(lm.get("x", 0.0)) - wx) / ref
        y = (float(lm.get("y", 0.0)) - wy) / ref
        z = (float(lm.get("z", 0.0)) - wz) / ref
        vector.extend([x, y, z])
    return vector


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return dot / (mag_a * mag_b)


@dataclass
class GestureEntry:
    name: str
    vector: list[float]  # 63-element normalised feature vector


class GestureLibrary:
    """Manages a collection of named custom gesture poses.

    Usage
    -----
    >>> lib = GestureLibrary()
    >>> lib.load("custom_gestures.json")      # load saved gestures
    >>> lib.record("my_pose", landmarks)      # record current pose
    >>> name, sim = lib.match(landmarks) or (None, 0)
    >>> lib.save("custom_gestures.json")
    """

    def __init__(self, threshold: float = 0.92) -> None:
        self._entries: list[GestureEntry] = []
        self.threshold = threshold  # cosine similarity threshold (0-1)

    # ------------------------------------------------------------------
    def record(self, name: str, landmarks: list[dict]) -> bool:
        """Record a new gesture or overwrite an existing one with the same name.

        Returns True on success, False if landmarks are invalid.
        """
        vec = _normalise_landmarks(landmarks)
        if vec is None:
            return False
        # Remove existing entry with the same name
        self._entries = [e for e in self._entries if e.name != name]
        self._entries.append(GestureEntry(name=name, vector=vec))
        return True

    def delete(self, name: str) -> bool:
        """Remove a gesture by name. Returns True if it existed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.name != name]
        return len(self._entries) < before

    def list_gestures(self) -> list[str]:
        """Return the list of all recorded gesture names."""
        return [e.name for e in self._entries]

    def match(self, landmarks: list[dict]) -> tuple[str, float] | None:
        """Find the best-matching gesture above the threshold.

        Returns ``(name, similarity)`` or ``None`` if no match.
        """
        if not self._entries:
            return None
        vec = _normalise_landmarks(landmarks)
        if vec is None:
            return None

        best_name = ""
        best_sim = -1.0
        for entry in self._entries:
            sim = _cosine_similarity(vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_name = entry.name

        if best_sim >= self.threshold:
            return (best_name, best_sim)
        return None

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise the gesture library to JSON."""
        data = [{"name": e.name, "vector": e.vector} for e in self._entries]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load(self, path: str) -> None:
        """Load gesture library from JSON. Silently ignores missing file."""
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._entries = [
                GestureEntry(name=item["name"], vector=item["vector"])
                for item in data
                if "name" in item and "vector" in item
            ]
        except Exception:
            pass  # corrupt file – start fresh
