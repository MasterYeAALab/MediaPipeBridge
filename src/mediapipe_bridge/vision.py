from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import FeatureConfig
from .gesture import classify_canned_gesture


@dataclass
class VisionPacket:
    timestamp_ms: int
    width: int
    height: int
    source_name: str
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _landmark_to_dict(landmark: Any) -> dict[str, float]:
    item = {
        "x": float(getattr(landmark, "x", 0.0)),
        "y": float(getattr(landmark, "y", 0.0)),
        "z": float(getattr(landmark, "z", 0.0)),
    }
    if hasattr(landmark, "visibility"):
        item["visibility"] = float(landmark.visibility)
    if hasattr(landmark, "presence"):
        item["presence"] = float(landmark.presence)
    return item


class MediaPipeProcessor:
    def __init__(self) -> None:
        self._mp = None
        self._cv2 = None
        self._drawing = None
        self._pose = None
        self._hands = None
        self._face_mesh = None
        self._face_detection = None
        self._segmentation = None  # shared: selfie_segmentation + transparent_bg
        self.gesture_library = None
        self._feature_key: tuple | None = None
        self._load_error: str | None = None

    def _ensure_imports(self) -> bool:
        if self._mp is not None and self._cv2 is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            os.environ.setdefault(
                "MPLCONFIGDIR",
                os.path.join(tempfile.gettempdir(), "mediapipe-bridge-matplotlib"),
            )
            import cv2
            import mediapipe as mp

            solutions = getattr(mp, "solutions", None)
            if solutions is None:
                raise RuntimeError(
                    f"mediapipe {getattr(mp, '__version__', 'unknown')} does not expose "
                    "the legacy solutions API required by this build. Install "
                    "mediapipe==0.10.21 or switch this app to the Tasks API backend."
                )
            drawing = solutions.drawing_utils
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            self._load_error = f"MediaPipe/OpenCV unavailable: {exc}"
            return False
        self._cv2 = cv2
        self._mp = mp
        self._drawing = drawing
        return True

    def _rebuild_if_needed(self, features: FeatureConfig) -> None:
        key = (
            *features.enabled_key(),
            features.use_gpu,
            features.model_complexity,
        )
        if key == self._feature_key:
            return
        self.close()
        self._feature_key = key

        needs_segmentation = features.selfie_segmentation
        if not any(features.enabled_key()) and not needs_segmentation:
            return
        if not self._ensure_imports():
            return

        mp = self._mp
        complexity = features.model_complexity
        if features.pose:
            self._pose = mp.solutions.pose.Pose(
                model_complexity=complexity,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        if features.hands:
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=2,
                model_complexity=complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        if features.face_mesh:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        if features.face_detection:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5,
            )
        # Single segmentation instance for both selfie_segmentation and transparent_bg
        if needs_segmentation:
            # model_selection=0 is General model (better edges/fingers, slower)
            # model_selection=1 is Landscape model (faster, rougher edges)
            seg_model = 0 if features.model_complexity == 1 else 1
            self._segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=seg_model,
            )

    def process(
        self,
        frame_bgr: np.ndarray,
        features: FeatureConfig,
        timestamp_ms: int,
        source_name: str,
    ) -> tuple[np.ndarray, np.ndarray, VisionPacket]:
        """Return (preview_frame, ndi_frame, packet).

        preview_frame: has debug overlays drawn on it (for local preview).
        ndi_frame: clean frame without overlays (for NDI output), with
                   transparent background applied when enabled.
        """
        height, width = frame_bgr.shape[:2]
        packet = VisionPacket(
            timestamp_ms=timestamp_ms,
            width=width,
            height=height,
            source_name=source_name,
        )

        try:
            self._rebuild_if_needed(features)
        except Exception as exc:
            packet.errors.append(f"MediaPipe initialization failed: {exc}")
            return frame_bgr, frame_bgr, packet

        needs_segmentation = features.selfie_segmentation
        if not any(features.enabled_key()) and not needs_segmentation:
            return frame_bgr, frame_bgr, packet

        if not self._ensure_imports():
            packet.errors.append(self._load_error or "MediaPipe/OpenCV unavailable")
            return frame_bgr, frame_bgr, packet

        cv2 = self._cv2
        mp = self._mp
        drawing = self._drawing

        scale = features.process_scale
        if scale < 1.0:
            proc_w = max(1, int(width * scale))
            proc_h = max(1, int(height * scale))
            proc_bgr = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc_bgr = frame_bgr

        # --- Single BGR→RGB conversion, reused by all models ---
        rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)

        # --- Only copy for preview when overlays are needed ---
        needs_preview_copy = features.draw_overlays and any(features.enabled_key())
        preview = frame_bgr.copy() if needs_preview_copy else frame_bgr

        # --- Segmentation: run once, reuse mask for both features ---
        seg_mask = None
        if self._segmentation is not None:
            try:
                seg_result = self._segmentation.process(rgb)
                mask = getattr(seg_result, "segmentation_mask", None)
            except Exception as exc:
                packet.errors.append(f"Segmentation failed: {exc}")

            if mask is not None:
                if scale < 1.0:
                    seg_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                else:
                    seg_mask = mask
                
            if seg_mask is not None and features.selfie_segmentation:
                coverage = float(np.mean(seg_mask > 0.5))
                packet.results["selfie_segmentation"] = {"foreground_coverage": coverage}
                if features.draw_overlays:
                    tint = np.zeros_like(preview)
                    tint[:, :, 1] = 180
                    alpha = np.clip(seg_mask, 0.0, 1.0)[:, :, None] * 0.35
                    preview = (preview * (1.0 - alpha) + tint * alpha).astype(np.uint8)

        if self._pose is not None:
            try:
                pose = self._pose.process(rgb)
            except Exception as exc:
                packet.errors.append(f"Pose processing failed: {exc}")
                pose = None
            landmarks = getattr(pose, "pose_landmarks", None)
            world_landmarks = getattr(pose, "pose_world_landmarks", None)
            if landmarks is not None:
                items = [_landmark_to_dict(item) for item in landmarks.landmark]
                packet.results["pose"] = {"landmarks": items}
                if world_landmarks is not None:
                    packet.results["pose"]["world_landmarks"] = [
                        _landmark_to_dict(item) for item in world_landmarks.landmark
                    ]
                if features.draw_overlays:
                    drawing.draw_landmarks(
                        preview,
                        landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                    )

        if self._hands is not None:
            try:
                hands = self._hands.process(rgb)
            except Exception as exc:
                packet.errors.append(f"Hands processing failed: {exc}")
                hands = None
            multi_landmarks = getattr(hands, "multi_hand_landmarks", None) or []
            handedness = getattr(hands, "multi_handedness", None) or []
            packet.results["hands"] = []
            for index, landmarks in enumerate(multi_landmarks):
                label = ""
                score = 0.0
                if index < len(handedness) and handedness[index].classification:
                    classification = handedness[index].classification[0]
                    label = str(classification.label)
                    score = float(classification.score)
                lm_dicts = [_landmark_to_dict(item) for item in landmarks.landmark]
                packet.results["hands"].append(
                    {
                        "index": index,
                        "label": label,
                        "score": score,
                        "landmarks": lm_dicts,
                    }
                )
                
                # --- Gesture Recognition ---
                if features.gesture_recognition:
                    if "gesture_canned" not in packet.results:
                        packet.results["gesture_canned"] = []
                    canned_name = classify_canned_gesture(lm_dicts, label)
                    packet.results["gesture_canned"].append(
                        {
                            "index": index,
                            "name": canned_name,
                            "score": 1.0,  # Rule-based doesn't give a score
                        }
                    )
                
                if features.custom_gesture_recognition and self.gesture_library is not None:
                    if "gesture_custom" not in packet.results:
                        packet.results["gesture_custom"] = []
                    match_result = self.gesture_library.match(lm_dicts)
                    if match_result:
                        custom_name, custom_sim = match_result
                        packet.results["gesture_custom"].append(
                            {
                                "index": index,
                                "name": custom_name,
                                "score": custom_sim,
                            }
                        )

                if features.draw_overlays:
                    drawing.draw_landmarks(
                        preview,
                        landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

        if self._face_mesh is not None:
            try:
                face_mesh = self._face_mesh.process(rgb)
            except Exception as exc:
                packet.errors.append(f"Face mesh processing failed: {exc}")
                face_mesh = None
            faces = getattr(face_mesh, "multi_face_landmarks", None) or []
            packet.results["face_mesh"] = []
            for index, landmarks in enumerate(faces):
                packet.results["face_mesh"].append(
                    {
                        "index": index,
                        "landmarks": [_landmark_to_dict(item) for item in landmarks.landmark],
                    }
                )
                if features.draw_overlays:
                    connections = getattr(
                        mp.solutions.face_mesh,
                        "FACEMESH_CONTOURS",
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    )
                    drawing.draw_landmarks(preview, landmarks, connections)

        if self._face_detection is not None:
            try:
                face_detection = self._face_detection.process(rgb)
            except Exception as exc:
                packet.errors.append(f"Face detection failed: {exc}")
                face_detection = None
            detections = getattr(face_detection, "detections", None) or []
            packet.results["face_detection"] = []
            for index, detection in enumerate(detections):
                box = detection.location_data.relative_bounding_box
                score = float(detection.score[0]) if detection.score else 0.0
                packet.results["face_detection"].append(
                    {
                        "index": index,
                        "score": score,
                        "xmin": float(box.xmin),
                        "ymin": float(box.ymin),
                        "width": float(box.width),
                        "height": float(box.height),
                    }
                )
                if features.draw_overlays:
                    drawing.draw_detection(preview, detection)

        # --- Build NDI output: reuse seg_mask, no extra model run ---
        if features.selfie_segmentation and seg_mask is not None:
            bgra = np.empty((height, width, 4), dtype=np.uint8)
            bgra[:, :, :3] = frame_bgr
            bgra[:, :, 3] = (np.clip(seg_mask, 0.0, 1.0) * 255).astype(np.uint8)
            ndi_out = bgra
        else:
            ndi_out = frame_bgr  # no copy needed, just reference

        return preview, ndi_out, packet

    def close(self) -> None:
        for solution in (
            self._pose,
            self._hands,
            self._face_mesh,
            self._face_detection,
            self._segmentation,
        ):
            if solution is not None:
                solution.close()
        self._pose = None
        self._hands = None
        self._face_mesh = None
        self._face_detection = None
        self._segmentation = None
