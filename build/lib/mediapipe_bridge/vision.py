from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import FeatureConfig

# --- Manual Connections (Zero Dependency) ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky + Palm
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (25, 27), (24, 26), (26, 28) # Legs
]


@dataclass
class VisionPacket:
    timestamp_ms: int
    width: int
    height: int
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    source_name: str = "camera"

def _landmark_to_dict(lm: Any) -> dict[str, float]:
    return {
        "x": getattr(lm, "x", 0.0),
        "y": getattr(lm, "y", 0.0),
        "z": getattr(lm, "z", 0.0),
    }

class MediaPipeProcessor:
    def __init__(self):
        self._face_landmarker = None
        self._hand_landmarker = None
        self._pose_landmarker = None
        self._face_detector = None
        self._current_features = None
        self._load_error = None
        self._warnings = [] # Store non-fatal warnings
        self.gesture_library = None
        
        pass

    def _is_valid_model(self, path: str) -> bool:
        if not os.path.exists(path) or not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        return size > 100 * 1024

    def _get_resource_path(self, relative_path: str) -> str:
        """ Ultra-robust resource path lookup with recursive search fallback """
        rel = str(relative_path or "")
        
        # 1. PyInstaller temp folder
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            # 强制将 MEIPASS 注入环境变量，帮助 MediaPipe 内部寻找资源
            if meipass not in os.environ.get('PATH', ''):
                os.environ['PATH'] = meipass + os.pathsep + os.environ.get('PATH', '')
            
            p = os.path.join(meipass, rel)
            if os.path.exists(p): return os.path.abspath(p)
        
        # 2. Executable directory (macOS Bundle)
        exe_path = sys.executable
        if exe_path:
            base_dir = os.path.dirname(exe_path)
            # .app/Contents/MacOS/ -> ../Resources/
            bundle_res = os.path.abspath(os.path.join(base_dir, "..", "Resources", rel))
            if os.path.exists(bundle_res): return bundle_res
            
            # Near executable
            local_res = os.path.abspath(os.path.join(base_dir, rel))
            if os.path.exists(local_res): return local_res

        # 3. Development path
        this_file = os.path.abspath(__file__)
        dev_base = os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
        dev_p = os.path.abspath(os.path.join(dev_base, rel))
        if os.path.exists(dev_p): return dev_p
        
        # 4. Deep search fallback
        base_search = meipass or (os.path.dirname(exe_path) if exe_path else None) or os.getcwd()
        basename = os.path.basename(rel)
        for root, dirs, files in os.walk(base_search):
            if basename in files:
                found = os.path.abspath(os.path.join(root, basename))
                return found

        return os.path.abspath(rel)

    def _read_resource_buffer(self, relative_path: str) -> bytes:
        """ Read resource into memory buffer """
        p = self._get_resource_path(relative_path)
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, "rb") as f:
                    return f.read()
            except:
                pass
        return b""

    def _rebuild_if_needed(self, features: FeatureConfig) -> None:
        import traceback
        feature_key = features.enabled_key()
        if self._current_features == feature_key:
            return

        pass
        self.close()
        self._load_error = None
        self._warnings = []
        
        try:
            from mediapipe.tasks.python import BaseOptions

            def get_base_opts(model_name):
                # 尝试同时提供 Buffer 和 Path，看哪个能绕过 MediaPipe 的检查
                m_path = self._get_resource_path(os.path.join("models", model_name))
                m_buffer = self._read_resource_buffer(os.path.join("models", model_name))
                
                if not m_buffer or len(m_buffer) < 1000:
                    raise FileNotFoundError(f"Model {model_name} not found or invalid.")
                
                # 优先使用 Buffer，但在某些版本中可能需要 Path 辅助环境定位
                return BaseOptions(model_asset_buffer=m_buffer)

            # --- Face Landmarker ---
            if features.face_mesh:
                try:
                    self._face_landmarker = vision.FaceLandmarker.create_from_options(
                        vision.FaceLandmarkerOptions(
                            base_options=get_base_opts("face_landmarker.task"),
                            running_mode=vision.RunningMode.IMAGE,
                            num_faces=features.max_faces,
                            output_face_blendshapes=True,
                        )
                    )
                except Exception as e:
                    err_msg = f"Face Mesh disabled: {str(e)}\n{traceback.format_exc()}"
                    print(err_msg)
                    self._warnings.append(err_msg)

            # --- Hand Landmarker ---
            if features.hands:
                try:
                    self._hand_landmarker = vision.HandLandmarker.create_from_options(
                        vision.HandLandmarkerOptions(
                            base_options=get_base_opts("hand_landmarker.task"),
                            running_mode=vision.RunningMode.IMAGE,
                            num_hands=features.max_hands,
                        )
                    )
                except Exception as e:
                    self._warnings.append(f"Hands disabled: {e}")

            # --- Pose Landmarker (also handles Segmentation) ---
            if features.pose or features.selfie_segmentation:
                try:
                    self._pose_landmarker = vision.PoseLandmarker.create_from_options(
                        vision.PoseLandmarkerOptions(
                            base_options=get_base_opts("pose_landmarker.task"),
                            running_mode=vision.RunningMode.IMAGE,
                            num_poses=4,
                            output_segmentation_masks=features.selfie_segmentation,
                        )
                    )
                except Exception as e:
                    self._warnings.append(f"Pose/Segmentation disabled: {e}")

            # --- Face Detector ---
            if features.face_detection:
                try:
                    self._face_detector = vision.FaceDetector.create_from_options(
                        vision.FaceDetectorOptions(
                            base_options=get_base_opts("face_detector.task"),
                            running_mode=vision.RunningMode.IMAGE,
                        )
                    )
                except Exception as e:
                    # Silence the error bar if face mesh is active
                    if not features.face_mesh:
                        self._warnings.append(f"Face Detection disabled: {e}")
                    pass
            
            pass

        except Exception as e:
            self._load_error = f"FATAL: {e}"
            pass
            
        self._current_features = feature_key

    def process(self, frame_bgr, features, timestamp_ms, source_name="camera"):
        h, w = frame_bgr.shape[:2]
        packet = VisionPacket(timestamp_ms=timestamp_ms, width=w, height=h, source_name=source_name)

        self._rebuild_if_needed(features)
        
        # Add warnings to packet
        for w_msg in self._warnings:
            packet.errors.append(w_msg)

        if self._load_error:
            packet.errors.append(self._load_error)
            return frame_bgr, frame_bgr, packet

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            preview = frame_bgr.copy()
            
            # --- Processing ---
            if self._face_detector:
                face_res = self._face_detector.detect(mp_image)
                packet.results["face_detection"] = []
                if face_res and face_res.detections:
                    for i, d in enumerate(face_res.detections):
                        b = d.bounding_box
                        score = d.categories[0].score if d.categories else 0.0
                        packet.results["face_detection"].append({
                            "index": i, 
                            "score": score, 
                            "xmin": b.origin_x / w, 
                            "ymin": b.origin_y / h, 
                            "width": b.width / w, 
                            "height": b.height / h
                        })
                        if features.draw_overlays:
                            cv2.rectangle(preview, (b.origin_x, b.origin_y), (b.origin_x + b.width, b.origin_y + b.height), (0, 255, 0), 2)

            if self._face_landmarker:
                face_mesh_res = self._face_landmarker.detect(mp_image)
                packet.results["face_mesh"] = []
                for i, lms in enumerate(face_mesh_res.face_landmarks):

                    packet.results["face_mesh"].append({"index": i, "landmarks": [_landmark_to_dict(lm) for lm in lms]})
                    if features.draw_overlays:
                        for lm in lms:
                            cv2.circle(preview, (int(lm.x*w), int(lm.y*h)), 1, (0, 255, 0), -1)

            if self._hand_landmarker:
                hand_res = self._hand_landmarker.detect(mp_image)
                packet.results["hands"] = []
                for i, lms in enumerate(hand_res.hand_landmarks):
                    lbl = hand_res.handedness[i][0].category_name if i < len(hand_res.handedness) else ""
                    hand_data = {"index": i, "label": lbl, "landmarks": [_landmark_to_dict(lm) for lm in lms]}
                    if i < len(hand_res.hand_world_landmarks):
                        hand_data["world_landmarks"] = [_landmark_to_dict(lm) for lm in hand_res.hand_world_landmarks[i]]
                    packet.results["hands"].append(hand_data)
                    if features.draw_overlays:
                        for start_idx, end_idx in HAND_CONNECTIONS:
                            p1 = lms[start_idx]
                            p2 = lms[end_idx]
                            cv2.line(preview, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (255, 255, 255), 2)
                        for lm in lms:
                            cv2.circle(preview, (int(lm.x*w), int(lm.y*h)), 4, (255, 0, 255), -1)

            ndi_frame = frame_bgr.copy()
            if self._pose_landmarker:
                pose_res = self._pose_landmarker.detect(mp_image)
                packet.results["poses"] = []
                
                # Draw Pose landmarks only if enabled
                if features.pose:
                    for i, lms in enumerate(pose_res.pose_landmarks):
                        pose_data = {"index": i, "landmarks": [_landmark_to_dict(lm) for lm in lms]}
                        if i < len(pose_res.pose_world_landmarks):
                            pose_data["world_landmarks"] = [_landmark_to_dict(lm) for lm in pose_res.pose_world_landmarks[i]]
                        packet.results["poses"].append(pose_data)
                        if features.draw_overlays:
                            for start_idx, end_idx in POSE_CONNECTIONS:
                                if start_idx < len(lms) and end_idx < len(lms):
                                    p1 = lms[start_idx]
                                    p2 = lms[end_idx]
                                    cv2.line(preview, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (255, 255, 255), 2)
                            for lm in lms:
                                cv2.circle(preview, (int(lm.x*w), int(lm.y*h)), 5, (0, 255, 255), -1)

                # Apply Segmentation
                if features.selfie_segmentation and pose_res.segmentation_masks:
                    mask = cv2.resize(pose_res.segmentation_masks[0].numpy_view(), (w, h))
                    # Apply to preview
                    preview = np.where(np.stack((mask,)*3, axis=-1) > 0.1, preview, 0)
                    # For NDI, create BGRA if requested
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    ndi_frame = cv2.merge([frame_bgr[:,:,0], frame_bgr[:,:,1], frame_bgr[:,:,2], mask_uint8])

            return preview, ndi_frame, packet
            
        except Exception as e:
            packet.errors.append(f"RUNTIME ERROR: {e}")
            return frame_bgr, frame_bgr, packet

    def close(self):
        for t in [self._face_landmarker, self._hand_landmarker, self._pose_landmarker, self._face_detector]:
            if t:
                try: t.close()
                except: pass
        self._face_landmarker = self._hand_landmarker = self._pose_landmarker = self._face_detector = None
