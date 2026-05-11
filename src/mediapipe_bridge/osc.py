from __future__ import annotations

from typing import Any

from .vision import VisionPacket


class OscOutput:
    def __init__(self, enabled: bool, host: str, port: int) -> None:
        self.enabled = enabled
        self.host = host
        self.port = port
        self._client = None
        self.error: str | None = None

        if not enabled:
            return
        try:
            from pythonosc.udp_client import SimpleUDPClient
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            self.error = f"python-osc unavailable: {exc}"
            return
        self._client = SimpleUDPClient(host, port)

    def send(self, packet: VisionPacket) -> None:
        if not self.enabled or self._client is None:
            return

        self._send(
            "/mp/frame",
            [packet.timestamp_ms, packet.width, packet.height, packet.source_name],
        )
        if packet.errors:
            for error in packet.errors:
                self._send("/mp/error", [error])

        self._send_pose(packet.results.get("pose"))
        self._send_hands(packet.results.get("hands") or [])
        self._send_face_mesh(packet.results.get("face_mesh") or [])
        self._send_face_detection(packet.results.get("face_detection") or [])
        segmentation = packet.results.get("selfie_segmentation")
        if segmentation:
            self._send(
                "/mp/selfie_segmentation/foreground_coverage",
                [float(segmentation.get("foreground_coverage", 0.0))],
            )

        self._send_gestures(
            packet.results.get("gesture_canned") or [],
            packet.results.get("gesture_custom") or [],
        )

    def _send_gestures(
        self,
        canned: list[dict[str, Any]],
        custom: list[dict[str, Any]],
    ) -> None:
        self._send("/mp/gesture/canned_count", [len(canned)])
        for gesture in canned:
            self._send(
                "/mp/gesture/canned",
                [
                    int(gesture.get("index", 0)),
                    str(gesture.get("name", "")),
                    float(gesture.get("score", 1.0)),
                ],
            )
            
        self._send("/mp/gesture/custom_count", [len(custom)])
        for gesture in custom:
            self._send(
                "/mp/gesture/custom",
                [
                    int(gesture.get("index", 0)),
                    str(gesture.get("name", "")),
                    float(gesture.get("score", 0.0)),
                ],
            )

    def _send_pose(self, pose: dict[str, Any] | None) -> None:
        if not pose:
            self._send("/mp/pose/count", [0])
            return
        landmarks = pose.get("landmarks") or []
        self._send("/mp/pose/count", [1 if landmarks else 0])
        for index, item in enumerate(landmarks):
            self._send_landmark("/mp/pose/landmark", [0, index], item)
        for index, item in enumerate(pose.get("world_landmarks") or []):
            self._send_landmark("/mp/pose/world_landmark", [0, index], item)

    def _send_hands(self, hands: list[dict[str, Any]]) -> None:
        self._send("/mp/hands/count", [len(hands)])
        for hand in hands:
            hand_index = int(hand.get("index", 0))
            self._send(
                "/mp/hands/handedness",
                [hand_index, hand.get("label", ""), float(hand.get("score", 0.0))],
            )
            for landmark_index, item in enumerate(hand.get("landmarks") or []):
                self._send_landmark("/mp/hands/landmark", [hand_index, landmark_index], item)

    def _send_face_mesh(self, faces: list[dict[str, Any]]) -> None:
        self._send("/mp/face_mesh/count", [len(faces)])
        for face in faces:
            face_index = int(face.get("index", 0))
            for landmark_index, item in enumerate(face.get("landmarks") or []):
                self._send_landmark("/mp/face_mesh/landmark", [face_index, landmark_index], item)

    def _send_face_detection(self, faces: list[dict[str, Any]]) -> None:
        self._send("/mp/face_detection/count", [len(faces)])
        for face in faces:
            self._send(
                "/mp/face_detection/box",
                [
                    int(face.get("index", 0)),
                    float(face.get("score", 0.0)),
                    float(face.get("xmin", 0.0)),
                    float(face.get("ymin", 0.0)),
                    float(face.get("width", 0.0)),
                    float(face.get("height", 0.0)),
                ],
            )

    def _send_landmark(self, address: str, prefix: list[Any], landmark: dict[str, Any]) -> None:
        self._send(
            address,
            [
                *prefix,
                float(landmark.get("x", 0.0)),
                float(landmark.get("y", 0.0)),
                float(landmark.get("z", 0.0)),
                float(landmark.get("visibility", 0.0)),
                float(landmark.get("presence", 0.0)),
            ],
        )

    def _send(self, address: str, values: list[Any]) -> None:
        try:
            self._client.send_message(address, values)
        except OSError as exc:
            self.error = f"OSC send failed: {exc}"
