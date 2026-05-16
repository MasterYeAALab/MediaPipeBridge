from __future__ import annotations
import traceback
import os
import time
from typing import Any
from .vision import VisionPacket

class OscOutput:
    def __init__(self, enabled: bool, host: str, port: int) -> None:
        self.enabled = enabled
        self.host = host
        self.port = port
        self._client = None
        self.error: str | None = None
        self._frame_count = 0

        if not enabled:
            return
        try:
            from pythonosc.udp_client import UDPClient
            self._client = UDPClient(host, port)
        except Exception as exc:
            self.error = f"OSC Init Error: {exc}"

    def send(self, packet: VisionPacket) -> None:
        if not self.enabled or self._client is None:
            return

        self._frame_count += 1
        fid = int(self._frame_count + 1000000000)

        try:
            # 1. Poses
            poses = packet.results.get("poses", [])
            if not poses and self._frame_count % 30 == 0:
                self._send_raw("/ofxmp/poses", fid, 0, []) # Empty heartbeat
            for pose in poses:
                lms = pose.get("landmarks", [])
                world_lms = pose.get("world_landmarks", [])
                
                # 强制补齐到 33 个点，防止 MediaPipe 在某些情况下返回点数不足导致下半身缺失
                while len(lms) < 33:
                    lms.append({"x": 0, "y": 0, "z": 0})
                while world_lms and len(world_lms) < 33:
                    world_lms.append({"x": 0, "y": 0, "z": 0})

                self._send_raw("/ofxmp/poses", fid, int(pose.get("index", 0)), lms[:33])
                if world_lms:
                    self._send_raw("/ofxmp/posesW", fid, int(pose.get("index", 0)), world_lms[:33])

            # 2. Hands
            hands = packet.results.get("hands", [])
            if not hands and self._frame_count % 30 == 0:
                self._send_raw("/ofxmp/hands", fid, 0, []) # Empty heartbeat
            for hand in hands:
                lms = hand.get("landmarks", [])
                world_lms = hand.get("world_landmarks", [])
                
                # 强制补齐到 21 个点
                while len(lms) < 21:
                    lms.append({"x": 0, "y": 0, "z": 0})
                while world_lms and len(world_lms) < 21:
                    world_lms.append({"x": 0, "y": 0, "z": 0})

                self._send_raw("/ofxmp/hands", fid, int(hand.get("index", 0)), lms[:21])
                if world_lms:
                    self._send_raw("/ofxmp/handsW", fid, int(hand.get("index", 0)), world_lms[:21])

            # 3. Face Mesh
            faces = packet.results.get("face_mesh", [])
            for face in faces:
                self._send_raw("/ofxmp/faces", fid, int(face.get("index", 0)), face.get("landmarks", [])[:478])
                # 如果未来有 Face World Landmarks，可以在这里添加 /ofxmp/facesW

        except Exception as e:
            self.error = f"OSC Error: {e}"

    def _send_raw(self, address: str, fid: int, oid: int, landmarks: list) -> None:
        from pythonosc.osc_message_builder import OscMessageBuilder
        builder = OscMessageBuilder(address=address)
        builder.add_arg(fid, arg_type='i')
        builder.add_arg(oid, arg_type='i')
        for lm in landmarks:
            builder.add_arg(float(lm["x"]), arg_type='f')
            builder.add_arg(float(lm["y"]), arg_type='f')
            builder.add_arg(float(lm["z"]), arg_type='f')
        msg = builder.build()
        self._client.send(msg)
