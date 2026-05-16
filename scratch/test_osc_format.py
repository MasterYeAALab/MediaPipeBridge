import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath("src"))

from mediapipe_bridge.osc import OscOutput
from mediapipe_bridge.vision import VisionPacket

def test_osc():
    # Mock the OSC client to capture messages
    osc = OscOutput(enabled=True, host="127.0.0.1", port=9000)
    mock_client = MagicMock()
    osc._client = mock_client
    
    # Create a dummy packet
    packet = VisionPacket(
        timestamp_ms=1000,
        width=1920,
        height=1080,
        source_name="test_camera",
        results={
            "hands": [
                {
                    "index": 0,
                    "label": "Left",
                    "score": 0.95,
                    "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}] * 21
                }
            ],
            "pose": {
                "landmarks": [{"x": 0.4, "y": 0.5, "z": 0.6}] * 33,
                "world_landmarks": [{"x": 1.0, "y": 2.0, "z": 3.0}] * 33
            }
        }
    )
    
    osc.send(packet)
    
    print(f"Total messages sent: {mock_client.send_message.call_count}")
    for call in mock_client.send_message.call_args_list:
        addr, args = call.args
        if len(args) > 10:
            print(f"{addr}: [{args[0]}, {args[1]}, ... ({len(args)-2} floats)]")
        else:
            print(f"{addr}: {args}")

if __name__ == "__main__":
    test_osc()
