import sys
import os

# Import app directly - PyInstaller's collect_all and spec file should handle resources
from mediapipe_bridge.app import main

if __name__ == '__main__':
    main()
