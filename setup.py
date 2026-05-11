from setuptools import find_packages, setup


setup(
    name="mediapipe-bridge",
    version="0.1.0",
    description=(
        "Cross-platform GUI bridge for camera/NDI input, MediaPipe tracking, "
        "OSC data, and NDI video output."
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9,<3.13",
    install_requires=[
        "numpy>=1.26,<2",
        "opencv-contrib-python>=4.9,<4.12",
        "mediapipe==0.10.9",
        "python-osc>=1.8",
        "PySide6>=6.6",
    ],
    extras_require={
        "ndi": ["cyndilib>=0.0.9,<0.2"],
        "dev": ["pytest>=8", "ruff>=0.6"],
    },
    entry_points={
        "console_scripts": [
            "mediapipe-bridge=mediapipe_bridge.app:main",
        ],
    },
)
