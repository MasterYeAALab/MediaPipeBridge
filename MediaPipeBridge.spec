# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [
    ('models', 'models'),
    ('Font', 'Font'),
]
binaries = []
hiddenimports = [
    'mediapipe.tasks.python.vision',
    'mediapipe.tasks.python.core',
]

# 核心修复：收集 MediaPipe 和相关库的所有二进制资源
for pkg in ['mediapipe', 'cyndilib', 'pythonosc']:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ['run.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MediaPipeBridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MediaPipeBridge',
)
app = BUNDLE(
    coll,
    name='MediaPipeBridge.app',
    icon=None,
    bundle_identifier='com.mediapipe.bridge',
    info_plist={
        'NSCameraUsageDescription': 'This app uses the camera for gesture and landmark tracking.',
        'NSMicrophoneUsageDescription': 'This app may use the microphone.',
    },
)
