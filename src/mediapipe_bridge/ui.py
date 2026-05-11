from __future__ import annotations

import json
import os
import time

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QDialog,
    QListWidget,
    QInputDialog,
    QMessageBox,
)

from .capture import CameraSource, VideoSource, list_camera_devices
from .config import AppConfig, FeatureConfig, InputConfig, NdiOutputConfig, OscConfig
from .ndi import CyndilibNdiReceiver, CyndilibNdiSender, list_ndi_sources
from .osc import OscOutput
from .vision import MediaPipeProcessor
from .gesture import GestureLibrary

def get_resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import sys
    import os
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(base_path, relative_path)

def get_data_path(filename: str) -> str:
    """ Get absolute path to writable data file, works for dev and for PyInstaller bundle. """
    import sys
    import os
    if getattr(sys, 'frozen', False):
        if sys.platform == 'darwin':
            data_dir = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'MediaPipeBridge')
        else:
            data_dir = os.path.join(os.path.expanduser('~'), '.mediapipe_bridge')
    else:
        data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)


class PipelineWorker(QThread):
    frame_ready = Signal(object)
    status_changed = Signal(str)
    stats_changed = Signal(str)
    vision_results_ready = Signal(object)

    def __init__(self, config: AppConfig, gesture_library: GestureLibrary = None) -> None:
        super().__init__()
        self.config = config
        self.gesture_library = gesture_library
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        source: VideoSource | CyndilibNdiReceiver | None = None
        processor = MediaPipeProcessor()
        processor.gesture_library = self.gesture_library
        osc = OscOutput(
            enabled=self.config.osc.enabled,
            host=self.config.osc.host,
            port=self.config.osc.port,
        )
        raw_sender: CyndilibNdiSender | None = None
        processed_sender: CyndilibNdiSender | None = None
        raw_sender_enabled = (
            self.config.input.mode == "camera" and self.config.ndi.send_camera_input
        )
        processed_sender_enabled = self.config.ndi.send_processed

        if osc.error:
            self.status_changed.emit(osc.error)

        try:
            source = self._create_source()
            source.open()
            self.status_changed.emit("Pipeline running")
            frame_count = 0
            last_vision_error = ""
            stats_started_at = time.monotonic()

            while self._running:
                packet = source.read()
                if packet is None:
                    self.msleep(5)
                    continue

                if raw_sender_enabled:
                    if raw_sender is None:
                        raw_sender = CyndilibNdiSender(
                            self.config.ndi.camera_input_name,
                            fps=self.config.input.fps,
                        )
                    try:
                        raw_sender.send_bgr(packet.frame_bgr)
                    except Exception as exc:
                        raw_sender_enabled = False
                        self.status_changed.emit(f"Raw NDI output disabled: {exc}")

                preview_frame, ndi_frame, vision_packet = processor.process(
                    packet.frame_bgr,
                    self.config.features,
                    packet.timestamp_ms,
                    packet.source_name,
                )
                if vision_packet.errors:
                    current_error = vision_packet.errors[0]
                    if current_error != last_vision_error:
                        self.status_changed.emit(current_error)
                        last_vision_error = current_error
                self.vision_results_ready.emit(vision_packet.results)
                osc.send(vision_packet)
                if osc.error:
                    self.status_changed.emit(osc.error)
                    osc.error = None

                if processed_sender_enabled:
                    if processed_sender is None:
                        processed_sender = CyndilibNdiSender(
                            self.config.ndi.processed_name,
                            fps=self.config.input.fps,
                        )
                    try:
                        out_frame = ndi_frame
                        if self.config.ndi.output_scale < 1.0:
                            import cv2
                            out_h = max(1, int(ndi_frame.shape[0] * self.config.ndi.output_scale))
                            out_w = max(1, int(ndi_frame.shape[1] * self.config.ndi.output_scale))
                            out_frame = cv2.resize(ndi_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                        if out_frame.ndim == 3 and out_frame.shape[2] == 4:
                            processed_sender.send_bgra(out_frame)
                        else:
                            processed_sender.send_bgr(out_frame)
                    except Exception as exc:
                        processed_sender_enabled = False
                        self.status_changed.emit(f"Processed NDI output disabled: {exc}")

                self.frame_ready.emit(preview_frame)
                frame_count += 1
                elapsed = time.monotonic() - stats_started_at
                if elapsed >= 1.0:
                    self.stats_changed.emit(f"{frame_count / elapsed:.1f} fps")
                    frame_count = 0
                    stats_started_at = time.monotonic()

                # No sleep — run at full speed, limited by source FPS and inference time
        except Exception as exc:
            self.status_changed.emit(f"Pipeline stopped: {exc}")
        finally:
            processor.close()
            if raw_sender is not None:
                raw_sender.close()
            if processed_sender is not None:
                processed_sender.close()
            if source is not None:
                source.close()
            self._running = False
            self.stats_changed.emit("Idle")

    def _create_source(self) -> VideoSource | CyndilibNdiReceiver:
        if self.config.input.mode == "camera":
            return CameraSource(
                camera_index=self.config.input.camera_index,
                width=self.config.input.width,
                height=self.config.input.height,
                fps=self.config.input.fps,
            )
        return CyndilibNdiReceiver(self.config.input.ndi_source_name)


class GestureManagerDialog(QDialog):
    def __init__(self, library: GestureLibrary, parent=None):
        super().__init__(parent)
        self.library = library
        self.parent_window = parent
        self.setWindowTitle("Custom Gestures Manager")
        self.resize(300, 400)
        
        layout = QVBoxLayout(self)
        
        self.list_widget = QListWidget()
        self._refresh_list()
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        self.btn_record = QPushButton("Record New")
        self.btn_delete = QPushButton("Delete Selected")
        btn_layout.addWidget(self.btn_record)
        btn_layout.addWidget(self.btn_delete)
        layout.addLayout(btn_layout)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        self.btn_record.clicked.connect(self._on_record)
        self.btn_delete.clicked.connect(self._on_delete)
        
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._on_tick)
        self._ticks = 0
        self._pending_name = ""

    def _refresh_list(self):
        self.list_widget.clear()
        self.list_widget.addItems(self.library.list_gestures())

    def _on_delete(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        name = item.text()
        self.library.delete(name)
        self.library.save(get_data_path("custom_gestures.json"))
        self._refresh_list()

    def _on_record(self):
        name, ok = QInputDialog.getText(self, "New Gesture", "Enter gesture name:")
        if not ok or not name.strip():
            return
        self._pending_name = name.strip()
        self.btn_record.setEnabled(False)
        self._ticks = 3
        self.status_label.setText(f"Get ready... {self._ticks}")
        self._countdown_timer.start(1000)

    def _on_tick(self):
        self._ticks -= 1
        if self._ticks > 0:
            self.status_label.setText(f"Get ready... {self._ticks}")
        else:
            self._countdown_timer.stop()
            self._capture()

    def _capture(self):
        self.btn_record.setEnabled(True)
        # Try to read the latest landmarks from the parent window's vision results
        if not hasattr(self.parent_window, "last_vision_results") or not self.parent_window.last_vision_results:
            self.status_label.setText("Error: Pipeline not running or no results")
            return
            
        hands = self.parent_window.last_vision_results.get("hands", [])
        if not hands:
            self.status_label.setText("Error: No hand detected!")
            return
            
        landmarks = hands[0].get("landmarks")
        if self.library.record(self._pending_name, landmarks):
            self.library.save(get_data_path("custom_gestures.json"))
            self.status_label.setText(f"Saved: {self._pending_name}")
            self._refresh_list()
        else:
            self.status_label.setText("Error: Invalid landmarks")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MediaPipe Bridge")
        self.resize(1280, 850)
        
        self.gesture_library = GestureLibrary()
        self.gesture_library.load(get_data_path("custom_gestures.json"))
        self.last_vision_results = {}

        self._worker: PipelineWorker | None = None
        self._build_ui()
        self._connect_signals()
        self._update_input_state()
        self._refresh_camera_sources()
        self.load_settings()

    def _apply_dopamine_style(self, group: QGroupBox, bg_color: str, text_color: str) -> None:
        group.setStyleSheet(f"""
            QGroupBox {{
                background-color: {bg_color};
                border-radius: 12px;
                border: 3px solid rgba(255,255,255,0.3);
                margin-top: 0px;
                padding-top: 32px;
                padding-left: 10px;
                padding-right: 10px;
                padding-bottom: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: padding;
                subcontrol-position: top left;
                left: 12px;
                top: 8px;
                color: {text_color};
                font-weight: bold;
                font-size: 16px;
            }}
            QLabel, QCheckBox, QRadioButton {{
                color: {text_color};
                background: transparent;
                font-weight: bold;
                font-size: 13px;
            }}
            QCheckBox::indicator, QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid #111111;
                background-color: rgba(255, 255, 255, 0.6);
            }}
            QCheckBox::indicator {{
                border-radius: 4px;
            }}
            QRadioButton::indicator {{
                border-radius: 9px;
            }}
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
                background-color: #111111;
            }}
            QComboBox, QLineEdit, QSpinBox, QPushButton {{
                background-color: rgba(255, 255, 255, 0.85);
                color: #111111;
                border: 1px solid rgba(0,0,0,0.1);
                border-radius: 6px;
                padding: 4px;
                font-weight: bold;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QPushButton:hover {{
                background-color: white;
            }}
        """)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        central.setStyleSheet("#centralWidget { background-color: #ED3883; }")
        shell = QVBoxLayout(central)
        shell.setContentsMargins(14, 14, 14, 14)
        shell.setSpacing(10)

        self.preview = QLabel("No video")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(720, 405)
        self.preview.setStyleSheet(
            "QLabel { background: #111111; color: #a8b0b7; border: 4px solid #15C3D3; border-radius: 12px; font-weight: bold; font-size: 16px; }"
        )
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        shell.addWidget(self.preview, 1)

        self.status = QLabel("Idle")
        self.stats = QLabel("Idle")
        self.status.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        self.stats.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        status_bar = QHBoxLayout()
        status_bar.addWidget(self.status, 1)
        status_bar.addWidget(self.stats, 0)
        shell.addLayout(status_bar, 0)

        controls = QFrame()
        controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(self._input_group())
        controls_layout.addWidget(self._features_group())
        controls_layout.addWidget(self._osc_group())
        controls_layout.addWidget(self._ndi_output_group())

        buttons = QVBoxLayout()
        self.toggle_button = QPushButton()
        self.toggle_button.setMinimumHeight(60)
        self.toggle_button.setMinimumWidth(100)
        self._update_toggle_button(False)
        
        buttons.addWidget(self.toggle_button)
        buttons.addStretch(1)
        controls_layout.addLayout(buttons)

        shell.addWidget(controls, 0)
        self.setCentralWidget(central)

    def _input_group(self) -> QGroupBox:
        group = QGroupBox("Input")
        self._apply_dopamine_style(group, "#15C3D3", "#FFFFFF")
        layout = QFormLayout(group)

        mode_row = QHBoxLayout()
        self.camera_radio = QRadioButton("Camera")
        self.ndi_radio = QRadioButton("NDI")
        self.camera_radio.setChecked(True)
        mode_row.addWidget(self.camera_radio)
        mode_row.addWidget(self.ndi_radio)
        mode_row.addStretch(1)
        layout.addRow("Mode", mode_row)

        camera_row = QHBoxLayout()
        self.camera_source = QComboBox()
        self.camera_source.setEditable(False)
        self.refresh_cameras = QPushButton("Refresh")
        camera_row.addWidget(self.camera_source, 1)
        camera_row.addWidget(self.refresh_cameras)
        layout.addRow("Camera", camera_row)

        size_row = QHBoxLayout()
        self.width = QSpinBox()
        self.width.setRange(160, 7680)
        self.width.setSingleStep(160)
        self.width.setValue(1280)
        self.height = QSpinBox()
        self.height.setRange(120, 4320)
        self.height.setSingleStep(90)
        self.height.setValue(720)
        size_row.addWidget(self.width)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.height)
        layout.addRow("Resolution", size_row)

        self.fps = QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setValue(30)
        layout.addRow("FPS", self.fps)

        ndi_row = QHBoxLayout()
        self.ndi_source = QComboBox()
        self.ndi_source.setEditable(True)
        self.refresh_sources = QPushButton("Refresh")
        ndi_row.addWidget(self.ndi_source, 1)
        ndi_row.addWidget(self.refresh_sources)
        layout.addRow("NDI Source", ndi_row)
        return group

    def _features_group(self) -> QGroupBox:
        group = QGroupBox("MediaPipe")
        self._apply_dopamine_style(group, "#F7D002", "#111111")
        layout = QGridLayout(group)
        layout.setVerticalSpacing(8)
        
        self.pose = QCheckBox("Pose")
        self.pose.setChecked(True)
        self.hands = QCheckBox("Hands")
        self.hands.setChecked(True)
        self.gesture_canned = QCheckBox("Canned Gestures")
        self.gesture_custom = QCheckBox("Custom Gestures")
        self.face_mesh = QCheckBox("Face Mesh")
        self.face_detection = QCheckBox("Face Detection")
        self.segmentation = QCheckBox("Segmentation")
        self.draw_overlays = QCheckBox("Draw overlays")
        self.draw_overlays.setChecked(True)
        
        self.btn_gesture_manager = QPushButton("Manage Gestures")
        
        layout.addWidget(self.pose, 0, 0)
        layout.addWidget(self.hands, 0, 1)
        layout.addWidget(self.face_mesh, 1, 0)
        layout.addWidget(self.face_detection, 1, 1)
        layout.addWidget(self.segmentation, 2, 0)
        layout.addWidget(self.draw_overlays, 2, 1)
        layout.addWidget(self.gesture_canned, 3, 0)
        layout.addWidget(self.gesture_custom, 3, 1)
        layout.addWidget(self.btn_gesture_manager, 4, 0, 1, 2)

        # Model complexity
        self.model_complexity_combo = QComboBox()
        self.model_complexity_combo.addItem("Fast", 0)
        self.model_complexity_combo.addItem("Balanced", 1)
        self.model_complexity_combo.setCurrentIndex(0)
        self.model_complexity_combo.setToolTip(
            "Fast: lower accuracy, higher FPS. Balanced: better accuracy, slower."
        )
        layout.addWidget(QLabel("Model:"), 5, 0)
        layout.addWidget(self.model_complexity_combo, 6, 0)

        # Process Scale
        self.process_scale_combo = QComboBox()
        self.process_scale_combo.addItem("100%", 1.0)
        self.process_scale_combo.addItem("50%", 0.5)
        self.process_scale_combo.addItem("25%", 0.25)
        self.process_scale_combo.setCurrentIndex(1)
        self.process_scale_combo.setToolTip(
            "Scale down image before processing to improve FPS. Results still map to high-res."
        )
        layout.addWidget(QLabel("Process Scale:"), 5, 1)
        layout.addWidget(self.process_scale_combo, 6, 1)

        # CPU / GPU delegate selection
        self.delegate_combo = QComboBox()
        self.delegate_combo.addItem("CPU", False)
        self.delegate_combo.addItem("GPU", True)
        self.delegate_combo.setCurrentIndex(0)
        layout.addWidget(QLabel("MediaPipe Runs On:"), 7, 0)
        layout.addWidget(self.delegate_combo, 8, 0, 1, 2)
        return group

    def _osc_group(self) -> QGroupBox:
        group = QGroupBox("OSC")
        self._apply_dopamine_style(group, "#FFB1CB", "#B80E1C")
        layout = QFormLayout(group)
        self.osc_enabled = QCheckBox("Send OSC")
        self.osc_enabled.setChecked(True)
        self.osc_host = QLineEdit("127.0.0.1")
        self.osc_port = QSpinBox()
        self.osc_port.setRange(1, 65535)
        self.osc_port.setValue(9000)
        layout.addRow("", self.osc_enabled)
        layout.addRow("Host", self.osc_host)
        layout.addRow("Port", self.osc_port)
        return group

    def _ndi_output_group(self) -> QGroupBox:
        group = QGroupBox("NDI Output")
        self._apply_dopamine_style(group, "#2F9DDF", "#FFFFFF")
        layout = QFormLayout(group)
        self.send_camera_ndi = QCheckBox("Send camera input")
        self.camera_ndi_name = QLineEdit("MediaPipe Bridge Input")
        self.send_processed_ndi = QCheckBox("Send processed video")
        self.processed_ndi_name = QLineEdit("MediaPipe Bridge Processed")
        
        self.ndi_output_scale_combo = QComboBox()
        self.ndi_output_scale_combo.addItem("100% (Match Input)", 1.0)
        self.ndi_output_scale_combo.addItem("50%", 0.5)
        self.ndi_output_scale_combo.addItem("25%", 0.25)
        self.ndi_output_scale_combo.setCurrentIndex(0)
        
        layout.addRow("", self.send_camera_ndi)
        layout.addRow("Input name", self.camera_ndi_name)
        layout.addRow("", self.send_processed_ndi)
        layout.addRow("Processed name", self.processed_ndi_name)
        layout.addRow("Output Scale", self.ndi_output_scale_combo)
        return group

    def _connect_signals(self) -> None:
        self.camera_radio.toggled.connect(self._update_input_state)
        self.ndi_radio.toggled.connect(self._update_input_state)
        self.refresh_cameras.clicked.connect(self._refresh_camera_sources)
        self.refresh_sources.clicked.connect(self._refresh_ndi_sources)
        self.toggle_button.clicked.connect(self._on_toggle_clicked)
        self.btn_gesture_manager.clicked.connect(self._open_gesture_manager)

    @Slot()
    def _open_gesture_manager(self) -> None:
        dialog = GestureManagerDialog(self.gesture_library, self)
        dialog.exec()

    @Slot()
    def _on_toggle_clicked(self) -> None:
        if self._worker is None:
            self._start()
        else:
            self._stop()

    def _update_toggle_button(self, running: bool) -> None:
        if running:
            self.toggle_button.setText("■")
            self.toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: #E31A22;
                    color: white;
                    border-radius: 12px;
                    font-size: 48px;
                    border: 3px solid rgba(255,255,255,0.4);
                    padding-bottom: 8px;
                }
                QPushButton:hover { background-color: #FF1A24; border: 3px solid white; }
                QPushButton:disabled { background-color: rgba(227, 26, 34, 0.4); color: rgba(255,255,255,0.4); }
            """)
        else:
            self.toggle_button.setText("▶")
            self.toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: #F7D002;
                    color: #111;
                    border-radius: 12px;
                    font-size: 30px;
                    border: 3px solid rgba(255,255,255,0.4);
                    padding-left: 6px;
                }
                QPushButton:hover { background-color: #FFEA00; border: 3px solid white; }
                QPushButton:disabled { background-color: rgba(247, 208, 2, 0.4); color: rgba(17,17,17,0.4); }
            """)

    @Slot()
    def _update_input_state(self) -> None:
        camera_mode = self.camera_radio.isChecked()
        self.camera_source.setEnabled(camera_mode)
        self.refresh_cameras.setEnabled(camera_mode)
        self.width.setEnabled(camera_mode)
        self.height.setEnabled(camera_mode)
        self.fps.setEnabled(camera_mode)
        self.send_camera_ndi.setEnabled(camera_mode)
        self.camera_ndi_name.setEnabled(camera_mode)
        self.ndi_source.setEnabled(not camera_mode)
        self.refresh_sources.setEnabled(not camera_mode)

    @Slot()
    def _refresh_ndi_sources(self) -> None:
        try:
            names = list_ndi_sources()
        except Exception as exc:
            self.status.setText(f"NDI discovery failed: {exc}")
            return
        current = self.ndi_source.currentText()
        self.ndi_source.clear()
        self.ndi_source.addItems(names)
        if current:
            self.ndi_source.setCurrentText(current)
        self.status.setText(f"Found {len(names)} NDI source(s)")

    @Slot()
    def _refresh_camera_sources(self) -> None:
        current_index = self.camera_source.currentData()
        self.camera_source.clear()
        try:
            devices = list_camera_devices()
        except Exception as exc:
            self.camera_source.addItem("Camera 0", 0)
            self.status.setText(f"Camera discovery failed: {exc}")
            return

        for device in devices:
            self.camera_source.addItem(device.label, device.index)

        if not devices:
            self.camera_source.addItem("Camera 0", 0)
            self.status.setText(
                "No cameras found. Check macOS Camera permission and whether another app is using it."
            )
            return

        if current_index is not None:
            row = self.camera_source.findData(current_index)
            if row >= 0:
                self.camera_source.setCurrentIndex(row)
        self.status.setText(f"Found {len(devices)} camera(s)")

    @Slot()
    def _start(self) -> None:
        if self._worker is not None:
            return
        config = self._read_config()
        self._set_controls_enabled(False)
        self._update_toggle_button(True)
        self._worker = PipelineWorker(config, self.gesture_library)
        self._worker.frame_ready.connect(self._show_frame)
        self._worker.status_changed.connect(self.status.setText)
        self._worker.stats_changed.connect(self.stats.setText)
        self._worker.vision_results_ready.connect(self._on_vision_results)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    @Slot()
    def _stop(self) -> None:
        if self._worker is not None:
            self.status.setText("Stopping...")
            self._worker.stop()

    @Slot()
    def _worker_finished(self) -> None:
        self._worker = None
        self._set_controls_enabled(True)
        self._update_toggle_button(False)
        self._update_input_state()

    def _read_config(self) -> AppConfig:
        return AppConfig(
            input=InputConfig(
                mode="camera" if self.camera_radio.isChecked() else "ndi",
                camera_index=int(self.camera_source.currentData() or 0),
                width=self.width.value(),
                height=self.height.value(),
                fps=self.fps.value(),
                ndi_source_name=self.ndi_source.currentText().strip(),
            ),
            features=FeatureConfig(
                pose=self.pose.isChecked(),
                hands=self.hands.isChecked(),
                face_mesh=self.face_mesh.isChecked(),
                face_detection=self.face_detection.isChecked(),
                selfie_segmentation=self.segmentation.isChecked(),
                draw_overlays=self.draw_overlays.isChecked(),
                gesture_recognition=self.gesture_canned.isChecked(),
                custom_gesture_recognition=self.gesture_custom.isChecked(),
                use_gpu=bool(self.delegate_combo.currentData()),
                model_complexity=int(self.model_complexity_combo.currentData() or 0),
                process_scale=float(self.process_scale_combo.currentData() or 1.0),
            ),
            osc=OscConfig(
                enabled=self.osc_enabled.isChecked(),
                host=self.osc_host.text().strip() or "127.0.0.1",
                port=self.osc_port.value(),
            ),
            ndi=NdiOutputConfig(
                send_camera_input=self.send_camera_ndi.isChecked(),
                camera_input_name=self.camera_ndi_name.text().strip()
                or "MediaPipe Bridge Input",
                send_processed=self.send_processed_ndi.isChecked(),
                processed_name=self.processed_ndi_name.text().strip()
                or "MediaPipe Bridge Processed",
                output_scale=float(self.ndi_output_scale_combo.currentData() or 1.0),
            ),
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.camera_radio,
            self.ndi_radio,
            self.camera_source,
            self.refresh_cameras,
            self.width,
            self.height,
            self.fps,
            self.ndi_source,
            self.refresh_sources,
            self.pose,
            self.hands,
            self.gesture_canned,
            self.gesture_custom,
            self.face_mesh,
            self.face_detection,
            self.segmentation,
            self.draw_overlays,
            self.model_complexity_combo,
            self.process_scale_combo,
            self.delegate_combo,
            self.osc_enabled,
            self.osc_host,
            self.osc_port,
            self.send_camera_ndi,
            self.camera_ndi_name,
            self.send_processed_ndi,
            self.processed_ndi_name,
            self.ndi_output_scale_combo,
        ):
            widget.setEnabled(enabled)

    @Slot(object)
    def _on_vision_results(self, results: dict) -> None:
        self.last_vision_results = results

    @Slot(np.ndarray)
    def _show_frame(self, frame: np.ndarray) -> None:
        if frame.size == 0:
            return
        channels = frame.shape[2] if frame.ndim == 3 else 1
        if channels == 4:
            # BGRA -> RGBA for display
            frame_rgba = np.ascontiguousarray(frame[:, :, [2, 1, 0, 3]])
            height, width = frame_rgba.shape[:2]
            image = QImage(
                frame_rgba.data, width, height, width * 4, QImage.Format_RGBA8888
            ).copy()
        else:
            # BGR -> RGB for display
            frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
            height, width = frame_rgb.shape[:2]
            image = QImage(
                frame_rgb.data, width, height, width * 3, QImage.Format_RGB888
            ).copy()
        pixmap = QPixmap.fromImage(image)
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_settings()
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()

    def load_settings(self) -> None:
        path = get_data_path("settings.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            self.camera_radio.setChecked(data.get("camera_radio", True))
            self.ndi_radio.setChecked(data.get("ndi_radio", False))
            self.width.setValue(data.get("width", 1280))
            self.height.setValue(data.get("height", 720))
            self.fps.setValue(data.get("fps", 30))
            
            self.pose.setChecked(data.get("pose", True))
            self.hands.setChecked(data.get("hands", True))
            self.face_mesh.setChecked(data.get("face_mesh", False))
            self.face_detection.setChecked(data.get("face_detection", False))
            self.segmentation.setChecked(data.get("segmentation", False))
            self.draw_overlays.setChecked(data.get("draw_overlays", True))
            
            if "model_complexity" in data:
                idx = self.model_complexity_combo.findData(data["model_complexity"])
                if idx >= 0: self.model_complexity_combo.setCurrentIndex(idx)
            if "process_scale" in data:
                idx = self.process_scale_combo.findData(data["process_scale"])
                if idx >= 0: self.process_scale_combo.setCurrentIndex(idx)
            if "use_gpu" in data:
                idx = self.delegate_combo.findData(data["use_gpu"])
                if idx >= 0: self.delegate_combo.setCurrentIndex(idx)
            
            self.osc_enabled.setChecked(data.get("osc_enabled", True))
            self.osc_host.setText(data.get("osc_host", "127.0.0.1"))
            self.osc_port.setValue(data.get("osc_port", 9000))
            
            self.send_camera_ndi.setChecked(data.get("send_camera_ndi", False))
            self.camera_ndi_name.setText(data.get("camera_ndi_name", "MediaPipe Bridge Input"))
            self.send_processed_ndi.setChecked(data.get("send_processed_ndi", False))
            self.processed_ndi_name.setText(data.get("processed_ndi_name", "MediaPipe Bridge Processed"))
            
            if "output_scale" in data:
                idx = self.ndi_output_scale_combo.findData(data["output_scale"])
                if idx >= 0: self.ndi_output_scale_combo.setCurrentIndex(idx)
                
            self._update_input_state()
        except Exception as e:
            print(f"Failed to load settings: {e}")

    def save_settings(self) -> None:
        data = {
            "camera_radio": self.camera_radio.isChecked(),
            "ndi_radio": self.ndi_radio.isChecked(),
            "width": self.width.value(),
            "height": self.height.value(),
            "fps": self.fps.value(),
            "pose": self.pose.isChecked(),
            "hands": self.hands.isChecked(),
            "face_mesh": self.face_mesh.isChecked(),
            "face_detection": self.face_detection.isChecked(),
            "segmentation": self.segmentation.isChecked(),
            "draw_overlays": self.draw_overlays.isChecked(),
            "model_complexity": self.model_complexity_combo.currentData(),
            "process_scale": self.process_scale_combo.currentData(),
            "use_gpu": self.delegate_combo.currentData(),
            "osc_enabled": self.osc_enabled.isChecked(),
            "osc_host": self.osc_host.text(),
            "osc_port": self.osc_port.value(),
            "send_camera_ndi": self.send_camera_ndi.isChecked(),
            "camera_ndi_name": self.camera_ndi_name.text(),
            "send_processed_ndi": self.send_processed_ndi.isChecked(),
            "processed_ndi_name": self.processed_ndi_name.text(),
            "output_scale": self.ndi_output_scale_combo.currentData(),
        }
        try:
            with open(get_data_path("settings.json"), "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    
    import sys
    import os
    from PySide6.QtGui import QFontDatabase, QFont

    font_path = get_resource_path(os.path.join("Font", "XQJF.ttf"))
    
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                family = families[0]
                app.setFont(QFont(family))
                # Force the font in the global stylesheet to ensure it's used even when other stylesheets are applied
                app.setStyleSheet(app.styleSheet() + f"\n* {{ font-family: '{family}'; }}")
        else:
            print(f"Failed to load font from {font_path}")
    else:
        print(f"Font file not found at {font_path}")

    window = MainWindow()
    window.show()
    return app.exec()
