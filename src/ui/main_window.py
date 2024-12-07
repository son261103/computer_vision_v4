from collections import defaultdict

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from pathlib import Path
import cv2
import time
from datetime import datetime

from .metrics_tab import EvaluationTab
from .video_thread import VideoThread
from .video_widget import VideoWidget
from src.core.detector import Detector

from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time
import psutil
import os

from ..core.metrics import DetectionMetrics


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Core components initialization
        self.metrics_tracker = DetectionMetrics()
        self.detector = Detector()
        self.video_thread = None
        self.evaluation_tab = None

        # State tracking
        self.current_video_path = None
        self.is_recording = False
        self.recording_start_time = None
        self.selected_classes = set()
        self.current_frame = 0
        self.current_metrics = None

        # Initialize settings
        self.settings = QSettings('YourCompany', 'TrafficDetector')

        # Setup UI components
        self.setup_window()
        self.setup_ui()

        # Initialize memory monitoring
        self.init_memory_monitoring()

        self.connect_signals()
        self.load_settings()

        # Initialize metrics update timer
        self.metrics_update_timer = QTimer()
        # self.metrics_update_timer.timeout.connect(self.refresh_metrics)
        self.metrics_update_timer.start(1000)  # Update every second

    def setup_window(self):
        self.setWindowTitle("Traffic Detection System")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 768)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
        """)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Video display with controls
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)

        self.video_widget = VideoWidget()
        video_layout.addWidget(self.video_widget)

        self.video_controls = self.create_video_controls()
        video_layout.addWidget(self.video_controls)
        left_layout.addWidget(video_container)

        # Stats panel
        stats_panel = self.create_stats_panel()
        left_layout.addWidget(stats_panel)

        main_layout.addWidget(left_panel, stretch=4)

        # Right panel with tabs
        right_panel = QTabWidget()
        right_panel.setStyleSheet("""
           QTabWidget {
               border: none;
           }
           QTabWidget::pane {
               border: 1px solid #ff69b4;
               border-radius: 5px;
               background-color: #2d2d2d;
           }
           QTabBar::tab {
               background-color: #1e1e1e;
               color: white;
               border: 1px solid #ff69b4;
               padding: 8px 15px;
               border-top-left-radius: 4px;
               border-top-right-radius: 4px;
           }
           QTabBar::tab:selected {
               background-color: #ff69b4;
           }
       """)

        # Add tabs
        right_panel.addTab(self.create_control_panel(), "Controls")
        right_panel.addTab(self.create_settings_panel(), "Settings")
        right_panel.addTab(self.create_export_panel(), "Export")
        right_panel.addTab(EvaluationTab(), "Đánh giá")

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("color: white; padding: 5px;")

        # Add status widgets
        self.recording_label = QLabel("⚫ Not Recording")
        self.status_bar.addPermanentWidget(self.recording_label)

        self.memory_label = QLabel("Memory: 0 MB")
        self.status_bar.addPermanentWidget(self.memory_label)

        # Add evaluation metrics
        self.eval_label = QLabel("Precision: 0% | Recall: 0% | F1: 0%")
        self.status_bar.addPermanentWidget(self.eval_label)

        # Memory monitoring
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(1000)

    def update_metrics(self, stats):
        if 'metrics' in stats:
            metrics = stats['metrics']
            # Update status bar
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            self.eval_label.setText(
                f"Precision: {precision:.1f}% | Recall: {recall:.1f}% | F1: {f1:.1f}%"
            )

            # Update evaluation tab
            eval_tab = self.findChild(EvaluationTab)
            if eval_tab:
                eval_tab.update_metrics(metrics)

    def create_video_controls(self):
        controls = QWidget()
        layout = QHBoxLayout(controls)

        # Speed control
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['0.5x', '1.0x', '1.5x', '2.0x'])
        self.speed_combo.setCurrentText('1.0x')
        self.speed_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #ff69b4;
                padding: 5px;
            }
        """)

        # Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #ff69b4;
                height: 8px;
                background: #2d2d2d;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ff69b4;
                border: 1px solid #ff69b4;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)

        # Screenshot button
        self.screenshot_btn = QPushButton("📸 Screenshot")
        self.screenshot_btn.setStyleSheet(self.get_button_style())

        # Record button
        self.record_btn = QPushButton("⚫ Record")
        self.record_btn.setStyleSheet(self.get_button_style())

        layout.addWidget(self.speed_combo)
        layout.addWidget(self.seek_slider)
        layout.addWidget(self.screenshot_btn)
        layout.addWidget(self.record_btn)

        return controls

    def create_settings_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Display settings
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_group)

        self.show_fps_cb = QCheckBox("Show FPS")
        self.show_labels_cb = QCheckBox("Show Labels")
        self.show_confidence_cb = QCheckBox("Show Confidence")

        for cb in [self.show_fps_cb, self.show_labels_cb, self.show_confidence_cb]:
            cb.setChecked(True)
            cb.setStyleSheet("color: white;")
            display_layout.addWidget(cb)

        # Export settings
        export_group = QGroupBox("Export Settings")
        export_layout = QVBoxLayout(export_group)

        self.auto_save_cb = QCheckBox("Auto Save Video")
        self.save_stats_cb = QCheckBox("Save Statistics")

        for cb in [self.auto_save_cb, self.save_stats_cb]:
            cb.setChecked(True)
            cb.setStyleSheet("color: white;")
            export_layout.addWidget(cb)

        layout.addWidget(display_group)
        layout.addWidget(export_group)
        layout.addStretch()

        return panel

    def create_export_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout(options_group)

        self.export_video_cb = QCheckBox("Export Video")
        self.export_stats_cb = QCheckBox("Export Statistics")
        self.export_frames_cb = QCheckBox("Export Frames")

        for cb in [self.export_video_cb, self.export_stats_cb, self.export_frames_cb]:
            cb.setChecked(True)
            cb.setStyleSheet("color: white;")
            options_layout.addWidget(cb)

        # Export button
        self.export_btn = QPushButton("📤 Export")
        self.export_btn.setStyleSheet(self.get_button_style())

        layout.addWidget(options_group)
        layout.addWidget(self.export_btn)
        layout.addStretch()

        return panel

    def connect_signals(self):
        # Existing signals
        self.load_btn.clicked.connect(self.load_video)
        self.start_btn.clicked.connect(self.start_detection)
        self.pause_btn.clicked.connect(self.pause_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.conf_slider.valueChanged.connect(self.update_confidence)

        # New control signals
        self.speed_combo.currentTextChanged.connect(self.update_speed)
        self.seek_slider.sliderReleased.connect(self.seek_video)
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.export_btn.clicked.connect(self.export_data)

        # Settings signals
        self.show_fps_cb.stateChanged.connect(self.update_display_settings)
        self.show_labels_cb.stateChanged.connect(self.update_display_settings)
        self.show_confidence_cb.stateChanged.connect(self.update_display_settings)

        # Connect class checkboxes
        for class_id, checkbox in self.class_checkboxes.items():
            checkbox.stateChanged.connect(
                lambda state, c=class_id: self.update_class_selection(c, state)
            )

    def update_speed(self, speed_text):
        if self.video_thread:
            speed = float(speed_text.replace('x', ''))
            self.video_thread.set_speed(speed)

    def seek_video(self):
        if self.video_thread:
            position = self.seek_slider.value()
            self.video_thread.seek_to_position(position)

    def take_screenshot(self):
        if hasattr(self, 'current_frame'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            path = Path(self.config['paths']['output']) / filename
            cv2.imwrite(str(path), self.current_frame)
            self.status_bar.showMessage(f"Screenshot saved: {filename}")

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.recording_start_time = time.time()
            self.record_btn.setText("⚫ Stop Recording")
            self.recording_label.setText("🔴 Recording")
        else:
            self.record_btn.setText("⚫ Record")
            self.recording_label.setText("⚫ Not Recording")

    def export_data(self):
        if not hasattr(self, 'current_video_path'):
            QMessageBox.warning(self, "Warning", "No video loaded")
            return

        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", str(Path.home())
        )

        if export_dir:
            if self.export_video_cb.isChecked():
                # Export processed video
                pass

            if self.export_stats_cb.isChecked():
                # Export statistics to CSV
                self.export_statistics(export_dir)

            if self.export_frames_cb.isChecked():
                # Export detected frames
                self.export_frames(export_dir)

    def update_memory_usage(self):
        try:
            import psutil
            process = psutil.Process()
            memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            self.memory_label.setText(f"Memory: {memory:.1f} MB")
        except:
            pass

    def load_settings(self):
        self.show_fps = self.settings.value('show_fps', True, type=bool)
        self.show_labels = self.settings.value('show_labels', True, type=bool)
        self.show_confidence = self.settings.value('show_confidence', True, type=bool)
        self.auto_save = self.settings.value('auto_save', True, type=bool)

    def save_settings(self):
        self.settings.setValue('show_fps', self.show_fps_cb.isChecked())
        self.settings.setValue('show_labels', self.show_labels_cb.isChecked())
        self.settings.setValue('show_confidence', self.show_confidence_cb.isChecked())
        self.settings.setValue('auto_save', self.auto_save_cb.isChecked())

    def closeEvent(self, event):
        self.save_settings()
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        self.memory_timer.stop()
        event.accept()

    def get_button_style(self):
        """Return consistent button style"""
        return """
            QPushButton {
                background-color: #ff69b4;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 14px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #ff1493;
            }
            QPushButton:pressed {
                background-color: #c51585;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """

    def create_button(self, text, icon):
        """Create a styled button with icon and text"""
        btn = QPushButton(f"{icon} {text}")
        btn.setMinimumHeight(40)
        btn.setEnabled(text == "Load Video")
        btn.setStyleSheet(self.get_button_style())
        return btn

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        panel.setStyleSheet("background-color: #2d2d2d; border-left: 2px solid #ff69b4;")

        # File controls
        file_group = QGroupBox("File Controls")
        file_layout = QVBoxLayout(file_group)

        self.load_btn = self.create_button("Load Video", "📁")
        self.start_btn = self.create_button("Start", "▶")
        self.pause_btn = self.create_button("Pause", "⏸")
        self.stop_btn = self.create_button("Stop", "⏹")

        for btn in [self.load_btn, self.start_btn, self.pause_btn, self.stop_btn]:
            file_layout.addWidget(btn)

        # Detection settings
        detect_group = QGroupBox("Detection Settings")
        detect_layout = QVBoxLayout(detect_group)

        # Confidence slider
        conf_widget = QWidget()
        conf_layout = QHBoxLayout(conf_widget)
        self.conf_label = QLabel("Confidence: 0.50")
        self.conf_label.setStyleSheet("color: white;")

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)

        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(self.conf_slider)
        detect_layout.addWidget(conf_widget)

        # Object selection
        self.class_checkboxes = {}
        classes = {
            "motorcycle": "Motorcycles",
            "car": "Cars",
            "bus": "Buses",
            "truck": "Trucks",
            "traffic_light": "Traffic Lights",
            "stop_sign": "Stop Signs"
        }

        for class_id, display_name in classes.items():
            cb = QCheckBox(display_name)
            cb.setChecked(True)
            cb.setStyleSheet("color: white;")
            self.class_checkboxes[class_id] = cb
            detect_layout.addWidget(cb)

        # Add groups to panel
        for group in [file_group, detect_group]:
            group.setStyleSheet("""
                QGroupBox {
                    color: white;
                    border: 1px solid #ff69b4;
                    border-radius: 5px;
                    margin-top: 0.5em;
                    padding-top: 0.5em;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    padding: 0 5px;
                }
            """)
            layout.addWidget(group)

        layout.addStretch()
        return panel

    def create_stats_panel(self):
        """Create detection statistics panel"""
        panel = QGroupBox("Detection Statistics")
        layout = QVBoxLayout(panel)

        # Top stats in grid layout
        stats_grid = QGridLayout()

        # Initialize stats labels
        self.stats_labels = {
            'total': QLabel("Total Objects: 0"),
            'fps': QLabel("FPS: 0.0"),
            'time': QLabel("Time: 00:00/00:00")
        }

        # Add vehicle type counters
        self.object_labels = {
            'Xe may': QLabel("Motorcycles: 0"),
            'o to': QLabel("Cars: 0"),
            'Xe buyt': QLabel("Buses: 0"),
            'Xe tai': QLabel("Trucks: 0"),
            'den giao thong': QLabel("Traffic Lights: 0"),
            'bien bao': QLabel("Stop Signs: 0")
        }

        # Add basic stats to grid
        stats_grid.addWidget(self.stats_labels['total'], 0, 0)
        stats_grid.addWidget(self.stats_labels['fps'], 0, 1)
        stats_grid.addWidget(self.stats_labels['time'], 0, 2)

        # Add vehicle counters to grid
        row = 1
        col = 0
        for label in self.object_labels.values():
            stats_grid.addWidget(label, row, col)
            col += 1
            if col > 2:  # 3 columns
                col = 0
                row += 1

        layout.addLayout(stats_grid)

        # Progress bar section
        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)

        progress_label = QLabel("Progress:")
        progress_label.setStyleSheet("background-color: transparent;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ff69b4;
                border-radius: 5px;
                background-color: #2d2d2d;
                color: white;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #ff69b4;
            }
        """)

        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar, stretch=1)

        layout.addWidget(progress_widget)

        # Style the panel and labels
        panel.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #ff69b4;
                border-radius: 5px;
                margin-top: 0.5em;
                padding: 0.5em;
                background-color: #2d2d2d;
            }
            QLabel {
                color: white;
                padding: 5px;
                background-color: #1e1e1e;
                border-radius: 3px;
                min-width: 120px;
            }
        """)

        return panel

    def load_video(self):
        """Handle video file loading"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            self.video_path = file_path
            self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
            self.start_btn.setEnabled(True)
            self.video_widget.clear_display()

            # Reset stats
            self.stats_labels['total'].setText("Total Objects: 0")
            self.stats_labels['fps'].setText("FPS: 0.0")
            self.stats_labels['time'].setText("Time: 00:00/00:00")
            self.progress_bar.setValue(0)
            for label in self.object_labels.values():
                label.setText(f"{label.text().split(':')[0]}: 0")

    def start_detection(self):
        if hasattr(self, 'video_path'):
            if not any(cb.isChecked() for cb in self.class_checkboxes.values()):
                QMessageBox.warning(self, "Warning", "Please select at least one object type to detect.")
                return

            self.video_thread = VideoThread(self.detector, self.video_path)
            self.video_thread.frame_ready.connect(self.video_widget.update_frame)
            self.video_thread.stats_ready.connect(self.update_stats)
            self.video_thread.metrics_ready.connect(self.update_metrics)  # Thêm kết nối metrics
            self.video_thread.error_occurred.connect(self.show_error)
            self.video_thread.start()

            self.update_button_states(True)

    def pause_detection(self):
        """Pause/Resume video processing"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.paused = not self.video_thread.paused
            self.pause_btn.setText("⏸ Resume" if self.video_thread.paused else "⏸ Pause")
            status = "Paused" if self.video_thread.paused else "Resumed"
            self.status_bar.showMessage(status)

    def stop_detection(self):
        """Stop video processing"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.update_button_states(False)
            self.video_widget.clear_display()
            self.status_bar.showMessage("Detection stopped")

            # Reset stats
            self.stats_labels['total'].setText("Total Objects: 0")
            self.stats_labels['fps'].setText("FPS: 0.0")
            self.stats_labels['time'].setText("Time: 00:00/00:00")
            self.progress_bar.setValue(0)
            for label in self.object_labels.values():
                label.setText(f"{label.text().split(':')[0]}: 0")

    def update_stats(self, stats):
        """Update statistics display"""
        try:
            # Update basic stats
            self.stats_labels['fps'].setText(f"FPS: {stats['fps']:.1f}")
            self.stats_labels['total'].setText(f"Total Objects: {stats['total_objects']}")

            # Update progress bar
            progress = stats['progress']
            self.progress_bar.setValue(int(progress))
            self.progress_bar.setFormat(f"{progress:.1f}%")

            # Update time
            current_frame = stats['frame_count']
            total_frames = stats['total_frames']
            fps = stats.get('video_fps', 30)
            current_time = self.frame_to_time(current_frame, fps)
            total_time = self.frame_to_time(total_frames, fps)
            self.stats_labels['time'].setText(f"Time: {current_time}/{total_time}")

            # Update object counts
            object_counts = stats.get('object_counts', {})
            for obj_type, label in self.object_labels.items():
                count = object_counts.get(obj_type, 0)
                label.setText(f"{label.text().split(':')[0]}: {count}")

        except Exception as e:
            print(f"Error updating stats: {e}")

    def frame_to_time(self, frame_count, fps):
        """Convert frame count to time string"""
        seconds = int(frame_count / fps)
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def update_confidence(self, value):
        """Update detection confidence threshold"""
        conf = value / 100
        self.conf_label.setText(f"Confidence: {conf:.2f}")
        if hasattr(self, 'detector'):
            self.detector.model.conf_threshold = conf

    def update_class_selection(self, class_id, state):
        """Update selected object classes for detection"""
        if hasattr(self, 'detector'):
            enabled = state == Qt.CheckState.Checked.value
            class_ids = []

            # Map class names to IDs
            class_mapping = {
                "motorcycle": [3],
                "car": [2],
                "bus": [5],
                "truck": [7],
                "traffic_light": [9],
                "stop_sign": [11]
            }

            if class_id in class_mapping:
                class_ids = class_mapping[class_id]

            # Update detector classes
            for class_num in class_ids:
                if enabled and class_num not in self.detector.model.classes:
                    self.detector.model.classes.append(class_num)
                elif not enabled and class_num in self.detector.model.classes:
                    self.detector.model.classes.remove(class_num)

    def update_button_states(self, processing):
        """Update button states based on processing status"""
        self.load_btn.setEnabled(not processing)
        self.start_btn.setEnabled(not processing)
        self.pause_btn.setEnabled(processing)
        self.stop_btn.setEnabled(processing)

        # Also disable/enable controls during processing
        self.conf_slider.setEnabled(not processing)
        for checkbox in self.class_checkboxes.values():
            checkbox.setEnabled(not processing)

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        """Handle application closing"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()

    def create_settings_panel(self):
        """Create settings panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Display settings
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_group)

        # Create checkboxes
        self.show_fps_cb = QCheckBox("Show FPS")
        self.show_labels_cb = QCheckBox("Show Labels")
        self.show_confidence_cb = QCheckBox("Show Confidence")

        for cb in [self.show_fps_cb, self.show_labels_cb, self.show_confidence_cb]:
            cb.setChecked(True)
            cb.setStyleSheet("color: white;")
            display_layout.addWidget(cb)

        # Export settings
        export_group = QGroupBox("Export Settings")
        export_layout = QVBoxLayout(export_group)

        self.auto_save_cb = QCheckBox("Auto Save Video")
        self.save_stats_cb = QCheckBox("Save Statistics")

        for cb in [self.auto_save_cb, self.save_stats_cb]:
            cb.setChecked(True)
            cb.setStyleSheet("color: white;")
            export_layout.addWidget(cb)

        # Style the groups
        for group in [display_group, export_group]:
            group.setStyleSheet("""
                QGroupBox {
                    color: white;
                    border: 1px solid #ff69b4;
                    border-radius: 5px;
                    margin-top: 0.5em;
                    padding-top: 0.5em;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    padding: 0 5px;
                }
            """)
            layout.addWidget(group)

        layout.addStretch()
        return panel

    def create_export_panel(self):
        """Create export panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout(options_group)

        self.export_video_cb = QCheckBox("Export Video")
        self.export_stats_cb = QCheckBox("Export Statistics")
        self.export_frames_cb = QCheckBox("Export Frames")

        for cb in [self.export_video_cb, self.export_stats_cb, self.export_frames_cb]:
            cb.setChecked(True)
            cb.setStyleSheet("color: white;")
            options_layout.addWidget(cb)

        # Export button
        self.export_btn = QPushButton("📤 Export")
        self.export_btn.setStyleSheet(self.get_button_style())

        # Style the group
        options_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #ff69b4;
                border-radius: 5px;
                margin-top: 0.5em;
                padding-top: 0.5em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 5px;
            }
        """)

        layout.addWidget(options_group)
        layout.addWidget(self.export_btn)
        layout.addStretch()
        return panel

    def update_display_settings(self):
        """Update display settings based on checkboxes"""
        if hasattr(self, 'detector'):
            # Update display settings in detector
            self.detector.model.show_fps = self.show_fps_cb.isChecked()
            self.detector.model.show_labels = self.show_labels_cb.isChecked()
            self.detector.model.show_confidence = self.show_confidence_cb.isChecked()

            # Update export settings
            if hasattr(self.detector, 'video_stream'):
                self.detector.video_stream.save_output = self.auto_save_cb.isChecked()

            # Save settings
            self.save_settings()

    def load_settings(self):
        """Load saved settings"""
        settings = QSettings('YourCompany', 'TrafficDetector')

        # Load display settings
        self.show_fps_cb.setChecked(settings.value('show_fps', True, type=bool))
        self.show_labels_cb.setChecked(settings.value('show_labels', True, type=bool))
        self.show_confidence_cb.setChecked(settings.value('show_confidence', True, type=bool))

        # Load export settings
        self.auto_save_cb.setChecked(settings.value('auto_save', True, type=bool))
        self.save_stats_cb.setChecked(settings.value('save_stats', True, type=bool))

        # Update detector settings
        self.update_display_settings()

    def save_settings(self):
        """Save current settings"""
        settings = QSettings('YourCompany', 'TrafficDetector')

        # Save display settings
        settings.setValue('show_fps', self.show_fps_cb.isChecked())
        settings.setValue('show_labels', self.show_labels_cb.isChecked())
        settings.setValue('show_confidence', self.show_confidence_cb.isChecked())

        # Save export settings
        settings.setValue('auto_save', self.auto_save_cb.isChecked())
        settings.setValue('save_stats', self.save_stats_cb.isChecked())

    def connect_settings_signals(self):
        """Connect settings signals"""
        # Display settings
        self.show_fps_cb.stateChanged.connect(self.update_display_settings)
        self.show_labels_cb.stateChanged.connect(self.update_display_settings)
        self.show_confidence_cb.stateChanged.connect(self.update_display_settings)

        # Export settings
        self.auto_save_cb.stateChanged.connect(self.update_display_settings)
        self.save_stats_cb.stateChanged.connect(self.update_display_settings)

        # Export button
        self.export_btn.clicked.connect(self.export_data)

    def export_data(self):
        """Export data based on selected options"""
        if not hasattr(self, 'video_path'):
            QMessageBox.warning(self, "Warning", "No video loaded")
            return

        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", str(Path.home())
        )

        if export_dir:
            try:
                if self.export_video_cb.isChecked():
                    self.export_video(export_dir)

                if self.export_stats_cb.isChecked():
                    self.export_statistics(export_dir)

                if self.export_frames_cb.isChecked():
                    self.export_frames(export_dir)

                self.status_bar.showMessage("Export completed successfully")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_video(self, export_dir):
        """Export processed video"""
        if hasattr(self, 'detector') and hasattr(self.detector, 'video_stream'):
            self.detector.video_stream.save_output = True
            self.detector.video_stream._get_output_path = lambda _: str(
                Path(export_dir) / f"processed_{Path(self.video_path).name}"
            )

    def export_statistics(self, export_dir):
        """Export detection statistics to CSV"""
        if hasattr(self, 'detector'):
            stats = self.detector.get_stats()
            stats_file = Path(export_dir) / "detection_stats.csv"

            with open(stats_file, 'w') as f:
                f.write("Category,Count\n")
                for category, count in stats['by_class'].items():
                    f.write(f"{category},{count}\n")

    def export_frames(self, export_dir):
        """Export frames with detections"""
        frames_dir = Path(export_dir) / "detected_frames"
        frames_dir.mkdir(exist_ok=True)

        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.frame_ready.connect(
                lambda frame: self.save_frame(frame, frames_dir)
            )

    def save_frame(self, frame, directory):
        """Save individual frame"""
        if frame is not None:
            frame_count = len(list(directory.glob("*.jpg")))
            frame_path = directory / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)

    def init_memory_monitoring(self):
        """Initialize memory monitoring"""
        # Create memory monitoring timer
        self.memory_timer = QTimer(self)
        self.memory_timer.setInterval(1000)  # Update every second
        self.memory_timer.timeout.connect(self.update_memory_usage)

        # Create memory usage label with styling
        self.memory_label = QLabel("Memory: 0 MB")
        self.memory_label.setStyleSheet("""
            QLabel {
                color: white;
                padding: 2px 8px;
                background-color: #2d2d2d;
                border-radius: 3px;
                border: 1px solid #ff69b4;
            }
        """)

        # Add to status bar
        self.status_bar.addPermanentWidget(self.memory_label)

        # Start monitoring
        self.memory_timer.start()

    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            # Get process memory info
            process = psutil.Process()
            memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

            # Update label
            self.memory_label.setText(f"Memory: {memory:.1f} MB")

            # Add warning style if memory usage is high
            if memory > 1000:  # Warning if over 1GB
                self.memory_label.setStyleSheet("""
                    QLabel {
                        color: white;
                        padding: 2px 8px;
                        background-color: #d32f2f;
                        border-radius: 3px;
                        border: 1px solid #ff69b4;
                    }
                """)
            else:
                self.memory_label.setStyleSheet("""
                    QLabel {
                        color: white;
                        padding: 2px 8px;
                        background-color: #2d2d2d;
                        border-radius: 3px;
                        border: 1px solid #ff69b4;
                    }
                """)

        except Exception as e:
            print(f"Error updating memory usage: {e}")