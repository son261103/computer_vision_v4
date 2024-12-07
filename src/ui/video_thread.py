from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time
import psutil
import os


class VideoThread(QThread):
    frame_ready = pyqtSignal(object)
    stats_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    processing_finished = pyqtSignal()
    metrics_ready = pyqtSignal(dict)
    memory_updated = pyqtSignal(float)

    def __init__(self, detector, video_path, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.video_path = video_path
        self.running = False
        self.paused = False
        self.current_position = 0
        self.skip_frames = 1
        self.total_frames = 0
        self.fps = 0

        # Performance tracking
        self.processing_times = []
        self.fps_history = []
        self.memory_usage = []

        # Metrics tracking
        self.gt_data = self.load_ground_truth()
        self.detection_results = []
        self.frame_metrics = defaultdict(list)
        self.class_names = [
            'Xe may', 'Xe dap', 'o to', 'Xe buyt',
            'Xe tai', 'den giao thong', 'bien bao'
        ]

    def load_ground_truth(self):
        """Load and process ground truth annotations"""
        try:
            gt_path = Path(self.video_path).with_suffix('.txt')
            if gt_path.exists():
                df = pd.read_csv(gt_path)
                return self.process_ground_truth(df)
            return None
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return None

    def process_ground_truth(self, df):
        """Process ground truth data into structured format"""
        processed_data = defaultdict(list)
        for _, row in df.iterrows():
            processed_data[row['frame']].append({
                'class': row['class'],
                'bbox': [row['x1'], row['y1'], row['x2'], row['y2']],
                'id': row['id'] if 'id' in row else None,
                'confidence': row.get('confidence', 1.0)
            })
        return processed_data

    def calculate_metrics(self, detections, frame_number):
        """Calculate comprehensive detection metrics"""
        metrics = {
            'fps': self.calculate_current_fps(),
            'frame_number': frame_number,
            'total_frames': self.total_frames,
            'processing_time': self.get_processing_time()
        }

        if not self.gt_data or frame_number not in self.gt_data:
            # Add empty metrics for each class when no ground truth
            for class_name in self.class_names:
                metrics[class_name] = self.get_empty_class_metrics()
            return metrics

        current_gt = self.gt_data[frame_number]

        for class_name in self.class_names:
            # Filter detections and ground truth for current class
            class_dets = [d for d in detections if d['class_type'] == class_name]
            class_gt = [g for g in current_gt if g['class'] == class_name]

            # Match detections with ground truth
            matches = self.match_detections(class_dets, class_gt)

            # Calculate metrics
            tp = len(matches)
            fp = len(class_dets) - tp
            fn = len(class_gt) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'support': len(class_gt),
                'detections': len(class_dets)
            }

            metrics[class_name] = class_metrics
            self.frame_metrics[class_name].append(class_metrics)

        return metrics

    def get_empty_class_metrics(self):
        """Return empty metrics structure for a class"""
        return {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'support': 0,
            'detections': 0
        }

    def calculate_current_fps(self):
        """Calculate current FPS based on processing times"""
        if not self.processing_times:
            return 0
        recent_times = self.processing_times[-30:]  # Use last 30 frames
        return 1.0 / np.mean(recent_times) if recent_times else 0

    def get_processing_time(self):
        """Get most recent processing time in milliseconds"""
        return (self.processing_times[-1] * 1000) if self.processing_times else 0

    def match_detections(self, detections, ground_truth, iou_threshold=0.5):
        """Match detections with ground truth using IoU"""
        matches = []
        if not ground_truth or not detections:
            return matches

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(ground_truth)))
        for i, det in enumerate(detections):
            for j, gt in enumerate(ground_truth):
                iou_matrix[i, j] = self.calculate_iou(det['bbox'], gt['bbox'])

        # Find matches using greedy assignment
        while True:
            if not iou_matrix.size:
                break

            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[i, j] < iou_threshold:
                break

            matches.append((detections[i], ground_truth[j]))
            iou_matrix = np.delete(iou_matrix, i, axis=0)
            iou_matrix = np.delete(iou_matrix, j, axis=1)

            if i < len(detections):
                detections.pop(i)
            if j < len(ground_truth):
                ground_truth.pop(j)

        return matches

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def update_memory_usage(self):
        """Update current memory usage"""
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        self.memory_usage.append(memory)
        self.memory_updated.emit(memory)
        return memory

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Could not open video file: {self.video_path}")
                return

            self.running = True
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            start_time = time.time()
            frame_count = self.current_position

            # Set initial position if needed
            if self.current_position > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_position)

            while self.running and cap.isOpened():
                if not self.paused:
                    frame_start = time.time()

                    # Skip frames if needed
                    for _ in range(self.skip_frames - 1):
                        ret, _ = cap.read()
                        frame_count += 1

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    self.current_position = frame_count

                    # Process frame and get detections
                    processed_frame, detections = self.detector.process_frame(frame)
                    self.detection_results.extend(detections)

                    # Calculate processing time and metrics
                    processing_time = time.time() - frame_start
                    self.processing_times.append(processing_time)

                    frame_metrics = self.calculate_metrics(detections, frame_count)

                    # Calculate stats
                    elapsed_time = time.time() - start_time
                    progress = (frame_count / self.total_frames * 100) if self.total_frames > 0 else 0

                    object_counts = defaultdict(int)
                    for det in detections:
                        object_counts[det['class_type']] += 1

                    stats = {
                        'fps': frame_metrics['fps'],
                        'total_objects': len(detections),
                        'progress': progress,
                        'frame_count': frame_count,
                        'total_frames': self.total_frames,
                        'object_counts': dict(object_counts),
                        'video_fps': self.fps,
                        'elapsed_time': elapsed_time,
                        'processing_time': frame_metrics['processing_time'],
                        'estimated_time': self.calculate_estimated_time(
                            frame_count, self.total_frames, frame_metrics['fps']
                        ),
                        'memory_usage': self.update_memory_usage()
                    }

                    # Emit signals
                    self.frame_ready.emit(processed_frame)
                    self.stats_ready.emit(stats)
                    self.metrics_ready.emit(frame_metrics)

                    # Control frame rate
                    target_time = 1.0 / (self.fps * 2)
                    elapsed = time.time() - frame_start
                    if elapsed < target_time:
                        self.msleep(int((target_time - elapsed) * 1000))

            cap.release()
            self.processing_finished.emit()

        except Exception as e:
            self.error_occurred.emit(f"Error processing video: {str(e)}")
        finally:
            self.running = False

    def calculate_estimated_time(self, current_frame, total_frames, current_fps):
        """Calculate estimated remaining time"""
        if current_fps <= 0:
            return "Unknown"
        remaining_frames = total_frames - current_frame
        estimated_seconds = remaining_frames / current_fps
        minutes = int(estimated_seconds // 60)
        seconds = int(estimated_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def set_speed(self, speed_factor):
        """Set video playback speed"""
        self.skip_frames = max(1, int(speed_factor))

    def seek_to_position(self, position_percent):
        """Seek to specific position in video"""
        self.current_position = int((position_percent / 100) * self.total_frames)

    def stop(self):
        """Stop video processing"""
        self.running = False
        self.wait()

    def pause(self):
        """Pause video processing"""
        self.paused = True

    def resume(self):
        """Resume video processing"""
        self.paused = False

    def reset(self):
        """Reset all tracking variables"""
        self.processing_times.clear()
        self.fps_history.clear()
        self.memory_usage.clear()
        self.detection_results.clear()
        self.frame_metrics.clear()
        self.current_position = 0