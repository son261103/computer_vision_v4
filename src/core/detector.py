import cv2
import numpy as np
import yaml
import time
import os
from pathlib import Path
from src.models.yolo_detector import YOLOv8Detector
from src.utils.video_stream import VideoStream
from src.utils.visualization import Visualizer
import matplotlib.pyplot as plt
from collections import defaultdict


class Detector:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.model = YOLOv8Detector(config_path)
        self.video_stream = VideoStream(config_path)
        self.visualizer = Visualizer(config_path)
        self.reset_stats()
        self.selected_classes = set()

        # Metrics tracking
        self.metrics = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int),
            'processing_times': [],
            'class_metrics': defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
        }

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config with UTF-8: {e}")
            with open(config_path, 'r', encoding='latin-1') as f:
                return yaml.safe_load(f)

    def setup_directories(self):
        dirs = [
            self.config['paths']['input'],
            self.config['paths']['output'],
            self.config['paths']['weights'],
            os.path.join(self.config['paths']['output'], 'metrics')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def calculate_metrics(self):
        metrics = {}
        for class_type in self.metrics['true_positives'].keys():
            tp = self.metrics['true_positives'][class_type]
            fp = self.metrics['false_positives'][class_type]
            fn = self.metrics['false_negatives'][class_type]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[class_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            # Store metrics history
            self.metrics['class_metrics'][class_type]['precision'].append(precision)
            self.metrics['class_metrics'][class_type]['recall'].append(recall)
            self.metrics['class_metrics'][class_type]['f1'].append(f1)

        if self.metrics['processing_times']:
            avg_time = np.mean(self.metrics['processing_times'])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            metrics['fps'] = fps

        return metrics

    def generate_metrics_visualization(self):
        metrics = self.calculate_metrics()
        output_dir = os.path.join(self.config['paths']['output'], 'metrics')

        # Generate comparison chart
        plt.figure(figsize=(12, 6))
        classes = list(metrics.keys())
        x = np.arange(len(classes))
        width = 0.25

        precisions = [metrics[cls]['precision'] for cls in classes if cls != 'fps']
        recalls = [metrics[cls]['recall'] for cls in classes if cls != 'fps']
        f1_scores = [metrics[cls]['f1_score'] for cls in classes if cls != 'fps']

        plt.bar(x - width, precisions, width, label='Precision')
        plt.bar(x, recalls, width, label='Recall')
        plt.bar(x + width, f1_scores, width, label='F1-Score')

        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Detection Metrics by Class')
        plt.xticks(x, [cls for cls in classes if cls != 'fps'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
        plt.close()

        # Generate metrics table
        with open(os.path.join(output_dir, 'metrics_report.txt'), 'w') as f:
            f.write("Detection Metrics Report\n")
            f.write("=" * 50 + "\n\n")
            if 'fps' in metrics:
                f.write(f"Average FPS: {metrics['fps']:.2f}\n\n")
            f.write("Class-wise Metrics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
            f.write("-" * 50 + "\n")
            for cls in metrics:
                if cls != 'fps':
                    f.write(f"{cls:<15} {metrics[cls]['precision']:>10.2f} "
                            f"{metrics[cls]['recall']:>10.2f} "
                            f"{metrics[cls]['f1_score']:>10.2f}\n")

    def process_video(self, video_path: str, ground_truth_path: str = None):
        try:
            if not self.video_stream.start_stream(video_path):
                raise ValueError(f"Could not open video: {video_path}")

            ground_truth = self._load_ground_truth(ground_truth_path) if ground_truth_path else None

            start_time = time.time()
            self.reset_stats()
            frame_count = 0
            total_frames = self.video_stream.total_frames

            while True:
                frame = self.video_stream.read_frame()
                if frame is None:
                    break

                frame_count += 1
                frame_start_time = time.time()

                # Get ground truth for current frame if available
                current_gt = ground_truth.get(frame_count) if ground_truth else None

                detections, processed_frame = self.model.detect(frame)

                # Update metrics
                if current_gt:
                    self._update_metrics(detections, current_gt)

                processing_time = time.time() - frame_start_time
                self.metrics['processing_times'].append(processing_time)

                self.update_stats(detections)

                # Calculate progress and timing
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Add visualizations
                processed_frame = self.process_frame_visualization(
                    processed_frame,
                    detections,
                    {
                        'progress': progress,
                        'fps': current_fps,
                        'frame_count': frame_count,
                        'total_frames': total_frames,
                        'elapsed_time': elapsed_time,
                        'metrics': self.calculate_metrics()
                    }
                )

                self.video_stream.write_frame(processed_frame)
                cv2.imshow('Traffic Detection System', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Generate final metrics visualization
            self.generate_metrics_visualization()

        except Exception as e:
            print(f"Error processing video: {str(e)}")
        finally:
            self.video_stream.release()
            cv2.destroyAllWindows()

    def _load_ground_truth(self, ground_truth_path):
        try:
            with open(ground_truth_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return None

    def _update_metrics(self, detections, ground_truth, iou_threshold=0.5):
        detected_boxes = defaultdict(list)
        gt_boxes = defaultdict(list)

        for det in detections:
            detected_boxes[det['class_type']].append({
                'bbox': det['bbox'],
                'matched': False
            })

        for gt in ground_truth:
            gt_boxes[gt['class_type']].append({
                'bbox': gt['bbox'],
                'matched': False
            })

        for class_type in detected_boxes.keys():
            for det_box in detected_boxes[class_type]:
                best_iou = iou_threshold
                best_match = None

                for gt_box in gt_boxes[class_type]:
                    if not gt_box['matched']:
                        iou = self._calculate_iou(det_box['bbox'], gt_box['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = gt_box

                if best_match is not None:
                    best_match['matched'] = True
                    det_box['matched'] = True
                    self.metrics['true_positives'][class_type] += 1
                else:
                    self.metrics['false_positives'][class_type] += 1

            for gt_box in gt_boxes[class_type]:
                if not gt_box['matched']:
                    self.metrics['false_negatives'][class_type] += 1

    def _calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def process_frame_visualization(self, frame, detections, stats):
        frame = self.visualizer.draw_detections(frame, detections)

        if self.config['video']['draw_fps']:
            frame = self.visualizer.draw_fps(frame, stats['fps'])

        frame = self.visualizer.draw_progress(
            frame,
            stats['progress'],
            stats['frame_count'],
            stats['total_frames']
        )

        info = self.create_info_dict(stats)
        frame = self.visualizer.create_info_panel(frame, info)

        # Draw metrics if available
        if 'metrics' in stats:
            frame = self.visualizer.draw_metrics(frame, stats['metrics'])

        return frame

    def create_info_dict(self, stats):
        info = {
            'Total Vehicles': self.stats['total_detections'],
            'Frame': f"{stats['frame_count']}/{stats['total_frames']}",
            'FPS': f"{stats['fps']:.1f}",
            'Time': f"{stats['elapsed_time']:.1f}s"
        }

        for class_type, count in self.class_stats.items():
            if count > 0:
                info[class_type] = count

        return info

    def update_stats(self, detections):
        filtered_detections = [
            det for det in detections
            if not self.selected_classes or det['class_type'] in self.selected_classes
        ]

        self.stats['total_detections'] += len(filtered_detections)

        for det in filtered_detections:
            class_type = det['class_type']
            self.class_stats[class_type] = self.class_stats.get(class_type, 0) + 1

    def reset_stats(self):
        self.stats = {'total_detections': 0}
        self.class_stats = {
            'Xe may': 0,
            'Xe dap': 0,
            'o to': 0,
            'Xe buyt': 0,
            'Xe tai': 0,
            'den giao thong': 0,
            'bien bao': 0
        }

        # Reset metrics
        self.metrics = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int),
            'processing_times': [],
            'class_metrics': defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
        }

    def process_frame(self, frame):
        if frame is None:
            return None, []

        try:
            start_time = time.time()
            detections, processed_frame = self.model.detect(frame)

            if self.selected_classes:
                detections = [
                    det for det in detections
                    if det['class_type'] in self.selected_classes
                ]

            self.metrics['processing_times'].append(time.time() - start_time)
            return processed_frame, detections
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame, []

    def get_stats(self):
        return {
            'total': self.stats['total_detections'],
            'by_class': self.class_stats,
            'metrics': self.calculate_metrics()
        }

    def get_video_info(self):
        return self.video_stream.get_video_info()

    def set_confidence_threshold(self, threshold):
        if hasattr(self, 'model'):
            self.model.conf_threshold = threshold