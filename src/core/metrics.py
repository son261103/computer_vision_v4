import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from pathlib import Path
import time
import json
import cv2


class DetectionMetrics:
    def __init__(self):
        # Class mappings with Vietnamese and English names
        self.class_names = {
            'Xe may': 'Motorcycle',
            'Xe dap': 'Bicycle',
            'o to': 'Car',
            'Xe buyt': 'Bus',
            'Xe tai': 'Truck',
            'den giao thong': 'Traffic Light',
            'bien bao': 'Stop Sign'
        }

        # Initialize metrics storage with enhanced tracking
        self.metrics_history = {
            'precision': defaultdict(list),
            'recall': defaultdict(list),
            'f1': defaultdict(list),
            'fps': [],
            'processing_times': [],
            'class_counts': defaultdict(list),
            'confidence_scores': defaultdict(list)
        }

        # Frame-level tracking
        self.frame_metrics = defaultdict(dict)

        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)
        self.last_time = time.time()

        # Detection tracking
        self.frame_detections = []
        self.ground_truth = []

        # Additional metrics
        self.total_frames = 0
        self.detection_counts = defaultdict(int)
        self.confidence_thresholds = np.arange(0.1, 1.0, 0.1)

    def update(self, frame_number, detections, ground_truth=None, processing_time=None):
        """Update metrics with new frame data"""
        self.total_frames += 1

        # Update FPS and processing time
        if processing_time:
            self.metrics_history['processing_times'].append(processing_time)
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.metrics_history['fps'].append(fps)

        # Calculate frame-level metrics
        frame_metrics = self.calculate_frame_metrics(detections, ground_truth)

        # Store frame results
        self.frame_metrics[frame_number] = frame_metrics

        # Update detection counts
        for det in detections:
            class_type = det['class_type']
            self.detection_counts[class_type] += 1
            self.metrics_history['class_counts'][class_type].append(
                self.detection_counts[class_type]
            )
            self.metrics_history['confidence_scores'][class_type].append(
                det['confidence']
            )

        return frame_metrics

    def calculate_frame_metrics(self, detections, ground_truth, iou_threshold=0.5):
        """Calculate detailed metrics for a single frame"""
        metrics = {}

        for class_name in self.class_names:
            class_dets = [d for d in detections if d['class_type'] == class_name]
            class_gt = [g for g in ground_truth] if ground_truth else []

            # Match detections with ground truth
            matches = self._match_detections(class_dets, class_gt, iou_threshold)

            # Calculate basic metrics
            tp = len(matches)
            fp = len(class_dets) - tp
            fn = len(class_gt) - tp if ground_truth else 0

            # Calculate rates
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Calculate additional metrics
            avg_confidence = np.mean([d['confidence'] for d in class_dets]) if class_dets else 0
            avg_iou = np.mean([self._calculate_iou(m[0]['bbox'], m[1]['bbox']) for m in matches]) if matches else 0

            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'detection_count': len(class_dets),
                'average_confidence': avg_confidence,
                'average_iou': avg_iou
            }

            # Update history
            for metric, value in metrics[class_name].items():
                self.metrics_history[metric][class_name].append(value)

        return metrics

    def generate_evaluation_report(self, output_dir=None):
        """Generate comprehensive evaluation report"""
        report = {
            'overall_metrics': self.calculate_overall_metrics(),
            'class_metrics': self.calculate_class_metrics(),
            'performance_metrics': self.calculate_performance_metrics(),
            'temporal_analysis': self.calculate_temporal_metrics()
        }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save report
            with open(output_dir / 'evaluation_report.json', 'w') as f:
                json.dump(report, f, indent=4)

            # Generate visualizations
            self.plot_metrics_history(output_dir / 'metrics_history.png')
            self.plot_confusion_matrix(output_dir / 'confusion_matrix.png')
            self.plot_class_distribution(output_dir / 'class_distribution.png')
            self.plot_confidence_distribution(output_dir / 'confidence_dist.png')
            self.plot_performance_trends(output_dir / 'performance_trends.png')

        return report

    def calculate_overall_metrics(self):
        """Calculate aggregated metrics across all classes"""
        overall = {
            'total_frames': self.total_frames,
            'total_detections': sum(self.detection_counts.values()),
            'average_fps': np.mean(self.metrics_history['fps']) if self.metrics_history['fps'] else 0,
            'average_processing_time': np.mean(self.metrics_history['processing_times'])
            if self.metrics_history['processing_times'] else 0
        }

        # Calculate macro and weighted averages
        metrics_to_average = ['precision', 'recall', 'f1_score']
        for metric in metrics_to_average:
            values = [np.mean(self.metrics_history[metric][cls])
                      for cls in self.class_names if self.metrics_history[metric][cls]]
            overall[f'macro_avg_{metric}'] = np.mean(values) if values else 0

        return overall

    def calculate_class_metrics(self):
        """Calculate detailed metrics for each class"""
        class_metrics = {}

        for class_name in self.class_names:
            metrics = {}
            for metric in ['precision', 'recall', 'f1_score', 'true_positives',
                           'false_positives', 'false_negatives']:
                values = self.metrics_history[metric].get(class_name, [])
                if values:
                    metrics[f'average_{metric}'] = np.mean(values)
                    metrics[f'std_{metric}'] = np.std(values)
                    metrics[f'max_{metric}'] = np.max(values)
                    metrics[f'min_{metric}'] = np.min(values)

            class_metrics[class_name] = metrics

        return class_metrics

    def calculate_temporal_metrics(self):
        """Analyze temporal patterns in detections"""
        temporal = {}

        for class_name in self.class_names:
            counts = self.metrics_history['class_counts'].get(class_name, [])
            if counts:
                temporal[class_name] = {
                    'detection_rate': len(counts) / self.total_frames,
                    'peak_detections': max(counts),
                    'average_detections_per_frame': np.mean(counts),
                    'detection_variance': np.var(counts)
                }

        return temporal

    def calculate_performance_metrics(self):
        """Calculate detailed performance metrics"""
        performance = {
            'fps_stats': {
                'mean': np.mean(self.metrics_history['fps']),
                'std': np.std(self.metrics_history['fps']),
                'min': np.min(self.metrics_history['fps']),
                'max': np.max(self.metrics_history['fps'])
            },
            'processing_time_stats': {
                'mean_ms': np.mean(self.metrics_history['processing_times']) * 1000,
                'std_ms': np.std(self.metrics_history['processing_times']) * 1000,
                'min_ms': np.min(self.metrics_history['processing_times']) * 1000,
                'max_ms': np.max(self.metrics_history['processing_times']) * 1000
            }
        }

        return performance

    def reset(self):
        """Reset all metrics and history"""
        self.__init__()

    def export_metrics_data(self, output_path):
        """Export metrics data to CSV"""
        data = []
        for frame_num, metrics in self.frame_metrics.items():
            frame_data = {'frame': frame_num}
            for class_name, class_metrics in metrics.items():
                for metric, value in class_metrics.items():
                    frame_data[f"{class_name}_{metric}"] = value
            data.append(frame_data)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return df