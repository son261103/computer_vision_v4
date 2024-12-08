import os
import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import time


class YOLOv8Detector:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.device = self.config['model']['device']
        self.model = self._load_model()
        self.conf_threshold = self.config['model']['confidence_threshold']
        self.input_size = self.config['model']['input_size']
        self.selected_classes = None
        self.update_classes()

        # Metrics tracking
        self.metrics = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int),
            'processing_times': []
        }

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

    def _load_model(self):
        try:
            weights_path = os.path.join(self.config['paths']['weights'], 'yolo11x.pt')
            if os.path.exists(weights_path):
                model = YOLO(weights_path)
            else:
                print("Downloading YOLO model...")
                model = YOLO('yolo11x')
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                model.save(weights_path)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def update_classes(self):
        self.classes = []
        for class_name, class_ids in self.config['classes'].items():
            self.classes.extend(class_ids)
        self.classes = list(set(self.classes))

    def set_selected_classes(self, classes):
        self.selected_classes = set(classes) if classes else None

    def calculate_metrics(self):
        metrics = {}
        for class_type in self.metrics['true_positives'].keys():
            tp = self.metrics['true_positives'][class_type]
            fp = self.metrics['false_positives'][class_type]
            fn = self.metrics['false_negatives'][class_type]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # đọ chính xác
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # độ nhạy
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # tính f1 core

            metrics[class_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

        # Calculate FPS
        if self.metrics['processing_times']:
            avg_time = np.mean(self.metrics['processing_times'])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            metrics['fps'] = fps

        return metrics

    def detect(self, frame, ground_truth=None): # phát hiện đối tượng
        if frame is None:
            return [], None

        try:
            start_time = time.time()

            # Run inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )[0]

            # Process results
            detections = self._process_results(results)

            # Update metrics if ground truth is provided
            if ground_truth is not None:
                self._update_metrics(detections, ground_truth)

            # Calculate processing time
            processing_time = time.time() - start_time
            self.metrics['processing_times'].append(processing_time)

            # Draw detections on frame copy
            processed_frame = self._draw_detections(frame.copy(), detections)

            return detections, processed_frame

        except Exception as e:
            print(f"Error during detection: {e}")
            return [], frame

    def _update_metrics(self, detections, ground_truth, iou_threshold=0.5):
        detected_boxes = defaultdict(list)
        gt_boxes = defaultdict(list)

        # Organize detections by class
        for det in detections:
            detected_boxes[det['class_type']].append({
                'bbox': det['bbox'],
                'matched': False
            })

        # Organize ground truth by class
        for gt in ground_truth:
            gt_boxes[gt['class_type']].append({
                'bbox': gt['bbox'],
                'matched': False
            })

        # Match detections with ground truth
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

            # Count unmatched ground truth as false negatives
            for gt_box in gt_boxes[class_type]:
                if not gt_box['matched']:
                    self.metrics['false_negatives'][class_type] += 1

    def _calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # Calculate intersection coordinates
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        # Calculate areas
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        # Calculate IoU
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def _process_results(self, results): # kết quả trả về từ mô hình YOLO và trích xuất thông tin về các đối tượng phát hiện.
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    confidence = float(box.conf)
                    class_id = int(box.cls)

                    if (self.selected_classes is not None and
                            self._get_class_type(class_id) not in self.selected_classes):
                        continue

                    if class_id not in self.classes:
                        continue

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self._get_class_name(class_id),
                        'class_type': self._get_class_type(class_id)
                    })
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
        return detections

    def _get_class_type(self, class_id):
        types = {
            1: "Xe dap",
            2: "o to",
            3: "Xe may",
            5: "Xe buyt",
            7: "Xe tai",
            9: "den giao thong",
            11: "bien bao"
        }
        return types.get(class_id, "Không xác định")

    def _get_class_name(self, class_id):
        for category, ids in self.config['classes'].items():
            if class_id in ids:
                return category
        return 'unknown'

    def _draw_detections(self, frame, detections):
        vis_config = self.config['visualization']
        stats = {}

        # Draw detections
        for det in detections:
            try:
                x1, y1, x2, y2 = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                class_type = det['class_type']

                color = tuple(vis_config['colors'].get(class_name, [255, 255, 255]))
                stats[class_type] = stats.get(class_type, 0) + 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                              vis_config['box_thickness'])

                label = f"{class_type}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    vis_config['font_scale'],
                    vis_config['text_thickness']
                )

                cv2.rectangle(frame,
                              (x1, y1 - label_h - 10),
                              (x1 + label_w, y1),
                              color, -1)

                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            vis_config['font_scale'],
                            (255, 255, 255),
                            vis_config['text_thickness'])

            except Exception as e:
                print(f"Error drawing detection: {e}")
                continue

        # Draw statistics and metrics
        if self.config['stats']['show_count']:
            self._draw_stats(frame, stats)
            self._draw_metrics(frame)

        return frame

    def _draw_stats(self, frame, stats):
        if not stats:
            return frame

        y_pos = 30
        padding = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Calculate dimensions
        max_width = 0
        total_height = 0
        for obj_type, count in stats.items():
            text = f"{obj_type}: {count}"
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, w)
            total_height += h + padding

        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5),
                      (max_width + 15, total_height + 15),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        y_pos = 25
        for obj_type, count in stats.items():
            text = f"{obj_type}: {count}"
            cv2.putText(frame, text, (10, y_pos),
                        font, font_scale,
                        (255, 255, 255), thickness)
            y_pos += 25

        return frame

    def _draw_metrics(self, frame):
        metrics = self.calculate_metrics()
        if not metrics:
            return frame

        y_pos = 160
        x_pos = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)

        # Draw metrics background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 140),
                      (200, y_pos + len(metrics) * 60),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw FPS
        if 'fps' in metrics:
            cv2.putText(frame, f"FPS: {metrics['fps']:.2f}",
                        (x_pos, y_pos), font, font_scale, color, thickness)
            y_pos += 20

        # Draw class metrics
        for class_type, class_metrics in metrics.items():
            if class_type != 'fps':
                cv2.putText(frame, f"{class_type}:", (x_pos, y_pos),
                            font, font_scale, color, thickness)
                y_pos += 15
                cv2.putText(frame,
                            f"P:{class_metrics['precision']:.2f} "
                            f"R:{class_metrics['recall']:.2f} "
                            f"F1:{class_metrics['f1_score']:.2f}",
                            (x_pos + 10, y_pos), font, font_scale, color, thickness)
                y_pos += 25

        return frame

    def set_confidence(self, confidence):
        self.conf_threshold = confidence

    def __call__(self, frame, ground_truth=None):
        return self.detect(frame, ground_truth)