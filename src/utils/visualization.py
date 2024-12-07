import cv2
import numpy as np
import yaml
from typing import List, Tuple, Dict
from datetime import datetime


class Visualizer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize visualizer with configuration"""
        self.config = self._load_config(config_path)
        self.vis_config = self.config['visualization']
        self.setup_colors()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

    def setup_colors(self):
        """Setup color scheme for visualization"""
        # Default colors from config
        self.class_colors = self.vis_config['colors']

        # UI colors
        self.colors = {
            'background': (45, 45, 45),
            'text': (255, 255, 255),
            'progress': (0, 255, 0),
            'progress_bg': (100, 100, 100),
            'fps': (0, 255, 0),
            'warning': (0, 0, 255),
            'panel_bg': (0, 0, 0)
        }

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection boxes and labels on frame"""
        try:
            frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                class_type = det['class_type']
                confidence = det['confidence']

                # Get color for class
                color = tuple(self.class_colors.get(class_type, [255, 255, 255]))

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                              self.vis_config['box_thickness'])

                # Prepare label
                label = f"{class_type}: {confidence:.2f}"

                # Calculate label dimensions
                (label_w, label_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.vis_config['font_scale'],
                    self.vis_config['text_thickness']
                )

                # Draw label background with transparency
                overlay = frame.copy()
                cv2.rectangle(overlay,
                              (x1, y1 - label_h - 10),
                              (x1 + label_w + 5, y1),
                              color, -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # Draw label text
                cv2.putText(frame, label,
                            (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.vis_config['font_scale'],
                            (255, 255, 255),
                            self.vis_config['text_thickness'])

            return frame

        except Exception as e:
            print(f"Error drawing detections: {e}")
            return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter with background"""
        try:
            if self.config['video']['draw_fps']:
                text = f"FPS: {fps:.1f}"

                # Calculate text dimensions
                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # Draw background
                cv2.rectangle(frame,
                              (5, 5),
                              (text_w + 15, text_h + 15),
                              self.colors['background'],
                              -1)

                # Draw text
                cv2.putText(frame, text,
                            (10, text_h + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, self.colors['fps'], 2)

            return frame

        except Exception as e:
            print(f"Error drawing FPS: {e}")
            return frame

    def draw_progress(self, frame: np.ndarray, progress: float,
                      current_frame: int, total_frames: int) -> np.ndarray:
        """Draw enhanced progress bar with frame counter"""
        try:
            height, width = frame.shape[:2]

            # Progress bar dimensions
            bar_width = int(width * 0.6)
            bar_height = 25
            x = (width - bar_width) // 2
            y = height - 40

            # Draw main background
            cv2.rectangle(frame,
                          (x - 10, y - 10),
                          (x + bar_width + 120, y + bar_height + 10),
                          self.colors['background'],
                          -1)

            # Draw progress bar background
            cv2.rectangle(frame,
                          (x, y),
                          (x + bar_width, y + bar_height),
                          self.colors['progress_bg'],
                          -1)

            # Draw progress
            progress_width = int(bar_width * (progress / 100))
            cv2.rectangle(frame,
                          (x, y),
                          (x + progress_width, y + bar_height),
                          self.colors['progress'],
                          -1)

            # Draw progress text
            progress_text = f"{progress:.1f}%"
            frame_text = f"Frame: {current_frame}/{total_frames}"

            cv2.putText(frame, progress_text,
                        (x + bar_width + 10, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, self.colors['text'], 1)

            cv2.putText(frame, frame_text,
                        (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, self.colors['text'], 1)

            return frame

        except Exception as e:
            print(f"Error drawing progress: {e}")
            return frame

    def create_info_panel(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        """Create information panel with metrics"""
        try:
            # Panel configuration
            panel_height = 100
            padding = 10
            col_width = frame.shape[1] // 3

            # Create panel
            panel = np.full((panel_height, frame.shape[1], 3),
                            self.colors['panel_bg'],
                            dtype=np.uint8)

            # Draw sections
            sections = {
                'Detection': ['Total Objects'],
                'Performance': ['FPS', 'Frame'],
                'Time': ['Time']
            }

            x_pos = padding
            for section, keys in sections.items():
                # Draw section title
                cv2.putText(panel, section,
                            (x_pos, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, self.colors['progress'], 1)

                # Draw section info
                y_pos = 45
                for key in keys:
                    if key in info:
                        text = f"{key}: {info[key]}"
                        cv2.putText(panel, text,
                                    (x_pos, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, self.colors['text'], 1)
                        y_pos += 25

                x_pos += col_width

            # Add vehicle counts
            vehicle_types = ['Xe may', 'o to', 'Xe buyt', 'Xe tai']
            x_pos = padding
            y_pos = 70
            for v_type in vehicle_types:
                if v_type in info:
                    text = f"{v_type}: {info[v_type]}"
                    cv2.putText(panel, text,
                                (x_pos, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, self.colors['text'], 1)
                    x_pos += len(text) * 9 + padding

            # Combine panel with frame
            return np.vstack([panel, frame])

        except Exception as e:
            print(f"Error creating info panel: {e}")
            return frame

    def draw_metrics(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Draw detection metrics"""
        try:
            # Metrics configuration
            metrics_height = 120
            padding = 10
            x_pos = padding
            y_pos = 30

            # Create metrics panel
            metrics_panel = np.full((metrics_height, frame.shape[1], 3),
                                    self.colors['panel_bg'],
                                    dtype=np.uint8)

            # Draw title
            cv2.putText(metrics_panel, "Detection Metrics",
                        (x_pos, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, self.colors['text'], 2)

            # Draw metrics for each class
            for class_name, class_metrics in metrics.items():
                if isinstance(class_metrics, dict):
                    text = (f"{class_name}: P:{class_metrics['precision']:.2f} "
                            f"R:{class_metrics['recall']:.2f} "
                            f"F1:{class_metrics['f1_score']:.2f}")

                    cv2.putText(metrics_panel, text,
                                (x_pos, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, self.colors['text'], 1)

                    y_pos += 20
                    if y_pos >= metrics_height - padding:
                        x_pos += frame.shape[1] // 2
                        y_pos = 30

            # Combine metrics panel with frame
            return np.vstack([metrics_panel, frame])

        except Exception as e:
            print(f"Error drawing metrics: {e}")
            return frame

    def add_timestamp(self, frame: np.ndarray) -> np.ndarray:
        """Add timestamp to frame"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp,
                        (frame.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, self.colors['text'], 1)
            return frame

        except Exception as e:
            print(f"Error adding timestamp: {e}")
            return frame
