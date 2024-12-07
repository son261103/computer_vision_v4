from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtCharts import QChartView, QChart, QLineSeries, QValueAxis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import os


class EvaluationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_history = {}
        self.current_frame = 0
        self.setup_ui()

    def setup_ui(self):
        """Setup the evaluation tab UI with comprehensive metrics display"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Performance Metrics Section with expanded metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QHBoxLayout(perf_group)

        # Left column
        left_metrics = QVBoxLayout()
        self.fps_label = QLabel("FPS: 0")
        self.proc_time_label = QLabel("Processing Time: 0 ms")
        self.total_frames_label = QLabel("Total Frames: 0")
        self.current_frame_label = QLabel("Current Frame: 0")

        for label in [self.fps_label, self.proc_time_label,
                      self.total_frames_label, self.current_frame_label]:
            left_metrics.addWidget(label)

        # Right column - new metrics
        right_metrics = QVBoxLayout()
        self.total_detections_label = QLabel("Total Detections: 0")
        self.avg_confidence_label = QLabel("Average Confidence: 0.00")
        self.avg_iou_label = QLabel("Average IoU: 0.00")
        self.detection_rate_label = QLabel("Detection Rate: 0 obj/frame")

        for label in [self.total_detections_label, self.avg_confidence_label,
                      self.avg_iou_label, self.detection_rate_label]:
            right_metrics.addWidget(label)

        perf_layout.addLayout(left_metrics)
        perf_layout.addLayout(right_metrics)
        layout.addWidget(perf_group)

        # Detection Metrics Section with expanded table
        metrics_group = QGroupBox("Detection Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_table = QTableWidget()
        self.setup_metrics_table()
        metrics_layout.addWidget(self.metrics_table)

        # Add progress bar for detection accuracy
        self.accuracy_progress = QProgressBar()
        self.accuracy_progress.setTextVisible(True)
        self.accuracy_progress.setFormat("Overall Accuracy: %p%")
        metrics_layout.addWidget(self.accuracy_progress)

        # Real-time metrics chart with more series
        self.setup_realtime_chart()
        metrics_layout.addWidget(self.metrics_plot)

        layout.addWidget(metrics_group)

        # Enhanced Visualization Section
        vis_group = QGroupBox("Visualization")
        vis_layout = QHBoxLayout(vis_group)

        # Metrics history plot
        self.metrics_figure = plt.figure(figsize=(6, 4))
        self.metrics_canvas = FigureCanvasQTAgg(self.metrics_figure)
        vis_layout.addWidget(self.metrics_canvas)

        # Confusion matrix plot
        self.confusion_figure = plt.figure(figsize=(6, 4))
        self.confusion_canvas = FigureCanvasQTAgg(self.confusion_figure)
        vis_layout.addWidget(self.confusion_canvas)

        layout.addWidget(vis_group)

        # Enhanced Controls Section
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)

        # Class filter with additional options
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("All Classes")
        self.class_filter_combo.currentTextChanged.connect(self.filter_metrics_display)

        # Metric type filter
        self.metric_filter_combo = QComboBox()
        self.metric_filter_combo.addItems(["All Metrics", "Basic", "Advanced"])
        self.metric_filter_combo.currentTextChanged.connect(self.filter_metrics_display)

        # Export and reset buttons
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_report)
        self.reset_btn = QPushButton("Reset Statistics")
        self.reset_btn.clicked.connect(self.reset_statistics)

        for widget in [self.class_filter_combo, self.metric_filter_combo,
                       self.export_btn, self.reset_btn]:
            controls_layout.addWidget(widget)

        layout.addWidget(controls_group)
        self.style_components()

    def setup_metrics_table(self):
        """Setup metrics table with comprehensive columns"""
        headers = [
            "Class",
            "Precision",
            "Recall",
            "F1-Score",
            "True Positives",
            "False Positives",
            "False Negatives",
            "Detection Count",
            "Avg Confidence",
            "Avg IoU",
            "Detection Rate",
            "Miss Rate",
            "MOTA",
            "MOTP"
        ]
        self.metrics_table.setColumnCount(len(headers))
        self.metrics_table.setHorizontalHeaderLabels(headers)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)

    def setup_realtime_chart(self):
        """Setup real-time metrics chart"""
        self.metrics_plot = QChartView()
        self.metrics_plot.setMinimumHeight(200)

        chart = QChart()
        chart.setTitle("Real-time Metrics")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)

        # Setup axes
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("Frame")
        self.axis_x.setRange(0, 100)

        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("Score")
        self.axis_y.setRange(0, 1)

        chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)

        self.metrics_series = {}
        self.metrics_plot.setChart(chart)

    def update_metrics(self, metrics_data):
        """Update all metrics displays with comprehensive data"""
        if not metrics_data:
            return

        self.current_frame += 1
        self.current_frame_label.setText(f"Current Frame: {self.current_frame}")

        # Update basic performance metrics
        if 'fps' in metrics_data:
            fps = metrics_data['fps']
            self.fps_label.setText(f"FPS: {fps:.2f}")
            proc_time = 1000 / fps if fps > 0 else 0
            self.proc_time_label.setText(f"Processing Time: {proc_time:.1f} ms")

        # Calculate and update additional metrics
        total_detections = 0
        total_confidence = 0
        total_iou = 0
        detection_count = 0

        classes = [name for name in metrics_data.keys() if isinstance(metrics_data[name], dict)]
        self.metrics_table.setRowCount(len(classes))

        for class_name in classes:
            data = metrics_data[class_name]

            # Accumulate totals
            total_detections += data.get('detection_count', 0)
            if 'average_confidence' in data:
                total_confidence += data['average_confidence']
                detection_count += 1
            if 'average_iou' in data:
                total_iou += data['average_iou']

        # Update summary labels
        self.total_detections_label.setText(f"Total Detections: {total_detections}")
        if detection_count > 0:
            avg_conf = total_confidence / detection_count
            self.avg_confidence_label.setText(f"Average Confidence: {avg_conf:.2f}")
            avg_iou = total_iou / detection_count
            self.avg_iou_label.setText(f"Average IoU: {avg_iou:.2f}")

        detection_rate = total_detections / self.current_frame if self.current_frame > 0 else 0
        self.detection_rate_label.setText(f"Detection Rate: {detection_rate:.2f} obj/frame")

        # Update accuracy progress bar
        if detection_count > 0:
            overall_accuracy = (avg_conf + avg_iou) / 2 * 100
            self.accuracy_progress.setValue(int(overall_accuracy))

        # Update table and visualizations
        for row, class_name in enumerate(classes):
            data = metrics_data[class_name]

            if class_name not in self.metrics_history:
                self.metrics_history[class_name] = self.initialize_metrics_history()
                if class_name not in [self.class_filter_combo.itemText(i)
                                      for i in range(self.class_filter_combo.count())]:
                    self.class_filter_combo.addItem(class_name)

            self.update_metrics_history(self.metrics_history[class_name], data)
            self.update_table_row(row, class_name, data)

        # Update visualizations
        self.update_realtime_chart(metrics_data)
        self.update_history_plot()
        self.update_confusion_matrix(metrics_data)

    def update_metrics_history(self, history, data):
        """Update metrics history for a class"""
        for metric in ['precision', 'recall', 'f1']:
            value = data[f'{metric}_score' if metric == 'f1' else metric]
            history[metric].append(value)

        for metric in ['true_positives', 'false_positives', 'false_negatives']:
            history[metric].append(data.get(metric, 0))

    def update_table_row(self, row, class_name, data):
        """Update table row with comprehensive metrics"""
        # Calculate additional metrics
        detection_rate = data.get('detection_count', 0) / self.current_frame if self.current_frame > 0 else 0
        miss_rate = data.get('false_negatives', 0) / (data.get('false_negatives', 0) + data.get('true_positives', 1))

        items = [
            class_name,
            f"{data.get('precision', 0):.3f}",
            f"{data.get('recall', 0):.3f}",
            f"{data.get('f1_score', 0):.3f}",
            str(data.get('true_positives', 0)),
            str(data.get('false_positives', 0)),
            str(data.get('false_negatives', 0)),
            str(data.get('detection_count', 0)),
            f"{data.get('average_confidence', 0):.3f}",
            f"{data.get('average_iou', 0):.3f}",
            f"{detection_rate:.3f}",
            f"{miss_rate:.3f}",
            f"{data.get('mota', 0):.3f}",
            f"{data.get('motp', 0):.3f}"
        ]

        for col, item in enumerate(items):
            table_item = QTableWidgetItem(item)
            table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.metrics_table.setItem(row, col, table_item)

    def update_realtime_chart(self, metrics_data):
        """Update real-time metrics chart"""
        chart = self.metrics_plot.chart()
        selected_class = self.class_filter_combo.currentText()

        # Update x-axis range if needed
        if self.current_frame > self.axis_x.max():
            self.axis_x.setRange(0, self.current_frame + 50)

        for class_name, data in metrics_data.items():
            if class_name == 'fps' or (selected_class != "All Classes"
                                       and class_name != selected_class):
                continue

            for metric in ['precision', 'recall', 'f1_score']:
                series_name = f"{class_name}_{metric}"

                # Create or get series
                if series_name not in self.metrics_series:
                    series = QLineSeries()
                    series.setName(f"{class_name} {metric}")
                    chart.addSeries(series)
                    series.attachAxis(self.axis_x)
                    series.attachAxis(self.axis_y)
                    self.metrics_series[series_name] = series

                # Add new point
                self.metrics_series[series_name].append(
                    self.current_frame,
                    data[metric]
                )

    def update_history_plot(self):
        """Update metrics history plot with proper legend and layout"""
        self.metrics_figure.clear()

        # Create figure with adjusted size and spacing
        self.metrics_figure.set_size_inches(8, 5)
        ax = self.metrics_figure.add_subplot(111)
        ax.set_position([0.1, 0.1, 0.6, 0.8])  # [left, bottom, width, height]

        selected_class = self.class_filter_combo.currentText()
        selected_metric = self.metric_filter_combo.currentText()
        has_data = False

        metrics_to_plot = {
            'precision': {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
            'recall': {'color': 'red', 'linestyle': '--', 'marker': 's'},
            'f1': {'color': 'green', 'linestyle': ':', 'marker': '^'},
            'average_confidence': {'color': 'purple', 'linestyle': '-.', 'marker': 'D'},
            'average_iou': {'color': 'orange', 'linestyle': '--', 'marker': 'v'}
        }

        for class_name, history in self.metrics_history.items():
            if selected_class != "All Classes" and class_name != selected_class:
                continue

            frames = range(len(history['precision']))
            if len(frames) == 0:
                continue

            has_data = True
            for metric, style in metrics_to_plot.items():
                if metric in history and len(history[metric]) > 0:
                    if selected_metric == "Basic" and metric in ['average_confidence', 'average_iou']:
                        continue
                    if selected_metric == "Advanced" and metric in ['precision', 'recall', 'f1']:
                        continue
                    if selected_metric != "All Metrics" and selected_metric not in ["Basic", "Advanced"]:
                        if metric not in ['precision', 'recall', 'f1']:
                            continue

                    ax.plot(frames, history[metric],
                            label=f'{class_name} {metric}',
                            color=style['color'],
                            linestyle=style['linestyle'],
                            marker=style['marker'],
                            markersize=4,
                            markevery=max(1, len(frames) // 20))

        if has_data:
            ax.set_xlabel('Frame')
            ax.set_ylabel('Score')
            ax.set_title('Metrics History')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim([-0.05, 1.05])

            # Create legend outside the plot
            legend = ax.legend(bbox_to_anchor=(1.05, 1.0),
                               loc='upper left',
                               title='Metrics',
                               fontsize='small')
            legend.get_title().set_fontsize('small')

            # Adjust layout with tight_layout
            try:
                self.metrics_figure.tight_layout()
            except ValueError:
                # If tight_layout fails, use a fixed layout
                self.metrics_figure.subplots_adjust(right=0.85)

        self.metrics_canvas.draw()

    def update_confusion_matrix(self, metrics_data):
        """Update confusion matrix visualization with improved layout"""
        if not metrics_data:
            return

        self.confusion_figure.clear()

        # Create figure with adjusted size
        self.confusion_figure.set_size_inches(6, 5)
        ax = self.confusion_figure.add_subplot(111)

        # Filter out non-metric data
        classes = [name for name in metrics_data.keys()
                   if isinstance(metrics_data[name], dict)]

        if not classes:
            return

        matrix_size = len(classes)
        conf_matrix = np.zeros((matrix_size, matrix_size))

        # Create confusion matrix
        for i, true_class in enumerate(classes):
            if true_class in metrics_data:
                data = metrics_data[true_class]
                conf_matrix[i, i] = data.get('true_positives', 0)
                total_false_positives = data.get('false_positives', 0)
                # Distribute false positives evenly among other classes
                for j in range(matrix_size):
                    if i != j:
                        conf_matrix[i, j] = total_false_positives / (matrix_size - 1)

        # Create heatmap with improved formatting
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='.0f',
                    cmap='Blues',
                    xticklabels=classes,
                    yticklabels=classes,
                    ax=ax,
                    cbar_kws={'label': 'Count'})

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(),
                 rotation=45,
                 ha='right',
                 rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(),
                 rotation=0)

        # Adjust layout
        self.confusion_figure.tight_layout()
        self.confusion_canvas.draw()

    def filter_metrics_display(self, selected_option):
        """Filter displayed metrics based on selected class and metric type"""
        # Update history plot
        self.update_history_plot()

        # Reset and update realtime chart
        chart = self.metrics_plot.chart()
        chart.removeAllSeries()
        self.metrics_series.clear()

        # Update table based on selected metric type
        selected_metric = self.metric_filter_combo.currentText()
        if selected_metric != "All Metrics":
            visible_columns = {
                "Basic": ["Class", "Precision", "Recall", "F1-Score",
                          "True Positives", "False Positives", "False Negatives"],
                "Advanced": ["Class", "Detection Count", "Avg Confidence", "Avg IoU",
                             "Detection Rate", "Miss Rate", "MOTA", "MOTP"]
            }

            # Show/hide columns based on selection
            for col in range(self.metrics_table.columnCount()):
                header = self.metrics_table.horizontalHeaderItem(col).text()
                is_visible = header in visible_columns.get(selected_metric, [])
                self.metrics_table.setColumnHidden(col, not is_visible)
        else:
            # Show all columns
            for col in range(self.metrics_table.columnCount()):
                self.metrics_table.setColumnHidden(col, False)

        # Update performance metrics display
        selected_class = self.class_filter_combo.currentText()
        if selected_class != "All Classes":
            # Update performance metrics for selected class only
            if selected_class in self.metrics_history:
                history = self.metrics_history[selected_class]
                if history['detection_count']:
                    self.total_detections_label.setText(
                        f"Total Detections: {sum(history['detection_count'])}")
                    if history['average_confidence']:
                        self.avg_confidence_label.setText(
                            f"Average Confidence: {np.mean(history['average_confidence']):.2f}")
                    if history['average_iou']:
                        self.avg_iou_label.setText(
                            f"Average IoU: {np.mean(history['average_iou']):.2f}")

    def export_report(self):
        """Export evaluation report"""
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory"
        )
        if not save_dir:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_dir = os.path.join(save_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)

            # Export data
            metrics_df = self.get_metrics_dataframe()
            metrics_df.to_csv(os.path.join(metrics_dir, f"metrics_{timestamp}.csv"))

            # Export visualizations
            self.metrics_figure.savefig(
                os.path.join(metrics_dir, f"metrics_history_{timestamp}.png"))
            self.confusion_figure.savefig(
                os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png"))

            QMessageBox.information(
                self, "Success",
                f"Report exported successfully to:\n{metrics_dir}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error exporting report: {str(e)}"
            )

    def get_metrics_dataframe(self):
        """Convert metrics table to DataFrame"""
        rows = self.metrics_table.rowCount()
        cols = self.metrics_table.columnCount()
        data = []

        for row in range(rows):
            row_data = []
            for col in range(cols):
                item = self.metrics_table.item(row, col)
                row_data.append(item.text() if item else '')
            data.append(row_data)

        columns = [
            self.metrics_table.horizontalHeaderItem(i).text()
            for i in range(cols)
        ]
        return pd.DataFrame(data, columns=columns)

    def reset_statistics(self):
        """Reset all metrics and visualizations"""
        # Reset counters
        self.current_frame = 0
        self.metrics_history.clear()

        # Reset performance labels
        self.fps_label.setText("FPS: 0")
        self.proc_time_label.setText("Processing Time: 0 ms")
        self.total_frames_label.setText("Total Frames: 0")
        self.current_frame_label.setText("Current Frame: 0")

        # Reset metrics table
        self.metrics_table.clearContents()
        self.metrics_table.setRowCount(0)

        # Reset charts
        chart = self.metrics_plot.chart()
        for series in self.metrics_series.values():
            chart.removeSeries(series)
        self.metrics_series.clear()
        self.axis_x.setRange(0, 100)

        # Reset plots
        self.metrics_figure.clear()
        self.confusion_figure.clear()
        self.metrics_canvas.draw()
        self.confusion_canvas.draw()

        # Reset class filter
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("All Classes")

        # Display reset message
        QMessageBox.information(self, "Reset", "All statistics have been reset successfully")

    def style_components(self):
        """Apply styling to all UI components"""
        self.setStyleSheet("""
                    QGroupBox {
                        border: 1px solid #ff69b4;
                        border-radius: 5px;
                        margin-top: 1em;
                        padding-top: 1em;
                        background-color: #2d2d2d;
                    }
                    QGroupBox::title {
                        color: white;
                        subcontrol-origin: margin;
                        padding: 0 5px;
                        background-color: #2d2d2d;
                    }
                    QTableWidget {
                        background-color: #2d2d2d;
                        color: white;
                        gridline-color: #ff69b4;
                        border: none;
                    }
                    QHeaderView::section {
                        background-color: #ff69b4;
                        color: white;
                        padding: 5px;
                        border: 1px solid #2d2d2d;
                    }
                    QPushButton {
                        background-color: #ff69b4;
                        color: white;
                        border: none;
                        padding: 8px;
                        border-radius: 4px;
                        min-width: 100px;
                    }
                    QPushButton:hover {
                        background-color: #ff1493;
                    }
                    QPushButton:pressed {
                        background-color: #c71585;
                    }
                    QLabel {
                        color: white;
                        font-size: 12px;
                        padding: 5px;
                        background-color: #1e1e1e;
                        border-radius: 3px;
                    }
                    QComboBox {
                        background-color: #2d2d2d;
                        color: white;
                        border: 1px solid #ff69b4;
                        border-radius: 3px;
                        padding: 5px;
                        min-width: 150px;
                    }
                    QComboBox::drop-down {
                        border: none;
                    }
                    QComboBox::down-arrow {
                        image: none;
                        border-left: 5px solid transparent;
                        border-right: 5px solid transparent;
                        border-top: 5px solid #ff69b4;
                        width: 0;
                        height: 0;
                        margin-right: 5px;
                    }
                    QComboBox:on {
                        border: 2px solid #ff1493;
                    }
                    QComboBox QAbstractItemView {
                        background-color: #2d2d2d;
                        color: white;
                        selection-background-color: #ff69b4;
                        selection-color: white;
                        border: 1px solid #ff69b4;
                    }
                    QScrollBar:horizontal {
                        height: 12px;
                        background: #2d2d2d;
                        border-radius: 6px;
                    }
                    QScrollBar::handle:horizontal {
                        background: #ff69b4;
                        border-radius: 5px;
                        min-width: 20px;
                    }
                    QScrollBar::add-line:horizontal,
                    QScrollBar::sub-line:horizontal {
                        width: 0;
                        height: 0;
                    }
                    QScrollBar:vertical {
                        width: 12px;
                        background: #2d2d2d;
                        border-radius: 6px;
                    }
                    QScrollBar::handle:vertical {
                        background: #ff69b4;
                        border-radius: 5px;
                        min-height: 20px;
                    }
                    QScrollBar::add-line:vertical,
                    QScrollBar::sub-line:vertical {
                        width: 0;
                        height: 0;
                    }
                """)

    def generate_metrics_plots(self, metrics_dir, timestamp):
        """Generate additional metrics visualizations"""
        # Generate precision-recall curves
        plt.figure(figsize=(10, 6))
        for class_name, history in self.metrics_history.items():
            precision = history['precision']
            recall = history['recall']
            plt.plot(recall, precision, label=f'{class_name}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(metrics_dir, f'pr_curves_{timestamp}.png'))
        plt.close()

        # Generate F1 score trends
        plt.figure(figsize=(10, 6))
        for class_name, history in self.metrics_history.items():
            frames = range(len(history['f1']))
            plt.plot(frames, history['f1'], label=f'{class_name}')

        plt.xlabel('Frame')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Trends')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(metrics_dir, f'f1_trends_{timestamp}.png'))
        plt.close()

    def generate_detailed_report(self, metrics_dir, timestamp):
        """Generate detailed metrics report"""
        report_path = os.path.join(metrics_dir, f'detailed_report_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("Detailed Detection Metrics Report\n")
            f.write("=" * 50 + "\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Frames Processed: {self.current_frame}\n")
            f.write(f"Average FPS: {self.fps_label.text().split(': ')[1]}\n")
            f.write(f"Average Processing Time: {self.proc_time_label.text().split(': ')[1]}\n\n")

            # Class-wise metrics
            f.write("Class-wise Metrics:\n")
            f.write("-" * 30 + "\n")

            metrics_df = self.get_metrics_dataframe()
            f.write(metrics_df.to_string())
            f.write("\n\n")

            # Historical trends
            f.write("Metric Trends:\n")
            f.write("-" * 30 + "\n")
            for class_name, history in self.metrics_history.items():
                f.write(f"\n{class_name}:\n")
                f.write(f"Average Precision: {np.mean(history['precision']):.3f}\n")
                f.write(f"Average Recall: {np.mean(history['recall']):.3f}\n")
                f.write(f"Average F1 Score: {np.mean(history['f1']):.3f}\n")
                f.write(f"Total True Positives: {sum(history['true_positives'])}\n")
                f.write(f"Total False Positives: {sum(history['false_positives'])}\n")
                f.write(f"Total False Negatives: {sum(history['false_negatives'])}\n")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_R and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.reset_statistics()
        elif event.key() == Qt.Key.Key_E and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.export_report()
        super().keyPressEvent(event)

    def initialize_metrics_history(self):
        """Initialize comprehensive metrics history structure"""
        return {
            'precision': [],
            'recall': [],
            'f1': [],
            'true_positives': [],
            'false_positives': [],
            'false_negatives': [],
            'detection_count': [],
            'average_confidence': [],
            'average_iou': [],
            'detection_rate': [],
            'miss_rate': [],
            'mota': [],
            'motp': []
        }