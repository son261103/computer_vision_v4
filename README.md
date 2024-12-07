object_detection_system/
│
├── config/
│   └── config.yaml          # Cấu hình chung (model, classes, thresholds)
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── yolo_detector.py # YOLO model cho nhận diện
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── video_stream.py  # Xử lý video 
│   │   └── visualization.py # Vẽ boxes và labels
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── detector.py      # Xử lý chính: load model, detect objects, process results
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py   # Cửa sổ chính
│   │   ├── video_widget.py  # Hiển thị video
│   │   └── styles/
│   │       ├── style.qss    # QSS stylesheet
│   │       └── resources.qrc # Qt resources
│   │
│   └── __init__.py
│
├── input/
│   └── videos/             # Thư mục chứa video đầu vào
│
├── output/
│   └── videos/            # Video đã xử lý
│
├── weights/
│   └── yolo/             # File weights YOLO
│
├── requirements.txt      # Thư viện cần thiết
└── main.py              # File chạy chính


pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install opencv-python
pip install pyyaml
