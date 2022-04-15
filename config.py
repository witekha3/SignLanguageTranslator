import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSOR_DIR = os.path.join(ROOT_DIR, "tensor_dir")
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
EPOCHS = 1000
THRESHOLD = 0.7
