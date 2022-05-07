import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSOR_DIR = os.path.join(ROOT_DIR, "tensor_dir")
# TENSOR_DIR = os.path.join(ROOT_DIR, "tensor_dir_tanh")
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
EPOCHS = 100
THRESHOLD = 0.7
