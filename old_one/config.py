import os

import numpy as np

BODY_POINTS_FILENAME = "body_points.pkl"
FITED_MODELS_DIR = "FitedModels"
SELECTED_MODEL = "rf"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BODY_POINTS_DIR = os.path.join(ROOT_DIR, "BDDir")
NUM_OF_CAPT_REPEATS = 10
NUM_OF_FRAMES = 90
TEST_SIZE = 0.3
LOG_DIR = "Logs"
MODEL_FILENAME = "SLModel.h5"
THRESHOLD = 0.7
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
EPOCHS = 500