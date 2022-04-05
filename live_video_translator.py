import cv2
import numpy as np

import config
from body_detecotr import BodyDetector
from sign_trainer import SignTrainer
from translator import Translator


class LiveVideoTranslator(BodyDetector):

    def __init__(self):
        super().__init__()
        self.translator = Translator()

    def start(self, background=True):
        sequence, sentence, predictions = [], [], []
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            while self._video_capture.isOpened():
                detection_results = self._run_detection(holistic)
                sequence.append(BodyDetector.get_body_points(detection_results))

                if len(sequence) == self.translator.trainer.max_sequence_len:
                    results = self.translator.translate(sequence)
                a=2



a = LiveVideoTranslator()
a.start()
x=2