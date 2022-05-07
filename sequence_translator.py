import logging

import cv2
import pandas as pd

import config
from body_detector import BodyDetector
from translator import Translator
from mediapipe.python.solutions import holistic as holistic_model


class LiveVideoTranslator(BodyDetector):
    """
    Class responsible for real-time gesture detection
    """

    def __init__(self):
        super().__init__()
        self.translator = Translator()

    def start(self):
        """
        Runs real-time gesture detection.
        If you have finished showing the gesture, press 'q' to get the translation.
        """
        sequence = pd.DataFrame()
        with holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                     min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            while self._video_capture.isOpened():
                detection_results = self._run_detection(holistic)

                sequence = sequence.append(BodyDetector.get_body_points(detection_results), ignore_index=True)
                cv2.imshow('Raw Webcam Feed', self._current_image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    translation = self.translator.asl_to_eng(sequence)
                    logging.info(translation)
                    sequence = pd.DataFrame()

                if len(sequence) == self.translator.trainer.max_sequence_len:
                    translation = self.translator.asl_to_eng(sequence)
                    sequence = pd.DataFrame()
                    logging.info(translation)


class SequenceRecognizer(BodyDetector):
    """
    The class responsible for translating the gesture from the video
    """

    def __init__(self, video_path):
        super().__init__(cv2.VideoCapture(video_path))
        self.translator = Translator()

    def start(self):
        """Runs the translation"""
        sequence = pd.DataFrame()
        with holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                     min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            while self._video_capture.isOpened():
                detection_results = self._run_detection(holistic)
                if not detection_results:
                    break
                sequence = sequence.append(BodyDetector.get_body_points(detection_results), ignore_index=True)
                cv2.imshow('Raw Webcam Feed', self._current_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if len(sequence) == self.translator.trainer.max_sequence_len:
                    translation = self.translator.asl_to_eng(sequence)
                    sequence = pd.DataFrame()
                    logging.info(translation)
        translation = self.translator.asl_to_eng(sequence)
        logging.info(translation)
