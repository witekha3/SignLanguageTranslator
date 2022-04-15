import cv2
import numpy as np
import pandas as pd

import config
from body_detecotr import BodyDetector
from sign_trainer import SignTrainer
from translator import Translator


class LiveVideoTranslator(BodyDetector):

    def __init__(self):
        super().__init__()#cv2.VideoCapture("C:\\Users\\witek\\Downloads\\6265.mp4"))
        self.translator = Translator()


    def start(self, background=True):
        sentence, predictions = [], []
        sequence = pd.DataFrame()
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            while self._video_capture.isOpened():
                detection_results = self._run_detection(holistic)

                sequence = sequence.append(BodyDetector.get_body_points(detection_results), ignore_index=True)
                cv2.imshow('Raw Webcam Feed', self._current_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if (len(sequence) == self.translator.trainer.max_sequence_len): #or (
                        #len(sequence) == self.translator.trainer.min_sequence_len):
                    translation = self.translator.translate_to_eng(sequence)
                    sequence = pd.DataFrame()
                    print(translation)
        translation = self.translator.translate_to_eng(sequence)
        print(translation)
    #TODO: ZAPISAC GESTY W PANDAS JAKO OSOBNO GLOWA RECE ITD ABY MOZNA BYLO POTEM ODTWORZYC NAGRANIE Z PUNKTOW

# a = LiveVideoTranslator()
# a.start()
# x = 2


class SequenceRecognizer(BodyDetector):

    def __init__(self, video_path):
        super().__init__(cv2.VideoCapture(video_path))
        self.translator = Translator()

    def start(self):
        sequence = pd.DataFrame()
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            while self._video_capture.isOpened():
                detection_results = self._run_detection(holistic)
                if not detection_results:
                    break
                sequence = sequence.append(BodyDetector.get_body_points(detection_results), ignore_index=True)
                cv2.imshow('Raw Webcam Feed', self._current_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if (len(sequence) == self.translator.trainer.max_sequence_len): #or (
                        #len(sequence) == self.translator.trainer.min_sequence_len):
                    translation = self.translator.translate_to_eng(sequence)
                    sequence = pd.DataFrame()
                    print(translation)
        translation = self.translator.translate_to_eng(sequence)
        print(translation)

# SequenceRecognizer(r"C:\Users\witek\Downloads\22937.mp4").start()