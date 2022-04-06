import cv2
import numpy as np

import config
from body_detecotr import BodyDetector
from sign_trainer import SignTrainer
from translator import Translator


class LiveVideoTranslator(BodyDetector):

    def __init__(self):
        super().__init__()#cv2.VideoCapture("C:\\Users\\witek\\Downloads\\6265.mp4"))
        self.translator = Translator()


    def start(self, background=True):
        sequence, sentence, predictions = [], [], []
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            while self._video_capture.isOpened():
                detection_results = self._run_detection(holistic)

                sequence.append(BodyDetector.get_body_points(detection_results))
                cv2.imshow('Raw Webcam Feed', self._current_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if (len(sequence) == self.translator.trainer.max_sequence_len): #or (
                        #len(sequence) == self.translator.trainer.min_sequence_len):
                    translation = self.translator.translate(sequence)
                    sequence = []
                    print(translation)
        translation = self.translator.translate(sequence)
        print(translation)
    #TODO: ZAPISAC GESTY W PANDAS JAKO OSOBNO GLOWA RECE ITD ABY MOZNA BYLO POTEM ODTWORZYC NAGRANIE Z PUNKTOW

a = LiveVideoTranslator()
a.start()
x = 2
