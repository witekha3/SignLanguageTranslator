import logging
import time
from typing import List, NamedTuple, Optional

import cv2
import pandas as pd
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

import config
from body_detector import POINTS_NUM, BodyDetector
from sign_trainer import SignTrainer
import numpy as np


class DetectionResults(NamedTuple):
    """Represents 'SolutionOutputs' from Mediapipe"""
    face_landmarks: NormalizedLandmark
    left_hand_landmarks: NormalizedLandmark
    right_hand_landmarks: NormalizedLandmark
    pose_landmarks: NormalizedLandmark


class Translator:
    """
    Class responsible for making translations
    """

    def __init__(self):
        self.trainer = SignTrainer()
        self._model = self.trainer.load_model()
        self.translations = BodyDetector.get_all_actions_names()
        logging.debug(f"Available actions: {self.translations}")

    def asl_to_eng(self, sequence: pd.DataFrame) -> str:
        """
        Translates ASL into english
        :param sequence: sequence to translate
        :return: Translation of the sequence
        """
        sequence["ACTION"] = "ACTION"
        sequence = sequence.groupby("ACTION").agg(list)
        sequence = BodyDetector.flatten_action(sequence.iloc[0])
        sequence = self.trainer.pad_sequence(sequence)
        predictions = self._model.predict(np.expand_dims(sequence, axis=0))[0]
        translation = self.translations[np.argmax(predictions)]

        if predictions[np.argmax(predictions)] >= config.THRESHOLD:
            return translation
        # else:
            # logging.debug(f"Max threshold: {self.translations[np.argmax(predictions)]} - {predictions[np.argmax(predictions)]}")
        return ""

    @staticmethod
    def _action_to_landmarks(action_name: str, repeat_nbr: int = 0) -> List[NamedTuple]:
        """
        Retrieves data from a saved action and returns it as mediapipe landmarks
        :param action_name: action to read
        :param repeat_nbr: action repeat number
        :return: List of Landmarks per frame
        """
        action_per_frames = BodyDetector.get_points(action_name, repeat_nbr).apply(lambda x: x[0])
        landmarks_in_repeat = []
        for i, data in action_per_frames.iterrows():
            landmarks_dict = {}
            for key in POINTS_NUM.keys():
                landmark_list = []
                for point in data[key]:
                    nl = NormalizedLandmark()
                    nl.x, nl.y, nl.z, nl.visibility = point
                    nl.visibility = 1
                    landmark_list.append(nl)
                landmarks_dict[f"{key.lower()}_landmarks"] = landmark_pb2.NormalizedLandmarkList(landmark=landmark_list)
            landmarks_in_repeat.append(DetectionResults(**landmarks_dict))
        return landmarks_in_repeat

    @staticmethod
    def english_to_asl(action_name: str, repeat_nbr: Optional[int] = 0) -> None:
        """
        Converts english word into ASL
        :param action_name: word in english
        :param repeat_nbr: Optional -Video repeat number. Default = 0
        :return: None
        """
        action_landmarks = Translator._action_to_landmarks(action_name, repeat_nbr=repeat_nbr)
        while True:
            for results in action_landmarks:
                img = np.zeros([500, 500, 3], dtype=np.uint8)
                img.fill(255)
                BodyDetector.draw_all_body_points(results, img)
                cv2.imshow("action", img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    return
            time.sleep(1)

