import time
from typing import List, Dict, Any

import cv2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
import mediapipe as mp

import config
from body_detecotr import POINTS_NUM
from sign_trainer import SignTrainer
import numpy as np


class Translator:

    def __init__(self):
        self.trainer = SignTrainer()
        # self.model = self.trainer.load_model()
        self.translations = list(self.trainer.label_map)

    def translate_to_eng(self, sequence):
        sequence = self.trainer.pad_sequence(sequence)
        predictions = self.model.predict(np.expand_dims(sequence, axis=0))[0]

        if predictions[np.argmax(predictions)] >= config.THRESHOLD:
            return self.translations[np.argmax(predictions)]
        else:
            return None

    def _action_to_landmarks(self, action_name: str, repeat_nbr: int = 0) -> List[Dict[str, Any]]:
        action_per_frames = self.trainer.data.loc[[action_name]].iloc[[repeat_nbr]].apply(lambda x: x[0])
        landmarks_in_frame = []
        for i, data in action_per_frames.iterrows():
            landmarks_dict = {}
            for key in POINTS_NUM.keys():
                landmark_list = []
                for point in data[key]:
                    # TODO: make it prettier
                    nl = NormalizedLandmark()
                    nl.x = point[0]
                    nl.y = point[1]
                    nl.z = point[2]
                    nl.visibility = point[3]
                    landmark_list.append(nl)
                landmarks_dict[key] = landmark_pb2.NormalizedLandmarkList(landmark=landmark_list)
            landmarks_in_frame.append(landmarks_dict)
        return landmarks_in_frame

    def _display(self, img, landmark, connections):
        # TODO: REFACTOR! THIS CODE EXISTS IN BODY_DETECTOR
        try:
            mp.solutions.drawing_utils.draw_landmarks(img, landmark, connections)
        except ValueError:
            return

    def _display_action(self, action_name: str, repeat_nbr: int = 0):
        action_landmarks = self._action_to_landmarks(action_name, repeat_nbr=repeat_nbr)
        while True:
            for landmarks in action_landmarks:
                img = np.zeros([500, 500, 3], dtype=np.uint8)
                img.fill(255)
                # TODO: REFACTOR! THIS CODE EXISTS IN BODY_DETECTOR
                self._display(img, landmarks["FACE"], mp.solutions.holistic.FACEMESH_TESSELATION)
                self._display(img, landmarks["RIGHT_HAND"], mp.solutions.holistic.HAND_CONNECTIONS)
                self._display(img, landmarks["LEFT_HAND"], mp.solutions.holistic.HAND_CONNECTIONS)
                self._display(img, landmarks["POSE"], mp.solutions.holistic.POSE_CONNECTIONS)

                cv2.imshow("action", img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    return
            time.sleep(1)


# x = Translator()
# x._display_action("all2", 0)
