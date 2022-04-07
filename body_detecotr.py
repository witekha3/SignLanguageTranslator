import logging
import os
from typing import Tuple, Optional, List, NamedTuple, Dict

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from cv2 import VideoCapture
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions.holistic import Holistic
from numpy import ndarray

import config

logging.basicConfig(level=logging.DEBUG)

"""
A dictionary storing the number of all points that mediapipe finds for each body part
"""
POINTS_NUM = {
    "POSE": 132,
    "FACE": 1404,
    "LEFT_HAND": 63,
    "RIGHT_HAND": 63,
}


class BodyDetector:
    """
    Class responsible for detecting and handling landmarks on an actor's body

    :arg video_capture: VideoCapture from cv2
    """

    def __init__(self, video_capture: VideoCapture = None):
        self._drawing = mp.solutions.drawing_utils  # Drawing helpers
        self._video_capture = video_capture if video_capture else cv2.VideoCapture(0)
        self._holistic_model = mp.solutions.holistic
        self._current_image = None
        self._black_image = None

    def _draw_landmarks(
            self,
            landmark: NormalizedLandmarkList,
            connections: Optional[List[Tuple[int, int]]],
            color: Optional[Tuple] = (80, 110, 10),
            thickness: Optional[int] = 1,
            circle_radius: Optional[int] = 1
    ) -> None:
        """
        Draws landmarks on a image
        :param landmark: A normalized landmark list proto message to be annotated on the image
        :param connections: A list of landmark index tuples that specifies how landmarks to be connected in the drawing
        :param color: Landmarks color (Default =  (80, 110, 10))
        :param thickness: Landmarks thickness (Default = 1)
        :param circle_radius: Landmarks radius (Default = 1)
        :return: None
        """
        self._drawing.draw_landmarks(self._current_image, landmark, connections,
                                     self._drawing.DrawingSpec(color, thickness, circle_radius))

    def _run_detection(self, model: Holistic) -> NamedTuple:
        """
        Runs landmark detection
        :param model: MediaPipe holistic model
        :return: A frame and a namedTuple with fields describing the landmarks on the most prominate person detected:
        """
        _, frame = self._video_capture.read()
        self._current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._current_image.flags.writeable = False
        detection_results = model.process(self._current_image)
        self._current_image.flags.writeable = True
        self._current_image = cv2.cvtColor(self._current_image, cv2.COLOR_RGB2BGR)
        return detection_results

    def _draw_all_body_points(self, detection_results: NamedTuple) -> None:
        """
        Draws all landmarks from the detection result
        :param detection_results: Results from the detection
        :return: None
        """
        self._draw_landmarks(detection_results.face_landmarks, self._holistic_model.FACEMESH_TESSELATION)
        self._draw_landmarks(detection_results.right_hand_landmarks, self._holistic_model.HAND_CONNECTIONS)
        self._draw_landmarks(detection_results.left_hand_landmarks, self._holistic_model.HAND_CONNECTIONS)
        self._draw_landmarks(detection_results.pose_landmarks, self._holistic_model.POSE_CONNECTIONS)

    @staticmethod
    def get_body_points(detection_results: NamedTuple) -> Dict[str, List[list]]:
        """
        #TODO: CHANGE DOCSTRING
        Flattens the detection result to a single vector
        :param detection_results: Results from the detection
        :return: Vector of landmarks
        """
        body_points = {}
        for key in POINTS_NUM.keys():
            points = []
            landmarks = getattr(detection_results, f"{key.lower()}_landmarks")
            if getattr(detection_results, f"{key.lower()}_landmarks"):
                for landmark in landmarks.landmark:
                    if key == "POSE":
                        points.append(list(np.array([landmark.x, landmark.y, landmark.z, landmark.visibility]).flatten()))
                    else:
                        points.append(list(np.array([landmark.x, landmark.y, landmark.z]).flatten()))
            else:
                points.append(list(np.zeros(POINTS_NUM[key])))
            body_points[key] = points
        return body_points

    def detect_points(self, frames_start: int, frames_end: int) -> pd.DataFrame:
        """
        #TODO: CHANGE DOCSTRING
        Determines the landmarks for each frame in the interval
        :param frames_start: Initial frame number
        :param frames_end: Final frame number
        :return: Landmark vector for each frame
        """
        body_points = pd.DataFrame()
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            for i in range(frames_start, frames_end):
                self._video_capture.set(1, i)
                detection_results = self._run_detection(holistic)
                body_points = body_points.append(self.get_body_points(detection_results), ignore_index=True)
        return body_points

    @staticmethod
    def save_points(data: pd.DataFrame, action: str):
        """
        Saves landmark sequences from frames to a collective pandas dataframe
        :param data: Data to save
        :param action: Name for the frame sequence
        :return: None
        """
        if not data.empty:
            file = os.path.join(config.ROOT_DIR, "actions.pkl")
            data["ACTION"] = action
            data = data.groupby("ACTION").agg(list)
            if os.path.isfile(file):
                df = pd.read_pickle(file)

                if not data.isin(df).all().all():
                    df = df.append(data)
                else:
                    logging.warning("Data already exists in actions.pkl")
                    return
            else:
                df = data
            df.to_pickle(file)
            logging.info(f"Points for action {action} saved to {file}!")
        else:
            logging.warning("Data can't be None")

    @staticmethod
    def get_points():
        """
        :return: Saved actions
        """
        file = os.path.join(config.ROOT_DIR, "actions.pkl")
        if os.path.isfile(file):
            return pd.read_pickle(file)
        else:
            return None
