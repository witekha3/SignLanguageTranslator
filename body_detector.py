import glob
import logging
import os
from os import listdir
from os.path import isdir
from pathlib import Path
from typing import Tuple, Optional, List, NamedTuple, Dict, Union

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from cv2 import VideoCapture
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions import holistic as holistic_model
from pandas import Series

import config

logging.basicConfig(level=logging.DEBUG)

"""
A dictionary storing the number of all points that mediapipe finds for each body part
Sum of points = 2172
"""
POINTS_NUM = {
    "POSE": 132,  # 33 * 4
    "FACE": 1872,  # 468 * 4
    "LEFT_HAND": 84,  # 21 * 4
    "RIGHT_HAND": 84,  # 21 * 4
}

ACTIONS_DIR = os.path.join(config.ROOT_DIR, "actions")


class BodyDetector:
    """
    Class responsible for detecting and handling landmarks on an actor's body

    :arg video_capture: VideoCapture from cv2
    """

    def __init__(self, video_capture: VideoCapture = None):
        self._drawing = mp.solutions.drawing_utils  # Drawing helpers
        self._video_capture = video_capture if video_capture else cv2.VideoCapture(0)
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

    def _run_detection(self, model: Holistic) -> Union[NamedTuple, None]:
        """
        Runs landmark detection
        :param model: MediaPipe holistic model
        :return: Mediapipe SolutionOutputs
        """
        _, frame = self._video_capture.read()
        if frame is None:
            return None
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
        self._draw_landmarks(detection_results.face_landmarks, holistic_model.FACEMESH_TESSELATION)
        self._draw_landmarks(detection_results.right_hand_landmarks, holistic_model.HAND_CONNECTIONS)
        self._draw_landmarks(detection_results.left_hand_landmarks, holistic_model.HAND_CONNECTIONS)
        self._draw_landmarks(detection_results.pose_landmarks, holistic_model.POSE_CONNECTIONS)

    @staticmethod
    def get_body_points(detection_results: NamedTuple) -> Dict[str, List[list]]:
        """
        Flattens the detection result to a single vector for each body part
        :param detection_results: Results from the detection
        :return: Dictionary of points for each body part
        """
        body_points = {}
        for key in POINTS_NUM.keys():
            points = []
            landmarks = getattr(detection_results, f"{key.lower()}_landmarks")
            if getattr(detection_results, f"{key.lower()}_landmarks"):
                for landmark in landmarks.landmark:
                    points.append(list(
                        np.round(np.array([landmark.x, landmark.y, landmark.z, landmark.visibility]).flatten(), 6)))
            else:
                points.extend(list([list(x) for x in np.zeros((int(POINTS_NUM[key] / 4), 4))]))
            body_points[key] = points
        return body_points

    def detect_points(
            self,
            frames_start: int,
            frames_end: int,
            draw_landmarks: bool = True,
            show_cam: bool = True
    ) -> pd.DataFrame:
        """
        Determines the landmarks for each frame in the interval
        :param frames_start: Initial frame number
        :param frames_end: Final frame number
        :param draw_landmarks: Setting this flag will display the points in the video
        :param show_cam: Setting this flag will display webcam view
        :return: Landmark vector for each frame
        """
        body_points = pd.DataFrame()
        with holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            for i in range(frames_start, frames_end):
                self._video_capture.set(1, i)
                detection_results = self._run_detection(holistic)
                if show_cam:
                    if draw_landmarks:
                        self._draw_all_body_points(detection_results)
                    cv2.imshow('Raw Webcam Feed', self._current_image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                body_points = body_points.append(self.get_body_points(detection_results), ignore_index=True)
        return body_points

    @staticmethod
    def flatten_action(action_row: Series) -> np.ndarray:
        """
        Flattens action points
        :param action_row: Action points as Series
        :return: Array of flattened points
        """
        frames = []
        for i in range(0, len(action_row.values[0])):  # num of frames
            flattened_points = []
            for j in range(0, len(POINTS_NUM)):  # num of body parts
                flattened_points.extend(np.array(action_row.values[j][i]).flatten())
            frames.append(np.array(flattened_points))
        return np.array(frames)

    @staticmethod
    def _prepare_dir(action_name: str) -> None:
        """Creates dir for action points"""
        Path(os.path.join(ACTIONS_DIR, action_name)).mkdir(exist_ok=True)

    @staticmethod
    def find_last_action_repeat(action: str) -> int:
        """
        Finds the last repeat of saved points for the action
        :param action: action name
        :return: Last repeat num
        """
        action_files = glob.glob(f'{ACTIONS_DIR}/{action}/{action}__*')
        if not action_files:
            return -1
        last_repeat = max([int(os.path.basename(action).split("__")[1].split(".")[0]) for action in action_files])
        return last_repeat

    @staticmethod
    def save_points(data: pd.DataFrame, action: str) -> None:
        """
        Saves a sequence of landmarks from the action
        :param data: Data to save
        :param action: Name for the frame sequence
        :return: None
        """
        if not data.empty:
            data["ACTION"] = action
            data = data.groupby("ACTION").agg(list)
            BodyDetector._prepare_dir(action)
            last_action_repeat = BodyDetector.find_last_action_repeat(action)
            output_file = os.path.join(ACTIONS_DIR, action, f"{action}__{last_action_repeat + 1}")
            data.to_pickle(output_file)
            logging.info(f"Points for action {action} saved to {output_file}!")
        else:
            logging.warning("Data can't be None")

    @staticmethod
    def get_all_actions_names() -> List[str]:
        """Returns a list of all saved actions"""
        return sorted([f for f in listdir(ACTIONS_DIR) if isdir(os.path.join(ACTIONS_DIR, f))])

    @staticmethod
    def get_points(action_name, repeat_num):
        """
        Gets saved points for given action and repeat
        :param action_name: action name
        :param repeat_num: action repeat number
        :return: Saved action
        """
        return pd.read_pickle(os.path.join(ACTIONS_DIR, action_name, f"{action_name}__{repeat_num}"))
