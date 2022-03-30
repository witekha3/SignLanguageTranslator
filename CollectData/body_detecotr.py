import os

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

import config

POINTS_NUM = {
    "POSE": 132,
    "FACE": 1404,
    "LEFT_HAND": 63,
    "RIGHT_HAND": 63,
}


class BodyDetector:

    def __init__(self, video_capture=None):
        self._drawing = mp.solutions.drawing_utils  # Drawing helpers
        self._video_capture = video_capture if video_capture else cv2.VideoCapture(0)
        self._holistic_model = mp.solutions.holistic
        self._current_image = None
        self._black_image = None

    def _trace_landmark(self, landmark, connections, color=(80, 110, 10), thickness=1, circle_radius=1):
        self._drawing.draw_landmarks(self._current_image, landmark, connections,
                                     self._drawing.DrawingSpec(color, thickness, circle_radius))

    def _prepare_image_and_detection(self, model):
        _, frame = self._video_capture.read()
        if self._black_image is None:
            self._black_image = np.zeros(frame.shape, dtype=np.uint8)
        self._current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._current_image.flags.writeable = False
        detection_results = model.process(self._current_image)
        self._current_image.flags.writeable = True
        self._current_image = cv2.cvtColor(self._black_image, cv2.COLOR_RGB2BGR)
        return detection_results

    def _detect_all_body_points(self, detection_results): #IT IS PRINTING
        self._trace_landmark(detection_results.face_landmarks, self._holistic_model.FACEMESH_TESSELATION)
        self._trace_landmark(detection_results.right_hand_landmarks, self._holistic_model.HAND_CONNECTIONS)
        self._trace_landmark(detection_results.left_hand_landmarks, self._holistic_model.HAND_CONNECTIONS)
        self._trace_landmark(detection_results.pose_landmarks, self._holistic_model.POSE_CONNECTIONS)

    @staticmethod
    def _get_body_points(results):

        points = []
        for key in POINTS_NUM.keys():
            landmarks = getattr(results, f"{key.lower()}_landmarks")
            if getattr(results, f"{key.lower()}_landmarks"):
                for landmark in landmarks.landmark:
                    if key == "POSE":
                        points.append(np.array([landmark.x, landmark.y, landmark.z, landmark.visibility]).flatten())
                    else:
                        points.append(np.array([landmark.x, landmark.y, landmark.z]).flatten())
            else:
                points.append(np.zeros(POINTS_NUM[key]))
        return np.concatenate(points)


    def detect_points(self, frames_start, frames_end):
        body_points = []
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            for i in range(frames_start, frames_end):
                self._video_capture.set(1, i)
                detection_results = self._prepare_image_and_detection(holistic)
                body_points.append(self._get_body_points(detection_results))
        return body_points

    @staticmethod
    def save_points(data, action):
        if data is not None:
            file = os.path.join(config.BODY_POINTS_DIR, "actions.pkl")
            if os.path.isfile(file):
                df = pd.read_pickle(file)
                if data not in df["data"].values:
                    df = df.append({"action": action, "data": data}, ignore_index=True)
                else:
                    print("Data already exists in actions.pkl")
            else:
                df = pd.DataFrame({"action": action, "data": [data]})
            df.to_pickle(file)
            print(f"Points for action {action} saved to {file}!")
        else:
            print("Data can't be None")

    @staticmethod
    def get_points():
        file = os.path.join(config.BODY_POINTS_DIR, "actions.pkl")
        if os.path.isfile(file):
            return pd.read_pickle(file)
        else:
            return None
