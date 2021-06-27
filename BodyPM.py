import csv
import os

import numpy as np
import pandas as pd

import config


class BodyPM:
    POINTS_NUM = {
        "POSE": 132,
        "FACE": 1872,
        "L_HAND": 84,
        "R_HAND": 84,
    }

    @staticmethod
    def _get_points_from(landmarks, body_part):
        if landmarks is None:
            return [0 for _ in range(0, BodyPM.POINTS_NUM[body_part])]
        return list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks.landmark]).flatten())

    @staticmethod
    def save(data):
        if data is not None:
            if os.path.isfile(config.BODY_POINTS_FILENAME) and data:
                mode = 'a'
            else:
                mode = 'w'
            with open(config.BODY_POINTS_FILENAME, mode=mode, newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(data)

    @staticmethod
    def load():
        if os.path.isfile(config.BODY_POINTS_FILENAME):
            return pd.read_csv(config.BODY_POINTS_FILENAME)

    @staticmethod
    def get_body_points(results):
        pose_points = BodyPM._get_points_from(results.pose_landmarks, "POSE")
        face_points = BodyPM._get_points_from(results.face_landmarks, "FACE")
        l_hand_points = BodyPM._get_points_from(results.left_hand_landmarks, "L_HAND")
        r_hand_points = BodyPM._get_points_from(results.right_hand_landmarks, "R_HAND")

        all_points = pose_points + face_points + l_hand_points + r_hand_points
        return all_points
