import csv
import os
import pickle
import shutil

import numpy as np
import pandas as pd

import config


class BPExtractor:
    POINTS_NUM = {
        "POSE": 132,
        "FACE": 1404,
        "L_HAND": 63,
        "R_HAND": 63,
    }


    @staticmethod
    def load(action):
        return np.load(os.path.join(config.BODY_POINTS_DIR, f"{action}.npy"))

    @staticmethod
    def save(data, action, repeat_nbr, frame_nbr):
        if data is not None:
            file = os.path.join(config.BODY_POINTS_DIR, f"{action}.npy")
            if os.path.isfile(file):
                os.remove(file)
            np.save(file, data)
            #os.makedirs(path, exist_ok=True)
            #np.save(os.path.join(path, str(frame_nbr)), data)

    @staticmethod
    def get_body_points(results):
        pose_points = np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in
             results.pose_landmarks.landmark]).flatten() if results.pose_landmarks \
            else np.zeros(BPExtractor.POINTS_NUM["POSE"])

        face_points = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in
             results.face_landmarks.landmark]).flatten() if results.face_landmarks \
            else np.zeros(BPExtractor.POINTS_NUM["FACE"])

        l_hand_points = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in
             results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks \
            else np.zeros(BPExtractor.POINTS_NUM["L_HAND"])

        r_hand_points = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in
             results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks \
            else np.zeros(BPExtractor.POINTS_NUM["R_HAND"])

        return np.concatenate([pose_points, face_points, l_hand_points, r_hand_points])
