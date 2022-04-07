import os
from os import listdir

import numpy as np
import pandas as pd
import urllib.request
import tempfile
import cv2
import logging
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

from body_detecotr import BodyDetector, POINTS_NUM

logging.basicConfig(level=logging.DEBUG)

VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")

def _download_video(url: str, file_path: str) -> str:
    """
    Downloads video from given url
    :param url: url to video
    :param file_path: Path to save the file
    :return: Path to saved file
    """

    file_dir, file_name = os.path.split(file_path)
    file_path = os.path.join(file_dir, file_path.replace('/', '_'))

    if os.path.exists(file_path):
        return file_path

    logging.info(f"Saving file to {file_path}...")
    urllib.request.urlretrieve(url, file_path)
    logging.info(f"File save to {file_path}")
    return file_path


def collect_actions(save_to_temp=True) -> None:
    """
    Downloads video and then finds and records landmarks on the actor's body for each frame
    :return: None
    """
    saved_videos = {}
    uta_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uta_handshapes")
    for file in [file for file in listdir(uta_dir)]:
        df = pd.read_csv(os.path.join(uta_dir, file), header=1)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                for _, row in df.iterrows():
                    sign = row["Sign gloss"]
                    mov_url = f"http://vlm1.uta.edu/~haijing/asl/camera1/{row['MOV']}"
                    gloss_start = row["Gloss start"]
                    gloss_end = row["Gloss end"]
                    if mov_url not in saved_videos:
                        if not save_to_temp:
                            filename = _download_video(mov_url, os.path.join(VIDEOS_DIR, row['MOV']))
                        else:
                            filename = _download_video(mov_url, os.path.join(tmp_dir, row['MOV']))
                        saved_videos[mov_url] = filename
                    else:
                        filename = saved_videos[mov_url]

                    video_cap = cv2.VideoCapture(filename)
                    detector = BodyDetector(video_cap)
                    body_points = detector.detect_points(gloss_start, gloss_end)
                    detector.save_points(body_points, sign)
        except Exception as ex:
            logging.error(ex)


def display_saved_points(action_name: str, nbr: int = 0):
    bd_points = BodyDetector.get_points()
    action_per_frames = bd_points.loc[[action_name]].apply(lambda x: x[nbr])
    landmarks_in_frame = []
    for i, data in action_per_frames.iterrows():
        landmarks_dict = {}
        for key in POINTS_NUM.keys():
            landmark_list = []
            for point in data[key]:
                nl = NormalizedLandmark()
                nl.x = point[0]
                nl.y = point[1]
                nl.z = point[2]
                if key == "POSE":
                    nl.visibility = point[3]
                landmark_list.append(nl)
            landmarks_dict[key] = landmark_pb2.NormalizedLandmarkList(landmark=landmark_list)
        landmarks_in_frame.append(landmarks_dict)

    for landmarks in landmarks_in_frame:
        img = np.zeros([800, 800, 3], dtype=np.uint8)
        img.fill(255)
        mp.solutions.drawing_utils.draw_landmarks(img, landmarks["FACE"], mp.solutions.holistic.FACEMESH_TESSELATION)
        mp.solutions.drawing_utils.draw_landmarks(img, landmarks["RIGHT_HAND"], mp.solutions.holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(img, landmarks["LEFT_HAND"], mp.solutions.holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(img, landmarks["POSE"], mp.solutions.holistic.POSE_CONNECTIONS)
        cv2.imshow("aa", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # TODO: Maybe add a text to sign option?




# collect_actions(save_to_temp=False)
display_saved_points("again")