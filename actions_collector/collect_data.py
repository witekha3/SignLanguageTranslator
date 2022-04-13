import copy
import os
from os import listdir

import numpy as np
import pandas as pd
import urllib.request
from urllib import error as url_error
import tempfile
import cv2
import logging

from body_detecotr import BodyDetector, POINTS_NUM

logging.basicConfig(level=logging.DEBUG)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")


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

    logging.info(f"Downloading file from {url}")
    urllib.request.urlretrieve(url, file_path)
    logging.info(f"File saved to {file_path}")
    return file_path


def _save_action(action: str, filename: str, gloss_start: int, gloss_end: int) -> None:
    """
    Saves the points from the action
    :param action: action name
    :param filename: filename for the video containing the action
    :param gloss_start: action start frame
    :param gloss_end: action end frame
    :return: None
    """
    video_cap = cv2.VideoCapture(filename)
    detector = BodyDetector(video_cap)
    try:
        body_points = detector.detect_points(gloss_start, gloss_end)
    except cv2.error:
        logging.error(f"CV2 error! Action: {action}, file: {filename}")
        return
    detector.save_points(body_points, action)


def collect_actions(save_to_temp=True) -> None:
    """
    Downloads video and then finds and records landmarks on the actor's body for each frame
    :return: None
    """
    saved_videos = {}
    uta_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uta_handshapes")
    for file in [file for file in listdir(uta_dir)]:
        df = pd.read_csv(os.path.join(uta_dir, file), header=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            for _, row in df.iterrows():
                for i in range(1, 5):
                    action = row["Sign gloss"]
                    gloss_start = row["Gloss start"]
                    gloss_end = row["Gloss end"]
                    mov_name = row['MOV'].split(".mov")[0][:-1]+str(i)+".mov"
                    mov_url = f"http://vlm1.uta.edu/~haijing/asl/camera1/{mov_name}"

                    try:
                        if mov_url not in saved_videos:
                            if not save_to_temp:
                                filename = _download_video(mov_url, os.path.join(VIDEOS_DIR, mov_name))
                            else:
                                filename = _download_video(mov_url, os.path.join(tmp_dir, mov_name))
                            saved_videos[mov_url] = filename
                        else:
                            filename = saved_videos[mov_url]
                            # logging.info(f"File {filename} already downloaded!")
                    except url_error.HTTPError:
                        # logging.warning(f"Url: {mov_url} not found!")
                        continue
                    except url_error.URLError as err:
                        logging.error(f"{err}. Url: {mov_url}")
                        continue
                    _save_action(action, filename, gloss_start, gloss_end)


def create_smaller_dataset():
    data = BodyDetector.get_points().sort_values("ACTION")[3:103]
    data.to_pickle(os.path.join(ROOT_DIR, "actions_small.pkl"))


def load_smaller_dataset():
    file = os.path.join(ROOT_DIR, "actions_small.pkl")
    if os.path.isfile(file):
        return pd.read_pickle(file)
    else:
        return None


# collect_actions(False)
# create_smaller_dataset()


