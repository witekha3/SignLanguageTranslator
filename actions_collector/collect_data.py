import os
from os import listdir

import pandas as pd
import urllib.request
import tempfile
import cv2
import logging

from body_detecotr import BodyDetector

logging.basicConfig(level=logging.DEBUG)

def _download_video(url: str, file_path: str) -> str:
    """
    Downloads video from given url
    :param url: url to video
    :param file_path: Path to save the file
    :return: Path to saved file
    """
    file_dir, file_name = os.path.split(file_path)
    file_path = os.path.join(file_dir, file_path.replace('/', '_'))
    logging.info(f"Saving file to {file_path}...")
    urllib.request.urlretrieve(url, file_path)
    logging.info(f"File save to {file_path}")
    return file_path


def collect_actions() -> None:
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

