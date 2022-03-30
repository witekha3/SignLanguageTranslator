import os

import numpy as np
import pandas as pd
import urllib.request
import tempfile
import cv2
import mediapipe as mp
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from mediapipe.framework.formats import landmark_pb2

from CollectData.body_detecotr import BodyDetector, POINTS_NUM


#TODO: RENAME MODULE INTO TOOLS OR STH LIKE THAT

def download_movie(url, tmp_dir, movie):
    filename = os.path.join(tmp_dir, movie.replace('/', '_'))
    print(f"Saving file to {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"File save to {filename}")
    return filename


def collect_actions():
    saved_videos = {}
    df = pd.read_csv('uta_handshapes/merged_uta.csv', header=1)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for _, row in df.iterrows():
                sign = row["Sign gloss"]
                mov_url = f"http://vlm1.uta.edu/~haijing/asl/camera1/{row['MOV']}"
                gloss_start = row["Gloss start"]
                gloss_end = row["Gloss end"]
                if mov_url not in saved_videos:
                    filename = download_movie(mov_url, tmp_dir, row['MOV'])
                    saved_videos[mov_url] = filename
                else:
                    filename = saved_videos[mov_url]

                vidcap = cv2.VideoCapture(filename)
                detector = BodyDetector(vidcap)
                body_points = detector.detect_points(gloss_start, gloss_end)
                detector.save_points(body_points, sign)
    except Exception as ex:
        print(ex)


# collect_actions()

a = BodyDetector.get_points()
c=2