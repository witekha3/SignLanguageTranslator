import logging
import os
import glob
import time
from os import listdir

import cv2
import pandas as pd

import config
from body_detector import BodyDetector

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_last_action_repeat(action_name, videos_dir):
    action_videos = glob.glob(f'{videos_dir}/{action_name}__*')
    if not action_videos:
        return None
    last_repeat = max([int(os.path.basename(video).split("__")[1].split(".")[0]) for video in action_videos])
    return last_repeat


def _record_video(video_name, frames_num=None):
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # (*'MP42')
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))

    i = 0
    while cap.isOpened():
        logging.debug(f"Number of frames: {i}")
        if frames_num and i == frames_num:
            break
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def _show_wait_screen():
    cap = cv2.VideoCapture(0)
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if i == 0:
            cv2.putText(frame, "START!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 4, cv2.LINE_4)
        else:
            cv2.putText(frame, str(i), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 4, cv2.LINE_4)

        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('Video', frame)
        time.sleep(1)
        i -= 1
        if (cv2.waitKey(10) & 0xFF == ord('q')) or i == -1:
            break
    cap.release()
    cv2.destroyAllWindows()


def record_video(repeats_num=1, videos_dir=None, frames_per_video=None):
    action_name = input("Type action name: ").lower()
    if not videos_dir:
        videos_dir = os.path.join(ROOT_DIR, "own_videos")
    for i in range(0, repeats_num):
        last_repeat = _find_last_action_repeat(action_name, videos_dir)
        if last_repeat is None:
            repeat_num = 0
        else:
            repeat_num = last_repeat + 1
        logging.debug(f"Repeat num: {repeat_num}")
        file_path = os.path.join(videos_dir, f"{action_name}__{repeat_num}.mp4")
        _show_wait_screen()
        _record_video(file_path, frames_per_video)


def collect_actions(videos_dir=None, draw_landmarks=False, show_webcam=True):
    if not videos_dir:
        videos_dir = os.path.join(ROOT_DIR, "own_videos")
    for filename in listdir(videos_dir):
        video_cap = cv2.VideoCapture(os.path.join(videos_dir, filename))
        action = filename.split("__")[0]
        detector = BodyDetector(video_cap)
        try:
            body_points = detector.detect_points(0, int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)), draw_landmarks, show_webcam)
            BodyDetector.save_points(body_points, action)
        except cv2.error:
            logging.error(f"CV2 error! Action: {action}, file: {filename}")
            return


# record_video(50, r"E:\Own\SignLanguageTranslator\actions_collector\own_const_len_videos", 50)
# collect_actions(r"E:\Own\SignLanguageTranslator\actions_collector\own_videos", True, show_webcam=True)
#
#
