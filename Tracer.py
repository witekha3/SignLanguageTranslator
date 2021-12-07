import glob
import os
import time
from pathlib import Path

import mediapipe as mp
import cv2
import numpy as np

import config
from Actions import Actions
from BPExtractor import BPExtractor
from Trainer import Trainer


class Tracer:

    def __init__(self):
        self._drawing = mp.solutions.drawing_utils  # Drawing helpers
        self._video_capture = cv2.VideoCapture(0)
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

    def _detect_all_body_points(self, detection_results):
        self._trace_landmark(detection_results.face_landmarks, self._holistic_model.FACE_CONNECTIONS)
        self._trace_landmark(detection_results.right_hand_landmarks, self._holistic_model.HAND_CONNECTIONS)
        self._trace_landmark(detection_results.left_hand_landmarks, self._holistic_model.HAND_CONNECTIONS)
        self._trace_landmark(detection_results.pose_landmarks, self._holistic_model.POSE_CONNECTIONS)

    def start_recording(self):
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            for idx, action in enumerate(Actions.available_actions):
                all_boyd_points = []
                for repeat_nbr in range(config.NUM_OF_CAPT_REPEATS):
                    for frame_nbr in range(config.NUM_OF_FRAMES):
                        detection_results = self._prepare_image_and_detection(holistic)
                        self._detect_all_body_points(detection_results)

                        if frame_nbr == 0:
                            print(f"Action: {action}\nStarting in 4s...")
                            for i in range(4, 0, -1):
                                print(i)
                                cv2.waitKey(1000)
                        else:
                            print(f"Action: {action}\tRepeat {repeat_nbr+1}/{config.NUM_OF_CAPT_REPEATS}")

                        all_boyd_points.append(BPExtractor.get_body_points(detection_results))

                        cv2.imshow('Webcam', self._current_image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                BPExtractor.save(all_boyd_points, action)

    def start_recognizing(self):
        sequence, sentence, predictions = [], [], []
        model = Trainer.load_model()
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            while self._video_capture.isOpened():
                detection_results = self._prepare_image_and_detection(holistic)
                self._detect_all_body_points(detection_results)
                sequence.append(BPExtractor.get_body_points(detection_results))
                sequence = sequence[-config.NUM_OF_FRAMES:]

                if len(sequence) == config.NUM_OF_FRAMES:
                    # sequence = np.resize(np.asarray(sequence), (30, 1662))

                    results = model.predict(np.expand_dims(sequence, axis=0))[0]

                    # if results[np.argmax(results)] > config.THRESHOLD:
                    #     print(Actions.available_actions[np.argmax(results)], results[np.argmax(results)])
                    # predictions.append(np.argmax(results))

                    # if np.unique(predictions[-10:])[0] == np.argmax(results):
                    #     predictions = []
                    print(Actions.available_actions[np.argmax(results)], results[np.argmax(results)])
                    if results[np.argmax(results)] >= config.THRESHOLD:
                        sequence = []
                        current_sentence = Actions.available_actions[np.argmax(results)]
                        if current_sentence == "_":
                            continue
                        if len(sentence) > 0:
                            if current_sentence != sentence[-1]:
                                sentence.append(current_sentence)
                                print(current_sentence, results[np.argmax(results)])
                        else:
                            sentence.append(current_sentence)
                            print(current_sentence, results[np.argmax(results)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                cv2.rectangle(self._current_image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(self._current_image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Raw Webcam Feed', self._current_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    #TODO: Change displaying
    def tmp_display_prediction(self, body_language_class, body_language_prob):
        pass

    def __del__(self):
        self._video_capture.release()
        cv2.destroyAllWindows()

## TODO: Usprawnić proces
## TODO: Pomyślec czy może być niestandardowa liczba framów?
## TODO: Dodać GUI z możlwością zapisu nagrania z którego będzie się potem mogło uczyć
## TODO: Dodać możlwiosć uczenia nie z kamerki a z filmików1
##TODO: Może dodać cos takiego że sekwencja będzie dłuższa niż 30 framów, potem przeszuka się to całe w poszukiwaniu różnych
## znaków i zwórci ten z największym prawdopodobieństwem?

## TODO: !!!!!! NIE ROBI TEGO GRAFICZNIE!!!! DODAC MENU W TERMINALU GDZIE MOZNA WYBIERAC CZY CHCESZ DODAC NOWE ZNAKI CZY MOZE APPENDOWAC CZY NADPISAC

# Tracer().start_recording()
# Trainer().train_and_save_model()
Tracer().start_recognizing()
