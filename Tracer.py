import mediapipe as mp
import cv2
import numpy as np

import config
from BPExtractor import BPExtractor
from Trainer import Trainer


class Tracer:

    def __init__(self):
        self._drawing = mp.solutions.drawing_utils  # Drawing helpers
        self._video_capture = cv2.VideoCapture(0)
        self._holistic_model = mp.solutions.holistic
        self._current_image = None

    def _trace_landmark(self, landmark, connections, color=(80, 110, 10), thickness=1, circle_radius=1):
        self._drawing.draw_landmarks(self._current_image, landmark, connections,
                                     self._drawing.DrawingSpec(color, thickness, circle_radius))

    def _prepare_image_and_detection(self, model):
        _, frame = self._video_capture.read()
        self._current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._current_image.flags.writeable = False
        detection_results = model.process(self._current_image)
        self._current_image.flags.writeable = True
        self._current_image = cv2.cvtColor(self._current_image, cv2.COLOR_RGB2BGR)
        return detection_results

    def _detect_all_body_points(self, detection_results):
        self._trace_landmark(detection_results.face_landmarks, self._holistic_model.FACE_CONNECTIONS)
        self._trace_landmark(detection_results.right_hand_landmarks, self._holistic_model.HAND_CONNECTIONS)
        self._trace_landmark(detection_results.left_hand_landmarks, self._holistic_model.HAND_CONNECTIONS)
        self._trace_landmark(detection_results.pose_landmarks, self._holistic_model.POSE_CONNECTIONS)

    def start_recording(self):
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            ##TODO: DOBRZE BY BYLO ABY TO ACTION SIEDZIALOW PANDASIE
            for action in config.ACTIONS:
                for repeat_nbr in range(config.NUM_OF_CAPT_REPEATS):
                    for frame_nbr in range(config.NUM_OF_FRAMES):
                        detection_results = self._prepare_image_and_detection(holistic)
                        self._detect_all_body_points(detection_results)

                        if frame_nbr == 0:
                            print("Starting in 2s...")
                            cv2.waitKey(2000)
                        else:
                            print(f"Action: {action}\tRepeat {repeat_nbr} of {config.NUM_OF_CAPT_REPEATS}")

                        all_boyd_points = BPExtractor.get_body_points(detection_results)
                        ##TODO: TUTAJ ZAPPISUJE KAŻDĄ KLATKĘ OSOBNO, ALE MOŻE MOŻNA BY BYLo ZAPISYWAC W JEDNYM FOLDERZE WIELE KLATEK?
                        BPExtractor.save(all_boyd_points, action, repeat_nbr, frame_nbr)

                        cv2.imshow('Raw Webcam Feed', self._current_image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

    def start_recognizing(self):
        sequence, sentence, predictions = [], [], []
        model = Trainer.load_model()
        with self._holistic_model.Holistic(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                           min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE) as holistic:
            while self._video_capture.isOpened():
                detection_results = self._prepare_image_and_detection(holistic)
                self._detect_all_body_points(detection_results)
                sequence.append(BPExtractor.get_body_points(detection_results))
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(config.ACTIONS[np.argmax(res)])
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > config.THRESHOLD:

                            if len(sentence) > 0:
                                if config.ACTIONS[np.argmax(res)] != sentence[-1]:
                                    sentence.append(config.ACTIONS[np.argmax(res)])
                            else:
                                sentence.append(config.ACTIONS[np.argmax(res)])

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

#Trainer().train_and_save_model()
#Tracer().start_recording()
Tracer().start_recognizing()