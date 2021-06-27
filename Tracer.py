import mediapipe as mp
import cv2
import numpy as np

from BodyPM import BodyPM
from Trainer import Trainer


class Tracer:

    def __init__(self):
        self.drawing = mp.solutions.drawing_utils  # Drawing helpers
        self.holistic = mp.solutions.holistic  # Mediapipe Solutions
        self.detection_results = None
        self.image = None
        self.video_capture = cv2.VideoCapture(0)

    def _trace_landmark(self, landmark, connections, color=(80, 110, 10), thickness=1, circle_radius=1):
        self.drawing.draw_landmarks(self.image, landmark, connections,
                                    self.drawing.DrawingSpec(color, thickness, circle_radius))

    def _prepare_image(self):
        _, frame = self.video_capture.read()
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False

    def _detect_all_body_points(self):
        self._trace_landmark(self.detection_results.face_landmarks, self.holistic.FACE_CONNECTIONS)
        self._trace_landmark(self.detection_results.right_hand_landmarks, self.holistic.HAND_CONNECTIONS)
        self._trace_landmark(self.detection_results.left_hand_landmarks, self.holistic.HAND_CONNECTIONS)
        self._trace_landmark(self.detection_results.pose_landmarks, self.holistic.POSE_CONNECTIONS)

    def start_tracking(self, save_points=False, gesture="", predict=True):
        model = Trainer.load_model() if predict else None
        with self.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.video_capture.isOpened():
                self._prepare_image()
                self.detection_results = holistic.process(self.image)
                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                self._detect_all_body_points()

                all_points = BodyPM.get_body_points(self.detection_results)

                if save_points:
                    all_points.insert(0, gesture)
                    BodyPM.save(all_points)

                if model is not None:
                    body_language_class, body_language_prob = Trainer.predict(all_points, model)
                    self.tmp_display_prediction(body_language_class, body_language_prob)


                cv2.imshow('Raw Webcam Feed', self.image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    #TODO: Change displaying
    def tmp_display_prediction(self, body_language_class, body_language_prob):
        coords = tuple(np.multiply(
            np.array(
                (self.detection_results.pose_landmarks.landmark[self.holistic.PoseLandmark.LEFT_EAR].x,
                 self.detection_results.pose_landmarks.landmark[self.holistic.PoseLandmark.LEFT_EAR].y))
            , [640, 480]).astype(int))

        cv2.rectangle(self.image,
                      (coords[0], coords[1] + 5),
                      (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                      (245, 117, 16), -1)
        cv2.putText(self.image, body_language_class, coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Get status box
        cv2.rectangle(self.image, (0, 0), (250, 60), (245, 117, 16), -1)

        # Display Class
        cv2.putText(self.image, 'CLASS'
                    , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(self.image, body_language_class.split(' ')[0]
                    , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display Probability
        cv2.putText(self.image, 'PROB'
                    , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(self.image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                    , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def __del__(self):
        self.video_capture.release()
        cv2.destroyAllWindows()

#Trainer().train_and_save_model()
#Tracer().start_tracking(gesture="Hello", save_points=True, predict=False)
Tracer().start_tracking()