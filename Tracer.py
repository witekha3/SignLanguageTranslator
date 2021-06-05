import mediapipe as mp
import cv2


class Tracer:

    def __init__(self):
        self.drawing = mp.solutions.drawing_utils  # Drawing helpers
        self.holistic = mp.solutions.holistic  # Mediapipe Solutions
        self.results = None
        self.video_capture = cv2.VideoCapture(0)

    def trace_landmark(self, image, landmark, connections, color=(80, 110, 10), thickness=1, circle_radius=1):
        self.drawing.draw_landmarks(image, landmark, connections,
                                    self.drawing.DrawingSpec(color, thickness, circle_radius))

    def start_tracking(self):

        # Initiate holistic model
        with self.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detections
                self.results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Draw face landmarks
                self.trace_landmark(image, self.results.face_landmarks, self.holistic.FACE_CONNECTIONS)

                # 2. Right hand
                self.trace_landmark(image, self.results.right_hand_landmarks, self.holistic.HAND_CONNECTIONS)

                # 3. Left Hand
                self.trace_landmark(image, self.results.left_hand_landmarks, self.holistic.HAND_CONNECTIONS)

                # 4. Pose Detections
                self.trace_landmark(image, self.results.pose_landmarks, self.holistic.POSE_CONNECTIONS)

                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    def __del__(self):
        self.video_capture.release()
        cv2.destroyAllWindows()


Tracer().start_tracking()
