import cv2
import mediapipe as mp
import time


class HandTracker(mp.solutions.hands.Hands):

    def __init__(self, show_fps=True, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        super().__init__(static_image_mode, max_num_hands, min_detection_confidence, min_tracking_confidence)
        self.show_fps = show_fps
        self.cap = cv2.VideoCapture(0)
        self.current_time = 0
        self.prev_time = 0
        self.media_draw = mp.solutions.drawing_utils
        self.recognition_status = None
        self.hands_positions = list()

    def _track(self, draw=True):
        _, img = self.cap.read()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.recognition_status = self.process(rgb_img)
        # print(recognition_status.multi_hand_landmarks)
        if self.recognition_status.multi_hand_landmarks:
            for hand in self.recognition_status.multi_hand_landmarks:
                if draw:
                    self.media_draw.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def _add_fps(self, img):
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time)
        self.prev_time = self.current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    def find_hands_position(self, img, hand_ids=(0,), draw=True):
        #self.hands_positions = []
        if self.recognition_status.multi_hand_landmarks:
            for hand_nbr in hand_ids:
                try:
                    hand = self.recognition_status.multi_hand_landmarks[hand_nbr]
                    for id_, lm in enumerate(hand.landmark):
                        img_h, img_w, img_c = img.shape
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        self.hands_positions.append({"hand_id": hand_nbr, "lm_id": id_, "x": x, "y": y})
                        print({"hand_id": hand_nbr, "lm_id": id_, "x": x, "y": y})
                        if draw:
                            cv2.circle(img, (x, y), 10, (255, 125, 125), cv2.FILLED)
                except IndexError:
                    continue

    def start_tracking(self, draw=True, hand_ids=(0, 1)):
        while True:
            img = self._track(draw)
            if self.show_fps:
                self._add_fps(img)
            self.find_hands_position(img, hand_ids, draw=draw)
            cv2.imshow("Image", img)
            cv2.waitKey(1)


a = HandTracker(show_fps=True, min_detection_confidence=0.8)
a.start_tracking(hand_ids=(0, 1, 2))
