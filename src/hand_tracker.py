import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def get_index_finger_tip(self, frame):
        """Returns (x, y) pixel coordinates of the index fingertip, or None if no hand detected."""
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            # Landmark 8 = index finger tip
            tip = hand_landmarks.landmark[8]
            x, y = int(tip.x * w), int(tip.y * h)
            return (x, y)
        return None