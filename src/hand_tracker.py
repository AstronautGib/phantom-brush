import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, model_path="models/hand_landmarker.task", max_hands=1,
                 detection_confidence=0.7, tracking_confidence=0.7):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.results = None
        self._frame_timestamp_ms = 0

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_timestamp_ms += 33  # approx timestamp step, ~30fps
        self.results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if self.results.hand_landmarks and draw:
            h, w, _ = frame.shape
            for hand_landmarks in self.results.hand_landmarks:
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), cv2.FILLED)
        return frame

    def get_index_finger_tip(self, frame):
        """Returns (x, y) pixel coordinates of the index fingertip, or None if no hand detected."""
        if self.results and self.results.hand_landmarks:
            hand_landmarks = self.results.hand_landmarks[0]
            h, w, _ = frame.shape
            tip = hand_landmarks[8]  # index finger tip
            return (int(tip.x * w), int(tip.y * h))
        return None