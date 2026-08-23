import cv2
import numpy as np
import math

class Canvas:
    def __init__(self, width, height, color=(255, 255, 255), thickness=5, max_jump=80):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.color = color
        self.thickness = thickness
        self.prev_point = None
        self.max_jump = max_jump

    def draw(self, point):
        """Draw a line from the previous fingertip point to the current one."""
        if point is None:
            self.prev_point = None  # pen lifted, don't connect across a gap
            return

        if self.prev_point is not None:
            dist = math.hypot(point[0] - self.prev_point[0], point[1] - self.prev_point[1])
            if dist <= self.max_jump:
                cv2.line(self.canvas, self.prev_point, point, self.color, self.thickness)

        self.prev_point = point

    def set_color(self, color):
        self.color = color

    def clear(self):
        self.canvas[:] = 0
        self.prev_point = None

    def overlay(self, frame):
        """Blend the drawing canvas on top of the live camera frame."""
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        return cv2.add(frame_bg, canvas_fg)