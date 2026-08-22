import cv2
from src.hand_tracker import HandTracker
from src.canvas import Canvas

cap = cv2.VideoCapture(0)
tracker = HandTracker()

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Failed to grab initial frame.")
    exit()

h,w, _ = frame.shape
canvas = Canvas(width=w, height=h)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame = tracker.find_hands(frame, draw=False)

    fingertip = tracker.get_index_finger_tip(frame)
    canvas.draw(fingertip)

    output = canvas.overlay(frame)

    if fingertip:
        cv2.circle(output, fingertip, 10, (255, 0, 255), 2)

    cv2.imshow("PhantomBrush - Hand Tracking", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas.clear()
    if cv2.getWindowProperty("PhantomBrush - Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()