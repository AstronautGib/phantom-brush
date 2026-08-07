import cv2
from src.hand_tracker import HandTracker

cap = cv2.VideoCapture(0)
tracker = HandTracker()

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame = tracker.find_hands(frame)

    fingertip = tracker.get_index_finger_tip(frame)
    if fingertip:
        cv2.circle(frame, fingertip, 10, (255, 0, 255), cv2.FILLED)

    cv2.imshow("PhantomBrush - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()