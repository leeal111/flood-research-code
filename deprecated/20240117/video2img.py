import os
import cv2

root = "test"
for file in os.listdir():
    video_path = os.path.normpath(r"0.mp4")
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_c = frame
            break
    cap.release()
    cv2.imwrite("capture.jpg", frame_c)
