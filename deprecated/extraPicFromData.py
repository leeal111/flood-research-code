import cv2
import numpy as np

cap = cv2.VideoCapture(r'test.dat')
while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        frame_c=frame
cap.release()
cv2.imwrite('path.jpg',frame_c)