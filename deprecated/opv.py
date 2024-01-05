import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_motion(frame1, frame2):
    # 将帧转换为灰度图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1, 0)

    # 将光流可视化为箭头
    h, w = gray1.shape[:2]
    y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    sumx = np.sum(fx)
    sumy = np.sum(fy)
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=2)

    # 显示第一帧和可视化的光流
    # 创建一个大图像窗口
    plt.figure(figsize=(15, 9))

    # 只显示可视化的光流
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title('Optical Flow Visualization')
    plt.axis('off')
    plt.show()


# 读取视频
video_path = os.path.normpath(r'temp\expMP4\20231021\corrah.mp4')
cap = cv2.VideoCapture(video_path)

# 读取第一帧和第二帧
ret, frame1 = cap.read()

index = 10
while index > 0:
    cap.read()
    index -= 1
ret, frame2 = cap.read()

if ret:
    # 可视化帧之间的像素移动
    visualize_motion(frame1, frame2)
else:
    print('无法读取视频帧。')

# 释放视频捕获器
cap.release()
