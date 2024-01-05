import cv2
import numpy as np
import os

imgPath = os.path.normpath(r"")
img = cv2.imread(imgPath)
videoSize = (img.shape[1], img.shape[0])
mat_intri = np.mat(
    [[6.47844226e03, 0, 1.916703337e03], [0, 6.46306821e03, 1.08277481e03], [0, 0, 1]]
)
coff_dis = np.mat(
    [2.84687846e-02, 1.60334895e00, -5.66590458e-03, -3.05205746e-03, -1.32985282e01]
)
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
    mat_intri, coff_dis, videoSize, 1, videoSize
)


map1, map2 = cv2.initUndistortRectifyMap(
    mat_intri, coff_dis, None, newcameramtx, videoSize, cv2.CV_32FC1
)
resImg = cv2.remap(img, map1, map2, cv2.INTER_NEAREST)
# resImg = cv2.undistort(img, mat_intri,
#                        coff_dis, None, newcameramtx)
img = cv2.imread(os.path.normpath(r""))
print(np.sum(resImg - img) / (3840 * 2160 * 3))
name, ext = os.path.splitext(os.path.basename(imgPath))
cv2.imwrite(os.path.join(os.path.dirname(imgPath), name + "_copy." + ext), resImg)
