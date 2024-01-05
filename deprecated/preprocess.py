import cv2
import os
import numpy as np
import shutil


def video2stis(video_path, frame_num, lines):
    # USage：
    # in ------
    #   video_path：video视频的字符串路径，str
    #   frame_num: 连续帧的帧数，
    #              如果不设定则默认为视频帧数
    #   lines：测速线列表。元素为（起始像素位置h，起始像素位置w, 像素长度）,
    #          如果不设置，默认为一个元素，中心点附近长度为视频帧，但是限制不超过一半的视频宽度
    # out -----
    #   stis: sti，为cv2格式图片
    #   img: 展示测速线位置的图片，为为cv2格式图片

    stis = [[] for _ in lines]
    cap = cv2.VideoCapture(video_path)

    # 遍历视频帧
    img = None
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
        if count == 1:
            img = np.copy(frame)
            for line in lines:
                img[line[0] : line[0] + 1, line[1] : line[1] + line[2]] = 255

        for i, line in enumerate(lines):
            stis[i].append(frame[line[0], line[1] : line[1] + line[2]])
        if count == frame_num:
            break
    cap.release()

    for i, sti in enumerate(stis):
        stis[i] = np.array(stis[i])
    return stis, img


def video2stis_test():
    video_path = os.path.normpath(r"temp\expMP4\03.mp4")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    lines = [[i, width // 2, 500] for i in range(0, height, 1)]
    stis, img = video2stis(video_path, 500, lines)

    save_path = os.path.normpath(r"temp\expSTI")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    cv2.imwrite(os.path.join(save_path, "loc.jpg"), img)
    for i, sti in enumerate(stis):
        cv2.imwrite(os.path.join(save_path, f"{i}.jpg"), sti)


def videoCorr(video_path):
    calbra = {}
    calbra["gx.mp4"] = [
        np.mat(
            [[2286.800827, 0, 1879.600802], [0, 2285.485163, 1079.846760], [0, 0, 1]]
        ),
        np.mat([-0.364276, 0.239451, 0.002382, -0.000345, 0]),
    ]

    calbra["ah.mp4"] = [
        np.mat(
            [[2849.233887, 0, 1976.950317], [0, 2849.366943, 1176.270508], [0, 0, 1]]
        ),
        np.mat([-0.338594, 0.171456, 0.000179, 0.000028, 0]),
    ]

    calbra["fj.mp4"] = [
        np.mat(
            [[2290.306908, 0, 1957.975359], [0, 2292.110084, 1124.695252], [0, 0, 1]]
        ),
        np.mat([-0.339745, 0.184411, 0.000361, -0.063824, 0]),
    ]

    # 不需要校正
    name = os.path.basename(video_path)
    if not name in calbra.keys():
        return video_path

    # 已经校正好了
    save_path = os.path.join(
        os.path.dirname(video_path), "corr" + os.path.basename(video_path)
    )
    if os.path.exists(save_path):
        return save_path

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4格式，输出格式为mp4格式
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(save_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h), True)

    i = 0
    videoSize = (w, h)
    mat_intri = calbra[os.path.basename(video_path)][0]
    coff_dis = calbra[os.path.basename(video_path)][1]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
        mat_intri, coff_dis, videoSize, 1, videoSize
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 当获取完最后一帧就结束
        out.write(cv2.undistort(frame, mat_intri, coff_dis, None, newcameramtx))
        i = i + 1
        print("已经处理对视频第{}帧完成畸变矫正，且已写入".format(i))
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return save_path


def getVideoLines(video_path, w, h):
    liness = {}
    liness["corrah.mp4"] = [[x, 0, w] for x in range(200, 650)]

    name = os.path.basename(video_path)
    if name in liness.keys():
        return liness[name]
    else:
        return [[x, w // 2, 750] for x in range(0, h)]


if __name__ == "__main__":
    pass
