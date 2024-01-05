import shutil
import cv2
import os
import method.preprocess as preprocess
import method.stiv as method
import time
import pandas as pd
import math

# 配置参数
ifSavePro = True
ifRight2Left = False  # False
TestMode = 4  # 0:video 1:imgs 2:img 3:imgs&spedd 4:imgss
video_path = os.path.normpath(
    r"D:\Documents\flood\temp\202311181421\gx\rtsp_ffmpeg.mp4"
)
imgDir_path = os.path.normpath(r"D:\Documents\Code\flood2\temp2\202311270959\gx")
img_path = os.path.normpath(r"D:\Documents\Code\flood2\temp\ah\202311111213\STI009.jpg")
root = "valid"  # valid temp

stiv = method.STIV(ifSavePro, ifRight2Left, methodName="sotabase")
# base: 最基本的FFT方法，没有任何技巧
# sotabase: 最基本的IFFT方法, std+clr+fftclr+ifft+crop

if TestMode == 0:
    # 校正视频若校正则更换为校正视频路径
    video_path = preprocess.videoCorr(video_path)

    # 读取视频的第一帧和宽高
    cap = cv2.VideoCapture(video_path)
    ret, img_c = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 根据视频路径得到保存路径
    res_path = (
        os.path.join(
            os.path.dirname(video_path),
            os.path.splitext(os.path.basename(video_path))[0] + "_result",
        )
        + "_"
        + stiv.methodName
    )
    stiv.savePath = res_path
    os.mkdir(res_path)

    # 获取测速线并生成STI
    lines = preprocess.getVideoLines(video_path, width, height)
    timeS = time.time()
    stis, img = preprocess.video2stis(video_path, width, lines)
    timeE = time.time()
    cv2.imwrite(os.path.join(res_path, "_line_loc.jpg"), img_c)

    # 开始计算
    ress, scores = stiv.stis2anglesTest(stis)
    timeM = time.time()

    # 判定正确结果
    file = open(os.path.join(res_path, "log.txt"), "w")
    for i, score in enumerate(scores):
        if score > 0.8:
            img_c[
                lines[i][0] : lines[i][0] + 1, lines[i][1] : lines[i][1] + lines[i][2]
            ] = 255
            print(f"{i} : {ress[i]} with score {score}")
            file.write(f"{i} : {ress[i]} with score {score}\n")
    file.close()
    cv2.imwrite(os.path.join(res_path, "__proper_line_loc.jpg"), img_c)
    print("generate use " + str(timeE - timeS) + " s")
    print("compute use " + str(timeM - timeE) + " s")

elif TestMode == 1:
    res_path = os.path.join(imgDir_path, "result") + "_" + stiv.methodName
    stiv.savePath = res_path

    preprocess.renameHuaWeiSti("sti", imgDir_path)

    stis = []
    for file in os.listdir(imgDir_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(imgDir_path, file))
        stis.append(img[:, :])

    # 开始计算
    timeE = time.time()
    ress, scores = stiv.stis2anglesTest(stis)
    timeM = time.time()

    # 判定正确结果
    for i, score in enumerate(scores):
        print(score)
        if score > 0.8:
            print(f"{i} : {ress[i]} with score {score}")
    print("compute use " + str(timeM - timeE) + " s")

elif TestMode == 2:
    res_path = (
        os.path.join(
            os.path.dirname(img_path),
            os.path.splitext(os.path.basename(img_path))[0] + "_result",
        )
        + "_"
        + stiv.methodName
    )
    stiv.savePath = res_path

    img = cv2.imread(img_path)

    # 开始计算
    timeE = time.time()
    res = stiv.sti2angleTest(img[:, :])
    timeM = time.time()

    print("compute use " + str(timeM - timeE) + " s")

elif TestMode == 3:
    res_path = os.path.join(imgDir_path, "resultReal") + "_" + stiv.methodName
    stiv.savePath = res_path

    preprocess.renameHuaWeiSti("sti", imgDir_path)

    stis = []
    num = 0
    for file in os.listdir(imgDir_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(imgDir_path, file))
        stis.append(img)
        num += 1

    df = pd.read_csv(
        os.path.join(imgDir_path, "flow_speed_evaluation_result.csv")
    ).tail(num)
    length = df.iloc[:, 7].values
    realSpeed = df.iloc[:, 5].values
    realress = [
        math.atan(realSpeed[i] * 750 / 25 / length[i]) * 180 / math.pi
        for i in range(len(realSpeed))
    ]

    # 开始计算
    timeE = time.time()
    ress, scores = stiv.stis2anglesTest(stis, realress)
    timeM = time.time()

    # 判定正确结果
    for i, score in enumerate(scores):
        print(score)
        if score > 0.8:
            print(f"{i} : {ress[i]} with score {score}")
    print("compute use " + str(timeM - timeE) + " s")

elif TestMode == 4:
    for dir1 in os.listdir(root):
        if dir1 == "ddh":
            stiv.ifRightToLeft = False
        elif dir1 == "jx":
            stiv.ifRightToLeft = True
        elif dir1 == "mc":
            stiv.ifRightToLeft = False
        elif dir1 == "ah":
            stiv.ifRightToLeft = True
        elif dir1 == "fj":
            stiv.ifRightToLeft = True
        elif dir1 == "gx":
            stiv.ifRightToLeft = False
        else:
            print("stiv.ifRightToLeft is not init")
            exit()

        for dir2 in os.listdir(os.path.join(root, dir1)):
            imgDir_path = os.path.join(root, dir1, dir2)

            if root == "temp":
                # 如果发现在是生数据，那么整理成hwMotCop的形式
                if "cop" not in os.listdir(imgDir_path):
                    os.makedirs(os.path.join(imgDir_path, "cop"))
                    os.makedirs(os.path.join(imgDir_path, "hwMot"), exist_ok=True)
                    for dir3 in os.listdir(imgDir_path):
                        if dir3 == "cop":
                            continue
                        fileName = os.path.join(imgDir_path, dir3)
                        copyName = os.path.join(imgDir_path, "cop", dir3)
                        if dir3.startswith("STI_MOT"):
                            shutil.move(
                                src=fileName,
                                dst=os.path.join(imgDir_path, "hwMot", dir3),
                            )
                            continue
                        if dir3.startswith("sti"):
                            continue
                        shutil.move(src=fileName, dst=copyName)

                preprocess.renameRemoteSti("sti", imgDir_path)
                preprocess.renameRemoteSti(
                    "STI_MOT", os.path.join(imgDir_path, "hwMot")
                )

            res_path = os.path.join(
                imgDir_path,
                "result" + ("_" if stiv.methodName != "" else "") + stiv.methodName,
            )

            # 如果已经测试过了那么就不测试了
            if os.path.basename(res_path) in os.listdir(imgDir_path):
                continue

            stiv.savePath = res_path

            stis = []
            for file in os.listdir(imgDir_path):
                if not file.endswith(".jpg"):
                    continue
                img = cv2.imread(os.path.join(imgDir_path, file))
                stis.append(img)

            # 开始计算
            timeE = time.time()
            ress, scores = stiv.stis2anglesTest(stis)
            timeM = time.time()

            # 判定正确结果
            for i, score in enumerate(scores):
                print(score)
                if score > 0.8:
                    print(f"{i} : {ress[i]} with score {score}")
            print("compute use " + str(timeM - timeE) + " s")

else:
    print("Unknown method")
    exit()
