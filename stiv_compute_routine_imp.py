import shutil
import cv2
import os
import pandas as pd
import math
import numpy as np


def ifRight2LeftForLoc(imgDir_path):
    imgDir_path_ = imgDir_path
    while True:
        dir1 = os.path.basename(imgDir_path)
        if dir1 == "ddh":
            return False
        elif dir1 == "jx":
            return True
        elif dir1 == "mc":
            return False
        elif dir1 == "ah":
            return True
        elif dir1 == "fj":
            return True
        elif dir1 == "gx":
            return False
        elif dir1 == "jxdx":
            return True

        if imgDir_path == os.path.dirname(imgDir_path):
            print(f"{imgDir_path_}: unknown ifRightToLeft ")
            exit()
        else:
            imgDir_path = os.path.dirname(imgDir_path)


def resPath(imgDir_path, name):
    res_path = os.path.join(
        imgDir_path,
        "result" + ("_" if name != "" else "") + name,
    )
    return res_path


def renameSti(prefix, path):
    for file in os.listdir(path):
        if not file.endswith(".jpg"):
            continue
        name, d = os.path.splitext(file)
        try:
            index = int(name[len(prefix) :])
        except ValueError:
            continue
        os.rename(
            os.path.join(path, file), os.path.join(path, f"{prefix}{index:03}" + d)
        )


def rowData2MyData(imgDir_path):
    # 如果发现在是生数据，那么整理成hwMotCop的形式
    if "cop" not in os.listdir(imgDir_path):
        os.makedirs(os.path.join(imgDir_path, "cop"))
        os.makedirs(os.path.join(imgDir_path, "hwMot"), exist_ok=True)
        for dir3 in os.listdir(imgDir_path):
            if dir3 == "cop" or dir3 == "hwMot":
                continue
            fileName = os.path.join(imgDir_path, dir3)
            copyName = os.path.join(imgDir_path, "cop", dir3)
            if dir3.startswith("STI_MOT") or dir3 == "flow_speed_evaluation_result.csv":
                shutil.move(
                    src=fileName,
                    dst=os.path.join(imgDir_path, "hwMot", dir3),
                )
                continue
            if dir3.startswith("sti"):
                continue
            shutil.move(src=fileName, dst=copyName)

        renameSti("sti", imgDir_path)
        renameSti("STI_MOT", os.path.join(imgDir_path, "hwMot"))


def ImgsTest(imgDir_path, stiv):
    print(imgDir_path)

    # 整理结果文件夹
    res_path = resPath(imgDir_path, stiv.methodName)
    stiv.savePath = res_path

    stis = []
    for file in os.listdir(imgDir_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(imgDir_path, file))
        stis.append(img[:, :])

    # 开始计算
    ress = stiv.stis2anglesTest(stis)
    print(ress, end="\n\n")


def ImgsTestWithSpeed(imgDir_path, stiv):
    print(imgDir_path + " with speed", end=" ")

    # 整理结果文件夹
    res_path = resPath(imgDir_path, stiv.methodName)
    stiv.savePath = res_path

    stis = []
    for file in os.listdir(imgDir_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(imgDir_path, file))
        stis.append(img)

    df = (
        pd.read_csv(
            os.path.join(imgDir_path, "hwMot", "flow_speed_evaluation_result.csv")
        )
        .dropna()
        .tail(len(stis))
    )

    length = df.iloc[:, 7].values
    realSpeed = df.iloc[:, 5].values
    realress = [
        math.atan(realSpeed[i] * 750 / 25 / length[i]) * 180 / math.pi
        for i in range(len(realSpeed))
    ]

    # 开始计算
    ress = stiv.stis2anglesTest(stis, realress)

    speed = [
        math.tan(ress[i] / 180 * math.pi) * 25 * length[i] / 750
        for i in range(len(ress))
    ]

    data = [length, realress, realSpeed, ress, speed]

    if os.path.exists(os.path.join(imgDir_path, "hwMot", "st_ress.txt")):
        print("and site_ress")
        st_ress = []
        with open(os.path.join(imgDir_path, "hwMot", "st_ress.txt"), "r") as f:
            for line in f.readlines():
                st_ress.append(float(line.strip()))
        st_ress = [90 - x if x < 90 else x - 90 for x in st_ress]
        st_speed = [
            math.tan(st_ress[i] / 180 * math.pi) * 25 * length[i] / 750
            for i in range(len(st_ress))
        ]

        relative_error = [
            (abs(st_speed[i] - speed[i])) / st_speed[i] * 100
            for i in range(len(st_ress))
        ]
        relative_error.append(np.array(relative_error).mean())

        data.append(st_ress)
        data.append(st_speed)
        data.append(relative_error)

        df_save = pd.DataFrame(data)
        df_save = df_save.T
        df_save.to_excel(
            os.path.join(stiv.savePath, "speed_result.xlsx"),
            index=False,
            header=[
                "真实长度",
                "真值角度",
                "真值速度",
                "算法角度",
                "算法速度",
                "站点角度",
                "站点速度",
                "相对差值(%)",
            ],
        )
    else:
        print("")
        df_save = pd.DataFrame(data)
        df_save = df_save.T
        df_save.to_excel(
            os.path.join(stiv.savePath, "speed_result.xlsx"),
            index=False,
            header=["真实长度", "真值角度", "真值速度", "算法角度", "算法速度"],
        )

    print(ress, end="\n\n")


def ImgTest(img_path, stiv):
    print(img_path)
    res_path = (
        os.path.join(
            os.path.dirname(img_path),
            os.path.splitext(os.path.basename(img_path))[0] + "_result",
        )
        + "_"
        + stiv.methodName
    )
    stiv.savePath = res_path

    # 开始计算
    img = cv2.imread(img_path)
    res = stiv.sti2angleTest(img[:, :])
    print(res, end="\n\n")
