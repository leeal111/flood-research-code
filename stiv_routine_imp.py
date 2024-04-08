import math
from os import listdir, makedirs, rename
import shutil
import cv2
import numpy as np
import pandas as pd
from display import add_angle_img
from stiv import STIV
from os.path import join, dirname, splitext, basename, normpath, exists, isdir
from utils import imgs_if_R2L
from values import (
    stiv_result_dir,
    xxx_img_dir,
    xxx_csv_name,
    stiv_csv_name,
    valid_threshold,
    stiv_real_name,
    xxx_img_prefix,
    xxx_others_dir,
    xxx_mot_prefix,
)

stiv = STIV()


def img_stiv_call(img_path, if_R2L, *arg, **kwarg):
    if not exists(img_path):
        print(f"img_path:{img_path} not exists")
        return
    img = cv2.imread(img_path)
    res = stiv.sti2angle(img, if_R2L)
    res_path = join(
        dirname(img_path),
        f"{splitext(basename(img_path))[0]}_{stiv_result_dir}",
    )
    makedirs(res_path, exist_ok=True)
    for index, [key, value] in enumerate(stiv.proImgs.items()):
        cv2.imwrite(join(res_path, f"0_{index:02}_{key}.jpg"), value)
    for index, [key, value] in enumerate(stiv.proDatas.items()):
        np.save(join(res_path, f"1_{index:02}_{key}.npy"), value)
    print(f"result={res:.1f} score={stiv.score:.2f}  img_path:{img_path}")


def interpolate_speed(speed, validity):
    # 根据当前计算出的速度以及有效性指标重新过滤整个计算结果
    new_speed = []
    temp = []
    last_s = 0
    for i, s in enumerate(speed):
        if validity[i] == 1:
            if len(temp) != 0:
                l = len(temp)
                for j in range(l):
                    new_speed.append(s * (j + 1) / (l + 1) + last_s * (l - j) / (l + 1))
                temp = []
            new_speed.append(s)
            last_s = s
        else:
            temp.append(1)
    if len(temp) != 0:
        l = len(temp)
        for j in range(l):
            new_speed.append(0 * (j + 1) / (l + 1) + last_s * (l - j) / (l + 1))
        temp = []

    return new_speed


def imgs_stiv_call(imgs_path, if_R2L, *arg, **kwarg):
    res_path = join(imgs_path, stiv_result_dir)
    print(imgs_path)
    ress = []
    scores = []
    i = 0
    for file in listdir(imgs_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(join(imgs_path, file))
        res = stiv.sti2angle(img, if_R2L)
        ress.append(res)
        scores.append(stiv.score)
        print(f"img {len(ress)}'s result------angle:{res:.1f} score:{stiv.score:.2f}")
        if i == 0:
            for j, [key, _] in enumerate(stiv.proImgs.items()):
                makedirs(join(res_path, f"0_{j:02}_{key}"), exist_ok=True)
            for j, [key, _] in enumerate(stiv.proDatas.items()):
                makedirs(join(res_path, f"1_{j:02}_{key}"), exist_ok=True)
        for j, [key, value] in enumerate(stiv.proImgs.items()):
            rPath = join(res_path, f"0_{j:02}_{key}", f"{i:04}.jpg")
            cv2.imwrite(rPath, value)
        for j, [key, value] in enumerate(stiv.proDatas.items()):
            rPath = join(res_path, f"1_{j:02}_{key}", f"{i:04}.npy")
            np.save(rPath, value)
        i += 1
    validity = np.where(np.array(scores) < valid_threshold, 0, 1)
    data = [
        ress,
        scores,
        validity,
    ]
    header = [
        "算法角度",
        "有效分数",
        "有效判定",
    ]
    speed_file_path = join(imgs_path, xxx_img_dir, xxx_csv_name)
    if exists(speed_file_path):
        data, header = imgs_stiv_xxx_extra(
            imgs_path, if_R2L, data, header, speed_file_path
        )
    res_excel_path = join(res_path, stiv_csv_name)
    data2excel(res_excel_path, data, header)
    print(f"detailed data save to path:{res_excel_path}")


def data2excel(res_excel_path, data, header):
    df_save = pd.DataFrame(data)
    df_save = df_save.T
    df_save.to_excel(
        res_excel_path,
        index=False,
        header=header,
    )


def imgs_stiv_xxx_extra(imgs_path, if_R2L, data, header, speed_file_path):

    ress = data[0]
    scores = data[1]
    validity = data[2]
    df = pd.read_csv(speed_file_path).dropna().tail(len(ress))
    line_length = df.iloc[:, 7].values
    real_speed = df.iloc[:, 5].values
    real_ress = [
        math.atan(real_speed[i] * 750 / 25 / line_length[i]) * 180 / math.pi
        for i in range(len(real_speed))
    ]
    speed = [
        math.tan(ress[i] / 180 * math.pi) * 25 * line_length[i] / 750
        for i in range(len(ress))
    ]
    filted_speed = interpolate_speed(speed, validity)
    absolute_error = [
        (abs(real_speed[i] - filted_speed[i])) * 100 for i in range(len(speed))
    ]
    error_mean = np.array(absolute_error).mean()
    absolute_error.append(error_mean)
    data = [
        line_length,
        real_ress,
        real_speed,
        ress,
        speed,
        scores,
        validity,
        filted_speed,
        absolute_error,
    ]
    _header = [
        "真实长度",
        "真值角度",
        "真值速度",
        header[0],
        "算法速度",
        header[1],
        header[2],
        "过滤速度",
        "绝对差值(cm/s)",
    ]
    res_path = join(imgs_path, stiv_result_dir)
    i = 0
    for file in listdir(imgs_path):
        if not file.endswith(".jpg"):
            continue
        if i == 0:
            makedirs(join(res_path, f"2_{0:02}_{stiv_real_name}"), exist_ok=True)
        img = cv2.imread(join(imgs_path, file), 0)
        if if_R2L:
            img = cv2.flip(img, 1)
        rPath = join(res_path, f"2_{0:02}_{stiv_real_name}", f"{i:04}.jpg")
        cv2.imwrite(rPath, add_angle_img(img, 90 - real_ress[i]))
        i += 1

    return data, _header


def imgss_stiv_call(imgs_path, *arg, **kwarg):
    imgs_stiv_xxx_preprocess(imgs_path)
    imgs_stiv_call(imgs_path, imgs_if_R2L(imgs_path))


def imgs_stiv_xxx_preprocess(imgs_path):
    if xxx_others_dir in listdir(imgs_path):
        return

    makedirs(join(imgs_path, xxx_others_dir))
    makedirs(join(imgs_path, xxx_img_dir), exist_ok=True)
    for file in listdir(imgs_path):
        if file == xxx_others_dir or file == xxx_img_dir:
            continue
        fileName = join(imgs_path, file)
        copyName = join(imgs_path, xxx_others_dir, file)
        if file.startswith(xxx_mot_prefix) or file == xxx_csv_name:
            shutil.move(
                src=fileName,
                dst=join(imgs_path, xxx_img_dir, file),
            )
            continue
        if file.startswith(xxx_img_prefix) or isdir(join(imgs_path, file)):
            continue
        shutil.move(src=fileName, dst=copyName)

    reindex_file(xxx_img_prefix, imgs_path)
    reindex_file(xxx_mot_prefix, join(imgs_path, xxx_img_dir))


def reindex_file(prefix, path):
    for file in listdir(path):
        if not file.endswith(".jpg"):
            continue
        name, d = splitext(file)
        try:
            index = int(name[len(prefix) :])
        except ValueError:
            print(f"path:{path} file:{file} a=int(b) ValueError")
            continue
        rename(join(path, file), join(path, f"{prefix}{index:03}" + d))


stiv_call = [img_stiv_call, imgs_stiv_call, imgss_stiv_call]
if __name__ == "__main__":
    imgss_stiv_call(normpath(r"test\stiv_routine\root"))
