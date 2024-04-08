import shutil
import cv2
import pandas as pd
import math
import numpy as np
from os import listdir, makedirs, remove, rename
from os.path import join, dirname, splitext, basename, exists, isdir
from display import add_angle_img
from stiv import STIV

root = "data_base"  # data_gaussian_noise data_base
stiv_method_name = "sotabase"
stiv_result_dir = "result" + ("_" if stiv_method_name != "" else "") + stiv_method_name
ifft_res_dir = join(stiv_result_dir, "0_09_IFFTRES")
sti_res_dir = join(stiv_result_dir, "0_10_STIRES")
img_dir = join(stiv_result_dir, "0_00_ORIGIN")
sum_data_dir = join(stiv_result_dir, "1_00_sumlist")
ifft_img_dir = join(stiv_result_dir, "0_06_ifft")
site_img_dir = "hwMot"


def imgs_if_R2L(imgs_path):
    imgs_path_ = imgs_path
    while True:
        dir1 = basename(imgs_path)
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
        elif dir1 == "yc":
            return False
        elif dir1 == "hd":
            return True
        elif dir1 == "ys":
            return True

        if imgs_path == dirname(imgs_path):
            print(f"{imgs_path_}: unknown ifRightToLeft ")
            exit()
        else:
            imgs_path = dirname(imgs_path)


def reindex_file(prefix, path):
    for file in listdir(path):
        if not file.endswith(".jpg"):
            continue
        name, d = splitext(file)
        try:
            index = int(name[len(prefix) :])
        except ValueError:
            continue
        rename(join(path, file), join(path, f"{prefix}{index:03}" + d))


def flip_img(img, if_R2L):
    if if_R2L:
        img_lr = cv2.flip(img, 1)
    else:
        img_lr = img
    return img_lr


def img_test(img_path, if_R2L):
    # 计算
    print(img_path)
    img = cv2.imread(img_path)
    img = flip_img(img, if_R2L)
    _, proImgs, proDatas = STIV().sti2angle(img)

    res_path = join(
        dirname(img_path),
        f"{splitext(basename(img_path))[0]}_result_{stiv_method_name}",
    )

    # 结果保存
    shutil.rmtree(res_path, ignore_errors=True)
    makedirs(res_path)
    for index, [key, value] in enumerate(proImgs.items()):
        cv2.imwrite(join(res_path, f"0_{index:02}-{key}.jpg"), value)
    for index, [key, value] in enumerate(proDatas.items()):
        np.save(join(res_path, f"1_{index:02}-{key}.npy"), value)


def imgs_test(imgs_path, if_R2L):
    print(imgs_path)
    imgs = []
    for file in listdir(imgs_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(join(imgs_path, file))
        imgs.append(img)

    res_path = join(
        imgs_path,
        stiv_result_dir,
    )
    shutil.rmtree(res_path, ignore_errors=True)
    makedirs(res_path)

    for i, img in enumerate(imgs):
        img = flip_img(img, if_R2L)
        _, proImgs, proDatas = STIV().sti2angle(img)

        if i == 0:
            for j, [key, _] in enumerate(proImgs.items()):
                makedirs(join(res_path, f"0_{j:02}_{key}"))
            for j, [key, _] in enumerate(proDatas.items()):
                makedirs(join(res_path, f"1_{j:02}_{key}"))
        for j, [key, value] in enumerate(proImgs.items()):
            rPath = join(res_path, f"0_{j:02}_{key}", f"{i:04}.jpg")
            cv2.imwrite(rPath, value)
        for j, [key, value] in enumerate(proDatas.items()):
            rPath = join(res_path, f"1_{j:02}_{key}", f"{i:04}.npy")
            np.save(rPath, value)


def imgs_test_with_speed(imgs_path, if_R2L, if_use_score=False):
    if not exists(join(imgs_path, site_img_dir, "flow_speed_evaluation_result.csv")):
        # imgs_test(imgs_path, if_R2L)
        return

    print(imgs_path + " with speed", end=" \n")
    return
    imgs = []
    for file in listdir(imgs_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(join(imgs_path, file))
        imgs.append(img)

    res_path = join(
        imgs_path,
        stiv_result_dir,
    )
    shutil.rmtree(res_path, ignore_errors=True)
    makedirs(res_path)

    df = (
        pd.read_csv(join(imgs_path, site_img_dir, "flow_speed_evaluation_result.csv"))
        .dropna()
        .tail(len(imgs))
    )

    length = df.iloc[:, 7].values
    realSpeed = df.iloc[:, 5].values
    realress = [
        math.atan(realSpeed[i] * 750 / 25 / length[i]) * 180 / math.pi
        for i in range(len(realSpeed))
    ]

    ress = []
    scores = []

    for i, img in enumerate(imgs):
        img = flip_img(img, if_R2L)
        stiv = STIV()
        res, proImgs, proDatas = stiv.sti2angle(img)
        ress.append(res)
        scores.append(stiv.score)

        proImgs["realRES"] = add_angle_img(proImgs["ORIGIN"], 90 - realress[i])
        if i == 0:
            for j, [key, _] in enumerate(proImgs.items()):
                makedirs(join(res_path, f"0_{j:02}_{key}"))
            for j, [key, _] in enumerate(proDatas.items()):
                makedirs(join(res_path, f"1_{j:02}_{key}"))
        for j, [key, value] in enumerate(proImgs.items()):
            rPath = join(res_path, f"0_{j:02}_{key}", f"{i:04}.jpg")
            cv2.imwrite(rPath, value)
        for j, [key, value] in enumerate(proDatas.items()):
            rPath = join(res_path, f"1_{j:02}_{key}", f"{i:04}.npy")
            np.save(rPath, value)

    speed = [
        math.tan(ress[i] / 180 * math.pi) * 25 * length[i] / 750
        for i in range(len(ress))
    ]
    if if_use_score:
        scores = np.where(np.array(scores) < 0.3, 0, 1)
        speed = interpolate_speed(speed, scores)
    data = [length, realress, realSpeed, ress, speed]

    if exists(join(imgs_path, site_img_dir, "st_ress.txt")):
        print("and site_ress")
        st_ress = []
        with open(join(imgs_path, site_img_dir, "st_ress.txt"), "r") as f:
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
            join(res_path, "speed_result.xlsx"),
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
        relative_error = [
            (abs(realSpeed[i] - speed[i])) * 100 for i in range(len(speed))
        ]
        error_mean = np.array(relative_error).mean()
        relative_error.append(error_mean)
        with open("test.txt", "w") as f:
            if error_mean < 5:
                f.write(f"{imgs_path}: {error_mean}")
        data.append(relative_error)    
        df_save = pd.DataFrame(data)
        df_save = df_save.T
        df_save.to_excel(
            join(res_path, "speed_result.xlsx"),
            index=False,
            header=[
                "真实长度",
                "真值角度",
                "真值速度",
                "算法角度",
                "算法速度",
                "相对差值(cm/s)",
            ],
        )


def stiv_row_call(imgs_path):
    if "cop" in listdir(imgs_path):
        return

    makedirs(join(imgs_path, "cop"))
    makedirs(join(imgs_path, site_img_dir), exist_ok=True)
    for file in listdir(imgs_path):
        if file == "cop" or file == site_img_dir:
            continue
        fileName = join(imgs_path, file)
        copyName = join(imgs_path, "cop", file)
        if file.startswith("STI_MOT") or file == "flow_speed_evaluation_result.csv":
            shutil.move(
                src=fileName,
                dst=join(imgs_path, site_img_dir, file),
            )
            continue
        if file.startswith("sti") or isdir(join(imgs_path, file)):
            continue
        shutil.move(src=fileName, dst=copyName)

    reindex_file("sti", imgs_path)
    reindex_file("STI_MOT", join(imgs_path, site_img_dir))


def stiv_compute_call(imgs_path):
    # if stiv_result_dir in listdir(imgs_path):
    #     return
    imgs_test_with_speed(imgs_path, imgs_if_R2L(imgs_path), if_use_score=True)


def stiv_del_call(imgs_path, **kwarg):
    if kwarg["del_path"] == "" or kwarg["del_path"] == None:
        del_dir = "sheild"
    else:
        del_dir = kwarg["del_path"]
    del_path = join(imgs_path, del_dir)
    shutil.rmtree(
        del_path,
        ignore_errors=True,
    )
    if exists(del_path):
        remove(del_path)


def interpolate_speed(speed, validity):
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
