from os.path import basename, dirname, join, exists
from os import listdir, remove
import shutil

import cv2
import numpy as np


def crop_img(img, crop_size=2**9):
    image = img
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    start_x = center_x - crop_size // 2
    start_y = center_y - crop_size // 2
    cropped_image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    return cropped_image


def get_imgs_datas(
    imgs_path,
    datas_dir,
):
    datas = []
    datas_path = join(imgs_path, datas_dir)
    for file in listdir(datas_path):
        data_path = join(datas_path, file)
        if file.endswith("jpg"):
            data = cv2.imread(data_path)
        elif file.endswith("npy"):
            data = np.load(data_path)
        else:
            print("get_imgs_data error")
            data = []
        datas.append(data)
    return datas


def get_imgs_data(
    imgs_path,
    data_dir,
):
    data_path = join(imgs_path, data_dir)
    if not exists(data_path):
        print("get_imgs_data error:{imgs_path} img_data not exists")
        return None
    if data_dir.endswith("jpg"):
        data = cv2.imread(data_path)
    elif data_dir.endswith("npy"):
        data = np.load(data_path)
    else:
        print("get_imgs_data error:{imgs_path} wrong tailfix")
        data = None
    return data


def call_for_imgss(imgs_paths, call_func, *arg, **kwarg):
    ress = []
    for imgs_path in imgs_paths:
        res = call_func(imgs_path, *arg, **kwarg)
        if res is None:
            continue
        ress.append(res)
    return ress


def get_imgs_paths(root):
    path_list = []
    for dir1 in listdir(root):
        for dir2 in listdir(join(root, dir1)):
            imgs_path = join(root, dir1, dir2)
            path_list.append(imgs_path)
    return path_list


def imgs_if_R2L(imgs_path):
    dic = {
        "ddh": False,
        "jx": True,
        "mc": False,
        "ah": True,
        "fj": True,
        "gx": False,
        "jxdx": True,
        "yc": False,
        "hd": True,
        "ys": True,
    }
    imgs_path_ = imgs_path
    while True:
        dir1 = basename(imgs_path)
        if dir1 in dic:
            return dic[dir1]

        if imgs_path == dirname(imgs_path):
            print(f"{imgs_path_}: unknown ifRightToLeft")
            return False
        else:
            imgs_path = dirname(imgs_path)


def imgss_del_call(imgs_path, del_dir):
    if del_dir == "" or del_dir == None:
        del_dir = "shield"
    del_path = join(imgs_path, del_dir)
    shutil.rmtree(
        del_path,
        ignore_errors=True,
    )
    if exists(del_path):
        remove(del_path)


if __name__ == "__main__":
    print(imgs_if_R2L("dds"))
