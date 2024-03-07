from os import listdir
from os.path import join
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, roc_curve


def crop_img(img, crop_size=2**9):
    image = img
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    start_x = center_x - crop_size // 2
    start_y = center_y - crop_size // 2
    cropped_image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    return cropped_image


def get_imgs_data(
    imgs_path,
    datas_dir,
):
    datas = []
    datas_path = join(imgs_path, datas_dir)
    for file in listdir(datas_path):
        data_path = join(datas_path, file)
        if file.endswith("jpg"):
            data = crop_img(cv2.imread(data_path))
        elif file.endswith("npy"):
            data = np.load(data_path)
        else:
            print("get_imgs_data error")
        datas.append(data)
    return datas


def call_for_imgss(imgs_paths, call_func, **kwarg):
    ress = []
    for imgs_path in imgs_paths:
        res = call_func(imgs_path, **kwarg)
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


def compute_precision(ans, pre):
    return precision_score(ans, pre)
    # TP = np.sum((ans == 1) & (pre == 1))
    # FP = np.sum((ans == 0) & (pre == 1))
    # if (TP + FP) == 0:
    #     return 1
    # else:
    #     return TP / (TP + FP)


def compute_auccracy(ans, pre):
    return accuracy_score(ans, pre)
    # return np.sum(pre == ans) / len(pre)


def compute_f1_score(ans, pre):
    return f1_score(ans, pre)


def compute_mean_precision(ans, score):
    return average_precision_score(ans, score)
