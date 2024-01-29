from os import listdir
from os.path import join
import cv2
import numpy as np


def get_imgs_data(
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
        datas.append(data)


def call4imgss(imgs_paths, call_func):
    for imgs_path in imgs_paths:
        call_func(imgs_path)
