from os import listdir, makedirs
import cv2
import joblib
import numpy as np
from os.path import exists, join, dirname, splitext, basename, normpath
from values import (
    stiv_result_dir,
    sum_data_dir,
    ifft_img_dir,
    img_dir,
    valid_score_dir,
    valid_result_dir,
)


def smoothness(sum_list):

    sum_list = sum_list / np.max(sum_list)
    maxIndex = np.argmax(sum_list)
    diff = np.diff(sum_list)
    lsum = np.sum(np.abs(diff[:maxIndex])) / (np.max(sum_list) - sum_list[0])
    rsum = np.sum(np.abs(diff[maxIndex : len(diff)])) / (
        np.max(sum_list) - sum_list[len(sum_list) - 1]
    )
    score = 2 / (lsum + rsum)

    return score


def signal_peek_value(sum_list):

    return np.log(np.mean(sum_list))


def signal_noise_radio(sum_list, range_len=5):
    max_index = np.argmax(sum_list)
    total = np.max(sum_list)
    for i in range(range_len):
        i += 1
        index = max_index - i if max_index - i >= 0 else 0
        total += sum_list[index]
        index = max_index + i if max_index + i < len(sum_list) else len(sum_list) - 1
        total += sum_list[index]

    return total / np.sum(sum_list)


def svm(sum_list, svm_model_path=""):

    # sum_list = sum_list / np.max(sum_list)
    # loaded_model = joblib.load(svm_model_path)
    return 1  # loaded_model.decision_function([sum_list])[0]


def nn(ifft_img, nn_model_path=""):
    return 1


sumlist_all = {
    "methods": [
        smoothness,
        signal_peek_value,
        signal_noise_radio,
        svm,
    ],
    "threshold": [
        0.7,
        0.27,
        23,
        0.5,
    ],
    "get_func": lambda data_path: np.load(f"{data_path}.npy"),
    "data_dir": sum_data_dir,
}
ifftimg_all = {
    "methods": [nn],
    "threshold": [0.5],
    "get_func": lambda data_path: cv2.imread(f"{data_path}.jpg", 0),
    "data_dir": ifft_img_dir,
}

def img_valid_call(img_path, *arg, **kwarg):
    res_path = join(
        dirname(img_path),
        f"{splitext(basename(img_path))[0]}_{stiv_result_dir}",
    )
    if not exists(res_path):
        print(f"res_path:{res_path} not exists")
        return
    print(f"img_path: {img_path}")
    sress = call_methods(
        np.load(join(res_path, f"{basename(sum_data_dir)}.npy")), sumlist_all["methods"]
    )
    iress = call_methods(
        cv2.imread(join(res_path, f"{basename(ifft_img_dir)}.jpg"), 0),
        ifftimg_all["methods"],
    )

    for i, res in enumerate(sress):
        result = 0 if res < sumlist_all["threshold"][i] else 1
        print(f"{sumlist_all["methods"][i].__name__}={res:.2f}  {result}")
    for i, res in enumerate(iress):
        result = 0 if res < ifftimg_all["threshold"][i] else 1
        print(f"{ifftimg_all["methods"][i].__name__}={res:.2f}  {result}")


def call_methods(data, data_methods):
    ress = []
    for met in data_methods:
        res = met(data)
        ress.append(res)
    return ress


def imgs_valid_call(imgs_path, *arg, **kwarg):
    print(imgs_path)
    _imgs_valid_call(
        imgs_path,
        sumlist_all,
    )
    _imgs_valid_call(
        imgs_path,
        ifftimg_all,
    )


def _imgs_valid_call(imgs_path, data_all):
    ress = []
    for i in range(len(listdir(join(imgs_path, data_all["data_dir"])))):
        data = data_all["get_func"](
            join(imgs_path, data_all["data_dir"], f"{i:04}")
        )
        ress.append(call_methods(data, data_all["methods"]))
    ress = np.array(ress)

    makedirs(join(imgs_path, valid_score_dir), exist_ok=True)
    for i, met in enumerate(data_all["methods"]):
        np.save(
            join(imgs_path, valid_score_dir, f"{met.__name__}"),
            ress[i],
        )
        result = np.where(ress[i] < data_all["threshold"][i], 0, 1)
        makedirs(join(imgs_path, valid_result_dir), exist_ok=True)
        np.save(join(imgs_path, valid_result_dir, f"{met.__name__}.npy"), result)


def imgss_valid_call(imgss_path, *arg, **kwarg):
    imgs_valid_call(imgss_path)


valid_call = [img_valid_call, imgs_valid_call, imgss_valid_call]
if __name__ == "__main__":
    img_valid_call(normpath(r"test\stiv_routine\root"))
