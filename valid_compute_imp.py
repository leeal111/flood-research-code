from os.path import join, exists
from os import listdir, makedirs
from stiv_compute_routine_imp import sum_data_dir, stiv_result_dir, img_dir
from svm_train import svm_model_dir, svm_process, model_name
import joblib
import numpy as np

valid_score_dir = "valid_score"
valid_result_dir = "valid_result"
valid_label_file = "result3.npy"


def v1_list_score(imgs_path, current_img_index):
    _list_path = join(imgs_path, sum_data_dir, f"{current_img_index:04}.npy")
    sum_list = np.load(_list_path)
    # 计算峰值比例
    sum_list = sum_list / np.max(sum_list)
    maxIndex = np.argmax(sum_list)
    diff = np.diff(sum_list)
    lsum = np.sum(np.abs(diff[:maxIndex])) / (np.max(sum_list) - sum_list[0])
    rsum = np.sum(np.abs(diff[maxIndex : len(diff)])) / (
        np.max(sum_list) - sum_list[len(sum_list) - 1]
    )
    score = 2 / (lsum + rsum)

    return score


def signal_peek_value_list_score(imgs_path, current_img_index):
    _list_path = join(imgs_path, sum_data_dir, f"{current_img_index:04}.npy")
    sum_list = np.load(_list_path)

    return np.log(np.mean(sum_list))


def signal_noise_radio_list_score(imgs_path, current_img_index, range_len=5):
    _list_path = join(imgs_path, sum_data_dir, f"{current_img_index:04}.npy")
    sum_list = np.load(_list_path)
    max_index = np.argmax(sum_list)
    total = np.max(sum_list)
    for i in range(range_len):
        i += 1
        index = max_index - i if max_index - i >= 0 else 0
        total += sum_list[index]
        index = max_index + i if max_index + i < len(sum_list) else len(sum_list) - 1
        total += sum_list[index]

    return total / np.sum(sum_list)


def svm_list_result(imgs_path, current_img_index):
    _modelName = model_name
    _list_path = join(imgs_path, sum_data_dir, f"{current_img_index:04}.npy")
    sum_list = svm_process(np.load(_list_path))
    loaded_model = joblib.load(join(svm_model_dir, _modelName + ".joblib"))
    return loaded_model.decision_function([sum_list])[0]


def nn_ifftimg_result(imgs_path, current_img_index):
    if exists(join(imgs_path, valid_score_dir, "nn_ifftimg_result.npy")):
        ress = np.load(join(imgs_path, valid_score_dir, "nn_ifftimg_result.npy"))
        return ress[current_img_index]
    else:
        return 0


valid_score_methods = [
    v1_list_score,
    signal_noise_radio_list_score,
    signal_peek_value_list_score,
    svm_list_result,
    nn_ifftimg_result,
]


def valid_score_call(imgs_path):
    print(imgs_path)
    if not exists(join(imgs_path, stiv_result_dir)):
        print(f"{imgs_path} not exists stiv_result")
        return

    for met in valid_score_methods:
        score_compute(imgs_path, met)


def valid_result_call(imgs_path, **kwarg):
    if not exists(join(imgs_path, valid_score_dir)):
        print(f"{imgs_path} not exists valid_score")
        return

    for i, met in enumerate(valid_score_methods):
        result_compute(imgs_path, met, kwarg["valid_thos"][i])


def score_compute(imgs_path, met):
    current_img_index = 0
    ress = []
    for file in listdir(join(imgs_path, img_dir)):
        if not file.endswith("jpg"):
            continue
        ress.append(met(imgs_path, current_img_index))
        current_img_index += 1
    makedirs(join(imgs_path, valid_score_dir), exist_ok=True)
    np.save(
        join(imgs_path, valid_score_dir, f"{met.__name__}.npy"),
        np.array(ress),
    )


def result_compute(imgs_path, met, tho):
    print(imgs_path)
    ress = np.load(join(imgs_path, valid_score_dir, f"{met.__name__}.npy"))
    ress = np.where(ress < tho, 0, 1)
    makedirs(join(imgs_path, valid_result_dir), exist_ok=True)
    np.save(join(imgs_path, valid_result_dir, f"{met.__name__}.npy"), ress)
