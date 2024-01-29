from os import listdir, makedirs
import joblib
from matplotlib import pyplot as plt
import numpy as np
from os.path import join, exists, basename
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from stiv_compute_routine import sum_data_dir
from svm_train import svm_model_dir, svm_process, model_name
from values import (
    valid_result_dir,
    valid_label_file,
    valid_score_dir,
    ananlyze_result_dir,
)
from stiv_compute_routine_imp import (
    root,
    stiv_result_dir,
    img_dir,
)


def v1_list_score(imgDir_path, current_img_index):
    _list_path = join(imgDir_path, sum_data_dir, f"{current_img_index:04}.npy")
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


def v2_list_score(imgDir_path, current_img_index):
    return 0.5


def signal_peek_value_list_score(imgDir_path, current_img_index):
    _list_path = join(imgDir_path, sum_data_dir, f"{current_img_index:04}.npy")
    sum_list = np.load(_list_path)

    return np.log(np.mean(sum_list))


def signal_noise_radio_list_score(imgDir_path, current_img_index, range_len=5):
    _list_path = join(imgDir_path, sum_data_dir, f"{current_img_index:04}.npy")
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


def svm_list_result(imgDir_path, current_img_index):
    _modelName = model_name
    _list_path = join(imgDir_path, sum_data_dir, f"{current_img_index:04}.npy")
    sum_list = svm_process(np.load(_list_path))
    loaded_model = joblib.load(join(svm_model_dir, _modelName + ".joblib"))
    res = loaded_model.predict([sum_list])
    if res > 0.5:
        return 1
    else:
        return 0


def nn_ifftimg_result(imgDir_path, current_img_index):
    ress = np.load(join(imgDir_path, valid_score_dir, f"nn_ifftimg_result.npy"))
    return ress[current_img_index]


valid_score_methods = [
    v1_list_score,
    v2_list_score,
    signal_peek_value_list_score,
    signal_noise_radio_list_score,
    nn_ifftimg_result,
]

valid_result_methods = [
    svm_list_result,
]


def valid_score_data(root, metName, loc=None):
    ans = []
    data = []
    for dir1 in listdir(root):
        if loc != None and loc != dir1:
            continue
        for dir2 in listdir(join(root, dir1)):
            imgDir_path = join(root, dir1, dir2)

            # 需要首先完成数据整理以及算法结果输出
            if not exists(join(imgDir_path, valid_score_dir, f"{metName}.npy")):
                print(f"{imgDir_path} not exists result")
                continue

            for x in np.load(join(imgDir_path, valid_result_dir, valid_label_file)):
                ans.append(x)
            for x in np.load(join(imgDir_path, valid_score_dir, f"{metName}.npy")):
                data.append(x)

    return (
        join(
            ananlyze_result_dir,
            valid_score_dir + (("_" + loc) if loc != None else ""),
            metName,
        ),
        np.array(ans),
        np.array(data),
    )


def valid_score_eval(res_Path, ans, data, div_num=100):
    ROCplot(basename(res_Path), ans, data)

    # 遍历阈值搜索最佳准确率结果
    precisions = []
    corrects = []

    max = np.max(data)
    min = np.min(data)
    for judge_value in np.linspace(min, max, div_num):
        _data = data.copy()
        _data = np.where(_data < judge_value, 0, 1)

        corrects.append(np.sum(_data == ans) / len(_data))

        TP = np.sum((ans == 1) & (_data == 1))
        FP = np.sum((ans == 0) & (_data == 1))
        if (TP + FP) == 0:
            precisions.append(1)
        else:
            precisions.append(TP / (TP + FP))

    correct_max_index = np.argmax(corrects)
    correct_max = corrects[correct_max_index]
    correct_max_precision = precisions[correct_max_index]
    correct_max_tho_value = np.linspace(min, max, div_num)[correct_max_index]
    full_precision_tho_value = None
    for j in range(len(precisions) - 1, -1, -1):
        if precisions[j] < 1:
            full_precision_tho_value = np.linspace(min, max, div_num)[j]
            break

    # 保存准确率和精确率中间结果
    makedirs(join(res_Path), exist_ok=True)
    np.save(join(res_Path, "correct.npy"), corrects)
    np.save(join(res_Path, "precision.npy"), precisions)

    # 保存各项数据
    with open(join(res_Path, "res.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"最大准确率值：{100*correct_max}\n",
        )
        f.write(f"最大准确率处的精确率值：{100*correct_max_precision}\n")
        f.write(f"最大准确率处的阈值：{correct_max_tho_value}\n")
        f.write(f"首次达到最大精确率值的阈值：{full_precision_tho_value}\n")

    return correct_max_tho_value


def valid_result_data(root, metName, loc=None):
    ans = []
    data = []
    for dir1 in listdir(root):
        if loc != None and loc != dir1:
            continue
        for dir2 in listdir(join(root, dir1)):
            imgDir_path = join(root, dir1, dir2)
            # 需要首先完成数据整理以及算法结果输出
            if not exists(join(imgDir_path, valid_result_dir, f"{metName}.npy")):
                print(f"{imgDir_path} not exists result")
                continue

            for x in np.load(join(imgDir_path, valid_result_dir, valid_label_file)):
                ans.append(x)
            for x in np.load(join(imgDir_path, valid_result_dir, f"{metName}.npy")):
                data.append(x)

    return (
        join(
            ananlyze_result_dir,
            valid_result_dir + (("_" + loc) if loc != None else ""),
            metName,
        ),
        np.array(ans),
        np.array(data),
    )


def valid_result_eval(res_Path, ans, data):
    _data = data.copy()
    correct = np.sum(_data == ans) / len(_data)
    TP = np.sum((ans == 1) & (_data == 1))
    FP = np.sum((ans == 0) & (_data == 1))
    if (TP + FP) == 0:
        precision = 1
    else:
        precision = TP / (TP + FP)

    # 保存各项数据
    makedirs(join(res_Path), exist_ok=True)
    with open(join(res_Path, "res.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"准确率：{100*correct}\n",
        )
        f.write(f"精确率：{100*precision}\n")


def ROCplot(name, ans, data):
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(data), np.array(ans), test_size=0.2, random_state=0
    )
    _y_test = ans
    _X_test = data
    titles = {
        "v1_list_score": "平稳程度",
        "v2_list_score": "v2_list_score",
        "signal_peek_value_list_score": "signal_peek_value_list_score",
        "signal_noise_radio_list_score": "峰值信噪比",
        "nn_ifftimg_result": "神经网络",
    }
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(_y_test, _X_test, pos_label=1)

    # 计算 AUC
    auc_score = roc_auc_score(_y_test, _X_test)

    # 绘制 ROC 曲线
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % auc_score)
    plt.plot([0, 1], [0, 1], "k--")  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("假阳率", fontsize=20)
    plt.ylabel("真阳率", fontsize=20)
    plt.title(titles[name], fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.show()

    # 打印 AUC
    print("AUC: {:.4f}".format(auc_score))


def valid_compute_call(imgs_path):
    # 检查是否计算以及是否已经标注
    if not exists(join(imgs_path, stiv_result_dir)):
        print(f"{imgs_path} not exists stiv_result")
        return
    if not exists(join(imgs_path, valid_result_dir, valid_label_file)):
        print(f"{imgs_path} not exists valid_label")
        return

    # 将不同指标结果直接保存
    for met in valid_score_methods:
        score_compute(imgs_path, met)


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
