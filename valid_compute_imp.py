from os import listdir, makedirs
import joblib
import numpy as np
from os.path import join, exists
from key_value import kvs
from svm_train import svm_process


def v1_list_score(imgDir_path, current_img_index):
    list_path = kvs.sumlistDir
    _list_path = join(imgDir_path, list_path, f"{current_img_index:04}.npy")
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
    list_path = kvs.sumlistDir
    _list_path = join(imgDir_path, list_path, f"{current_img_index:04}.npy")
    sum_list = np.load(_list_path)

    return np.log(np.mean(sum_list))


def signal_noise_radio_list_score(imgDir_path, current_img_index, range_len=5):
    list_path = kvs.sumlistDir
    _list_path = join(imgDir_path, list_path, f"{current_img_index:04}.npy")
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
    svm_res_path = kvs.svmResDir
    modelName = "search"  # kvs.svmModelName
    list_path = kvs.sumlistDir
    _list_path = join(imgDir_path, list_path, f"{current_img_index:04}.npy")
    sum_list = svm_process(np.load(_list_path))
    loaded_model = joblib.load(join(svm_res_path, modelName + ".joblib"))
    res = loaded_model.predict([sum_list])
    if res > 0.5:
        return 1
    else:
        return 0


def nn_list_result(imgDir_path, current_img_index):
    return 0


validScoreMethods = [
    v1_list_score,
    v2_list_score,
    signal_peek_value_list_score,
    signal_noise_radio_list_score,
]

validResultMethods = [svm_list_result, nn_list_result]


def valid_score_data(root, metName, loc=None):
    ans = []
    data = []
    for dir1 in listdir(root):
        if loc != None and loc != dir1:
            continue
        for dir2 in listdir(join(root, dir1)):
            imgDir_path = join(root, dir1, dir2)
            # 需要首先完成数据整理以及算法结果输出
            if not exists(join(imgDir_path, kvs.validResDir, kvs.validRealFileName)):
                print(f"{imgDir_path} not exists valid_result")
                continue

            for x in np.load(join(imgDir_path, kvs.validResDir, kvs.validRealFileName)):
                ans.append(x)
            for x in np.load(join(imgDir_path, kvs.validScoDir, f"{metName}.npy")):
                data.append(x)

    return (
        join(
            kvs.ananlyzeResDir,
            kvs.validResDir + "_" + (loc if loc != None else ""),
            metName,
        ),
        np.array(ans),
        np.array(data),
    )


def valid_score_eval(res_Path, ans, data, div_num=100):
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
