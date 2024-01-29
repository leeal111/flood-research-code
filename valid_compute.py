from os.path import join, exists
from os import listdir, makedirs
from valid_compute_imp import *
from stiv_compute_routine import (
    root,
    stiv_result_dir,
    img_dir,
)

for dir1 in listdir(root):
    for dir2 in listdir(join(root, dir1)):
        imgDir_path = join(root, dir1, dir2)

        # 检查是否计算以及是否已经标注
        if not exists(join(imgDir_path, stiv_result_dir)):
            print(f"{imgDir_path} not exists stiv_result")
            continue
        if not exists(join(imgDir_path, valid_result_dir, valid_label_file)):
            print(f"{imgDir_path} not exists valid_label")
            continue

        # 将不同指标结果直接保存
        for met in valid_score_methods:
            score_compute(imgDir_path, met)

# 计算阈值
thos = []
for met in valid_score_methods:
    path, ans, data = valid_score_data(root, met.__name__, loc=None)
    thos.append(valid_score_eval(path, ans, data))

# 计算结果
for dir1 in listdir(root):
    for dir2 in listdir(join(root, dir1)):
        imgDir_path = join(root, dir1, dir2)

        # 需要首先完成数据整理以及算法结果输出
        if not exists(join(imgDir_path, stiv_result_dir)):
            # print(f"{imgDir_path} not exists al_result")
            continue

        # 将不同指标结果直接保存
        for met in valid_result_methods:
            result_compute(imgDir_path, met)

        # 将不同指标通过阈值比较保存
        for i, met in enumerate(valid_score_methods):
            data = np.load(join(imgDir_path, valid_score_dir, f"{met.__name__}.npy"))
            _data = np.where(data < thos[i], 0, 1)
            makedirs(join(imgDir_path, valid_result_dir), exist_ok=True)
            np.save(
                join(imgDir_path, valid_result_dir, f"{met.__name__}.npy"),
                _data,
            )

for met in valid_score_methods + valid_result_methods:
    path, ans, data = valid_result_data(root, met.__name__, loc=None)
    valid_result_eval(path, ans, data)


def result_compute(imgDir_path, met):
    current_img_index = 0
    ress = []
    for file in listdir(join(imgDir_path, img_dir)):
        if not file.endswith("jpg"):
            continue
        ress.append(met(imgDir_path, current_img_index))
        current_img_index += 1

    makedirs(join(imgDir_path, valid_result_dir), exist_ok=True)
    np.save(
        join(imgDir_path, valid_result_dir, f"{met.__name__}.npy"),
        np.array(ress),
    )


def score_compute(imgDir_path, met):
    current_img_index = 0
    ress = []
    for file in listdir(join(imgDir_path, img_dir)):
        if not file.endswith("jpg"):
            continue
        ress.append(met(imgDir_path, current_img_index))
        current_img_index += 1

    makedirs(join(imgDir_path, valid_score_dir), exist_ok=True)
    np.save(
        join(imgDir_path, valid_score_dir, f"{met.__name__}.npy"),
        np.array(ress),
    )
