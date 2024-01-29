from os.path import join, exists
from os import listdir, makedirs
from utils import get_imgs_paths
from valid_compute_imp import *


path_list = get_imgs_paths(root)


        

# 计算阈值
thos = []
for met in valid_score_methods:
    path, ans, data = valid_score_data(root, met.__name__, loc=None)
    thos.append(valid_score_eval(path, ans, data))

# 计算结果
for dir1 in listdir(root):
    for dir2 in listdir(join(root, dir1)):
        imgs_path = join(root, dir1, dir2)

        # 需要首先完成数据整理以及算法结果输出
        if not exists(join(imgs_path, stiv_result_dir)):
            # print(f"{imgs_path} not exists al_result")
            continue

        # 将不同指标结果直接保存
        for met in valid_result_methods:
            result_compute(imgs_path, met)

        # 将不同指标通过阈值比较保存
        for i, met in enumerate(valid_score_methods):
            data = np.load(join(imgs_path, valid_score_dir, f"{met.__name__}.npy"))
            _data = np.where(data < thos[i], 0, 1)
            makedirs(join(imgs_path, valid_result_dir), exist_ok=True)
            np.save(
                join(imgs_path, valid_result_dir, f"{met.__name__}.npy"),
                _data,
            )

for met in valid_score_methods + valid_result_methods:
    path, ans, data = valid_result_data(root, met.__name__, loc=None)
    valid_result_eval(path, ans, data)


def result_compute(imgs_path, met):
    current_img_index = 0
    ress = []
    for file in listdir(join(imgs_path, img_dir)):
        if not file.endswith("jpg"):
            continue
        ress.append(met(imgs_path, current_img_index))
        current_img_index += 1

    makedirs(join(imgs_path, valid_result_dir), exist_ok=True)
    np.save(
        join(imgs_path, valid_result_dir, f"{met.__name__}.npy"),
        np.array(ress),
    )



