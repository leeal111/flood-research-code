from os.path import join, exists
from os import listdir, makedirs

from key_value import kvs
from valid_method import *

root = kvs.root
al_path = kvs.stivResPicDir
methods = validMethods
res_path = kvs.validResDir
for dir1 in listdir(root):
    for dir2 in listdir(join(root, dir1)):
        imgDir_path = join(root, dir1, dir2)

        # 需要首先完成数据整理以及算法结果输出
        if not exists(join(imgDir_path, al_path)):
            print(f"{imgDir_path} not exists al_result")
            continue

        # 将不同指标结果直接保存
        for met in methods:
            current_img_index = 0
            ress = []
            for file in listdir(join(imgDir_path, al_path)):
                if not file.endswith("jpg"):
                    continue
                ress.append(met(imgDir_path, current_img_index))
                current_img_index += 1

            makedirs(join(imgDir_path, res_path), exist_ok=True)
            np.save(join(imgDir_path, res_path, f"{met.__name__}.npy"), np.array(ress))
