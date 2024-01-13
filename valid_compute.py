from os.path import join, exists
from os import listdir, makedirs
from valid_compute_imp import *
from key_value import kvs

root = kvs.root
res_path = kvs.validResDir
scoreMethods = validScoreMethods
resultMethods = validResultMethods

al_path = kvs.stivResPicDir

# 计算分数
for dir1 in listdir(root):
    for dir2 in listdir(join(root, dir1)):
        imgDir_path = join(root, dir1, dir2)

        # 需要首先完成数据整理以及算法结果输出
        if not exists(join(imgDir_path, al_path)):
            print(f"{imgDir_path} not exists al_result")
            continue

        # 将不同指标结果直接保存
        for met in scoreMethods:
            current_img_index = 0
            ress = []
            for file in listdir(join(imgDir_path, al_path)):
                if not file.endswith("jpg"):
                    continue
                ress.append(met(imgDir_path, current_img_index))
                current_img_index += 1

            makedirs(join(imgDir_path, kvs.validScoDir), exist_ok=True)
            np.save(
                join(imgDir_path, kvs.validScoDir, f"{met.__name__}.npy"),
                np.array(ress),
            )
# 计算阈值
thos = []
for met in scoreMethods:
    path, ans, data = valid_score_data(root, met.__name__, loc=None)
    thos.append(valid_score_eval(path, ans, data))

# 计算结果
for dir1 in listdir(root):
    for dir2 in listdir(join(root, dir1)):
        imgDir_path = join(root, dir1, dir2)

        # 需要首先完成数据整理以及算法结果输出
        if not exists(join(imgDir_path, al_path)):
            # print(f"{imgDir_path} not exists al_result")
            continue

        # 将不同指标结果直接保存
        for met in resultMethods:
            current_img_index = 0
            ress = []
            for file in listdir(join(imgDir_path, al_path)):
                if not file.endswith("jpg"):
                    continue
                ress.append(met(imgDir_path, current_img_index))
                current_img_index += 1

            makedirs(join(imgDir_path, res_path), exist_ok=True)
            np.save(
                join(imgDir_path, res_path, f"{met.__name__}.npy"),
                np.array(ress),
            )

        # 将不同指标通过阈值比较保存
        for i, met in enumerate(scoreMethods):
            data = np.load(join(imgDir_path, kvs.validScoDir, f"{met.__name__}.npy"))
            _data = np.where(data < thos[i], 0, 1)
            makedirs(join(imgDir_path, res_path), exist_ok=True)
            np.save(
                join(imgDir_path, res_path, f"{met.__name__}.npy"),
                _data,
            )
