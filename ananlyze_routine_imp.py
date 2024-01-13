from os.path import join, exists, splitext
from os import listdir, makedirs
import shutil
import cv2
import numpy as np
from key_value import kvs
from utils import *


def valid_method_result_eval(res_Path, root, valid_path=kvs.validResDir, div_num=100):
    # 对于每一个站点单独进行测试
    for loc in listdir(root):
        ans_index = -1
        names = []
        datas = []
        for dir2 in listdir(join(root, loc)):
            imgDir_path = join(root, loc, dir2)

            # 需要首先完成数据整理以及算法结果输出
            if not exists(join(imgDir_path, valid_path)):
                print(f"{imgDir_path} not exists valid_result")
                continue

            # 将标注结果和有效性算法结果存入大数组中，并记录标注的序号和算法结果名称
            for i, file in enumerate(listdir(join(imgDir_path, valid_path))):
                if len(datas) == i:
                    datas.append([])
                    if file == kvs.validRealFileName:
                        ans_index = i
                    if i != ans_index:
                        names.append(splitext(file)[0])
                for x in np.load(join(imgDir_path, valid_path, file)):
                    datas[i].append(x)

        # 将标注结果区分出来
        _datas = []
        ans = np.array(datas[ans_index].copy())
        for i, x in enumerate(datas):
            if i == ans_index:
                continue
            _datas.append(np.array(x))

        # 遍历阈值搜索最佳准确率结果
        precisions = [[] for _ in _datas]
        corrects = [[] for _ in _datas]

        correct_max = []
        correct_max_precision = []
        correct_max_tho_values = []
        full_precision_tho_values = []

        for i, data in enumerate(_datas):
            max = 1
            min = 0
            for judge_value in np.linspace(min, max, div_num):
                _data = data.copy()
                _data = np.where(_data < judge_value, 0, 1)

                corrects[i].append(np.sum(_data == ans) / len(_data))

                TP = np.sum((ans == 1) & (_data == 1))
                FP = np.sum((ans == 0) & (_data == 1))
                if (TP + FP) == 0:
                    precisions[i].append(1)
                else:
                    precisions[i].append(TP / (TP + FP))
            correct_max_index = np.argmax(corrects[i])
            correct_max.append(corrects[i][correct_max_index])
            correct_max_precision.append(precisions[i][correct_max_index])
            correct_max_tho_values.append(
                np.linspace(min, max, div_num)[correct_max_index]
            )
            for j in range(len(precisions[i]) - 1, -1, -1):
                if precisions[i][j] < 1:
                    full_precision_tho_values.append(np.linspace(min, max, div_num)[j])
                    break

        # 保存准确率和精确率中间结果
        _valid_path = join(valid_path, loc)
        for i, res in enumerate(corrects):
            makedirs(join(res_Path, _valid_path, "correct"), exist_ok=True)
            np.save(join(res_Path, _valid_path, "correct", names[i] + ".npy"), res)
            cv2.imwrite(
                join(res_Path, _valid_path, "correct", names[i] + ".jpg"),
                listImg(res),
            )
        for i, res in enumerate(precisions):
            makedirs(join(res_Path, _valid_path, "precision"), exist_ok=True)
            np.save(join(res_Path, _valid_path, "precision", names[i] + ".npy"), res)
            cv2.imwrite(
                join(res_Path, _valid_path, "precision", names[i] + ".jpg"),
                listImg(res),
            )

        # 保存各项数据
        with open(join(res_Path, _valid_path, "res.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"方法名称：{names}\n",
            )
            f.write(
                f"最大准确率值：{[100*x for x in correct_max]}\n",
            )
            f.write(f"最大准确率处的精确率值：{[100*x for x in correct_max_precision]}\n")
            f.write(f"最大准确率处的阈值：{correct_max_tho_values}\n")
            f.write(f"首次达到最大精确率值的阈值：{full_precision_tho_values}\n")

        # 以最佳阈值为参考，获取算法判错图像用于更近一步的判断
        for index, name in enumerate(names):
            valid_method_wrong_img(
                res_Path,
                root,
                valid_path,
                name,
                correct_max_tho_values[index],
                loc,
            )
            valid_method_wrong_img(
                res_Path, root, valid_path, name, correct_max_tho_values[index], loc, 1
            )


def valid_method_wrong_img(
    res_Path, root, valid_path, name, judge_value, loc=None, mode=0
):  # mode：0 fp 1 tn
    list_imgs = []
    origin_imgs = []
    scores = []
    for dir1 in listdir(root):
        if loc != None and loc != dir1:
            continue
        for dir2 in listdir(join(root, dir1)):
            imgDir_path = join(root, dir1, dir2)

            # 需要首先完成数据整理以及算法结果输出
            if not exists(join(imgDir_path, valid_path)):
                # print(f"{imgDir_path} not exists valid_result")
                continue

            data = np.load(join(imgDir_path, valid_path, name + ".npy"))
            ans = np.load(join(imgDir_path, valid_path, "result" + ".npy"))
            _data = np.where(data < judge_value, 0, 1)
            for i, _ in enumerate(ans):
                if (mode == 0 and ans[i] == 0 and _data[i] == 1) or (
                    mode == 1 and ans[i] == 1 and _data[i] == 0
                ):
                    list_imgs.append(
                        cv2.imread(
                            join(
                                imgDir_path, "result_sotabase", "07_sum", f"{i:04}.jpg"
                            )
                        )
                    )
                    origin_imgs.append(
                        cv2.imread(
                            join(
                                imgDir_path,
                                "result_sotabase",
                                "11_STIRES",
                                f"{i:04}.jpg",
                            )
                        )
                    )
                    scores.append(data[i])
    wrong_path = "wrong" if mode == 0 else "tn_wrong"
    if loc == None:
        _valid_path = join(valid_path, wrong_path)
    else:
        _valid_path = join(valid_path, loc, wrong_path)
    for i, _ in enumerate(origin_imgs):
        makedirs(join(res_Path, _valid_path, name, "sumlist"), exist_ok=True)
        cv2.imwrite(
            join(res_Path, _valid_path, name, "sumlist", f"{i:04}_{scores[i]:.2f}.jpg"),
            list_imgs[i],
        )

        makedirs(join(res_Path, _valid_path, name, "STIRES"), exist_ok=True)
        cv2.imwrite(
            join(res_Path, _valid_path, name, "STIRES", f"{i:04}_{scores[i]:.2f}.jpg"),
            origin_imgs[i],
        )