from os.path import join, exists, splitext
from os import listdir, makedirs
import cv2
import numpy as np

from util_display import *


def valid_method_result_eval(res_Path, root, valid_path="valid_result", div_num=100):
    # 读取所有结果数据存放入datas，并记录数据名称和结果数据序号

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

            # 将结果存入大数组中
            for i, file in enumerate(listdir(join(imgDir_path, valid_path))):
                if len(datas) == i:
                    datas.append([])
                    if file.startswith("result"):
                        ans_index = i
                    if i != ans_index:
                        names.append(splitext(file)[0])
                for x in np.load(join(imgDir_path, valid_path, file)):
                    datas[i].append(x)

        # 整理成np数组,并区分数据
        _datas = []
        ans = np.array(datas[ans_index].copy())
        for i, x in enumerate(datas):
            if i == ans_index:
                continue
            _datas.append(np.array(x))

        # 计算结果
        precisions = [[] for _ in _datas]
        corrects = [[] for _ in _datas]

        max_correct = []
        max_precision = []
        max_precision_tho_values = []
        complete_tho_values = []

        for i, data in enumerate(_datas):
            max = np.max(data)
            min = np.min(data)
            for judge_value in np.linspace(min, max, div_num):
                _data = data.copy()
                _data = np.where(_data < judge_value, 0, 1)

                corrects[i].append(np.sum(_data == ans) / len(_data))

                TP = np.sum((ans == 1) & (_data == 1))
                FP = np.sum((ans == 0) & (_data == 1))
                precisions[i].append(TP / (TP + FP))
            max_index = np.argmax(corrects[i])
            max_correct.append(corrects[i][max_index])
            max_precision.append(precisions[i][max_index])
            max_precision_tho_values.append(
                np.linspace(min, max, div_num)[np.argmax(corrects[i])]
            )
            for j in range(len(precisions[i]) - 1, -1, -1):
                if precisions[i][j] < 1:
                    complete_tho_values.append(np.linspace(min, max, div_num)[j])
                    break

        # 图片保存以及精确率结果输出
        _valid_path = join(valid_path, loc)
        for i, res in enumerate(corrects):
            makedirs(join(res_Path, _valid_path, "correct"), exist_ok=True)
            cv2.imwrite(
                join(res_Path, _valid_path, "correct", names[i] + ".jpg"),
                listImg(res),
            )

        for i, res in enumerate(precisions):
            makedirs(join(res_Path, _valid_path, "precision"), exist_ok=True)
            cv2.imwrite(
                join(res_Path, _valid_path, "precision", names[i] + ".jpg"),
                listImg(res),
            )

        # 打印有效信息
        with open(join(res_Path, _valid_path, "res.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"方法名称：{names}\n",
            )
            f.write(
                f"最大准确率值：{[100*x for x in max_correct]}\n",
            )
            f.write(f"精确率值：{[100*x for x in max_precision]}\n")
            f.write(f"最佳阈值：{max_precision_tho_values}\n")
            f.write(f"完全阈值：{complete_tho_values}\n")

            print(f"最佳阈值：{max_precision_tho_values}")
            print(f"完全阈值：{complete_tho_values}")

        for index, name in enumerate(names):
            valid_method_wrong_img(
                res_Path,
                root,
                valid_path,
                name,
                max_precision_tho_values[index],
                loc,
            )


# loc用来控制是对整个数据集进行错误检查还是对单个地址的站点进行错误检查
def valid_method_wrong_img(res_Path, root, valid_path, name, judge_value, loc=None):
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
                if ans[i] == 0 and _data[i] == 1:
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

    if loc == None:
        _valid_path = join(valid_path, "wrong")
    else:
        _valid_path = join(valid_path, loc, "wrong")
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


# correct_rate_ananlyze(root, ananlyze_name, os.path.join(res_Path, "correct_rate"))
# wrong_img_ananlyze(root, ananlyze_name, os.path.join(res_Path, "wrong_img"))
# type_data_sum(
#     root,
#     ananlyze_name,
#     os.path.join(res_Path, "wrong_img"),
# )
# res_Path = os.path.join(res_Path, "valid_rate")
# os.makedirs(os.path.join(res_Path), exist_ok=True)
# for loc in os.listdir(root):
#     total_num = 0
#     valid_cor_num = 0
#     for dir in os.listdir(os.path.join(root, loc)):
#         data_path = os.path.join(root, loc, dir, ananlyze_name, "sumlist.txt")
#         if not os.path.exists(data_path):
#             continue
#         with open(data_path, "r") as resfile:
#             lines = resfile.readlines()
#             for line in lines:
#                 total_num += 1
#                 if line.split()[1] == line.split()[2]:
#                     valid_cor_num += 1
#     with open(os.path.join(res_Path, "result.txt"), "a", encoding="utf-8") as file:
#         if total_num == 0:
#             file.write(f"{loc}无有效图片\n")
#         else:
#             file.write(
#                 f"{loc}图片共{total_num}张，其中有效判断准确率为 {(valid_cor_num/total_num*100):.1f}%\n"
#             )


# def correct_rate_ananlyze(
#     root,
#     ananlyze_path,
#     res_Path,
# ):
#     os.makedirs(os.path.join(res_Path), exist_ok=True)
#     for loc in os.listdir(root):
#         total_num = 0
#         st_cor_num = 0
#         al_cor_num = 0
#         with open(os.path.join(res_Path, loc + ".txt"), "w") as file:
#             for dir in os.listdir(os.path.join(root, loc)):
#                 data_path = os.path.join(root, loc, dir, ananlyze_path, "res.txt")
#                 if not os.path.exists(data_path):
#                     continue
#                 with open(data_path, "r") as resfile:
#                     line = resfile.readline()
#                     total_num += int(line.split()[2])
#                     st_cor_num += int(line.split()[3])
#                     al_cor_num += int(line.split()[4])
#                     file.write(line)
#         with open(os.path.join(res_Path, "result.txt"), "a", encoding="utf-8") as file:
#             if total_num == 0:
#                 file.write(f"{loc}无有效图片\n")
#             else:
#                 file.write(
#                     f"{loc}有效图片共{total_num}张，其中站点和算法准确率分别为 {(st_cor_num/total_num*100):.1f}% 和 {(al_cor_num/total_num*100):.1f}%\n"
#                 )


# def wrong_img_ananlyze(
#     root,
#     ananlyze_path,
#     res_Path,
# ):
#     for loc in os.listdir(root):
#         wrong_imgs = []
#         for dir in os.listdir(os.path.join(root, loc)):
#             imgWrong_path = os.path.join(
#                 root, loc, dir, ananlyze_path, "valid_wrong_imgs"
#             )
#             if not os.path.exists(imgWrong_path):
#                 continue
#             for file in os.listdir(imgWrong_path):
#                 if not file.endswith("jpg"):
#                     continue
#                 wrong_imgs.append(cv2.imread(os.path.join(imgWrong_path, file)))
#         save_loc_path = os.path.join(res_Path, loc)

#         shutil.rmtree(save_loc_path, ignore_errors=True)
#         os.makedirs(save_loc_path)
#         for idx, img in enumerate(wrong_imgs):
#             cv2.imwrite(
#                 os.path.join(
#                     save_loc_path,
#                     f"{idx:04}.jpg",
#                 ),
#                 wrong_imgs[idx],
#             )


# def type_data_sum(
#     root,
#     ananlyze_path,
#     res_Path,
#     type_name,
# ):
#     type_list = [
#         "invalid",
#         "total_right",
#         "al_right",
#         "st_right",
#         "total_wrong",
#     ]
#     if type_name not in type_list:
#         return
#     shutil.rmtree(os.path.join(res_Path), ignore_errors=True)
#     for loc in os.listdir(root):
#         save_loc_path = os.path.join(res_Path, type_name, loc)
#         st_path = os.path.join(save_loc_path, "st")
#         al_path = os.path.join(save_loc_path, "al")
#         os.makedirs(st_path)
#         os.makedirs(al_path)

#         st_imgs = []
#         al_imgs = []
#         for dir in os.listdir(os.path.join(root, loc)):
#             imgs_path = os.path.join(root, loc, dir, ananlyze_path, type_name)
#             _al_path = os.path.join(imgs_path, "al")
#             for file in os.listdir(_al_path):
#                 if not file.endswith("jpg"):
#                     continue
#                 al_imgs.append(cv2.imread(os.path.join(_al_path, file)))

#             _st_path = os.path.join(imgs_path, "st")
#             for file in os.listdir(_st_path):
#                 if not file.endswith("jpg"):
#                     continue
#                 st_imgs.append(cv2.imread(os.path.join(_st_path, file)))

#         for idx, img in enumerate(st_imgs):
#             cv2.imwrite(
#                 os.path.join(
#                     st_path,
#                     f"{idx:04}.jpg",
#                 ),
#                 img,
#             )
#         for idx, img in enumerate(al_imgs):
#             cv2.imwrite(
#                 os.path.join(
#                     al_path,
#                     f"{idx:04}.jpg",
#                 ),
#                 img,
#             )
