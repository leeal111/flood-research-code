from os import listdir
import shutil
import numpy as np
from os.path import join,exists

from sklearn.metrics import roc_auc_score, roc_curve
from display import ROC_plot
from utils import call_for_imgss, get_imgs_paths
from stiv_compute_routine_imp import root
from valid_compute_imp import valid_score_methods,valid_score_dir,valid_result_dir,valid_label_file

ananlyze_result_dir = "result"


def ananlyze_result_wrong():
    shutil.rmtree(join(ananlyze_result_dir, "valid_wrong"), ignore_errors=True)
    for dir1 in listdir(root):
        for dir2 in listdir(join(root, dir1)):
            imgDir_path = join(root, dir1, dir2)
            print(imgDir_path)

            if not exists(join(imgDir_path, valid_result_dir, valid_label_file)):
                print(f"{imgDir_path} not exists valid_label")
                continue

            ans = np.load(join(imgDir_path, valid_result_dir, valid_label_file))
            for met in valid_score_methods:
                ress = np.load(
                    join(imgDir_path, valid_result_dir, f"{met.__name__}.npy")
                )

                current_img_index = 0
                for file in listdir(join(imgDir_path, img_dir)):
                    if not file.endswith("jpg"):
                        continue
                    _res_path = None
                    if ans[current_img_index] == 1 and ress[current_img_index] == 0:
                        _res_path = join(
                            ananlyze_result_dir, "valid_wrong", "tn", met.__name__
                        )
                        makedirs(_res_path, exist_ok=True)
                    if ans[current_img_index] == 0 and ress[current_img_index] == 1:
                        _res_path = join(
                            ananlyze_result_dir, "valid_wrong", "fp", met.__name__
                        )
                        makedirs(_res_path, exist_ok=True)
                    if ans[current_img_index] != ress[current_img_index]:
                        index = 0
                        for i, _ in enumerate(listdir(_res_path)):
                            index = i + 1
                        shutil.copy(
                            join(imgDir_path, img_dir, file),
                            join(_res_path, f"{index:04}.jpg"),
                        )
                        print(f"{index:04}.jpg")
                    current_img_index += 1


def ananlyze_valid_ROC():
    path_list = get_imgs_paths(root)
    for met in valid_score_methods:
        ress = call_for_imgss(path_list, ananlyze_valid_score_call, method=met)
        anss=call_for_imgss(path_list, ananlyze_valid_label_call, method=met)
        ROCplot()

def ananlyze_valid_score_call(imgs_path, **kwarg):
    return np.load(join(imgs_path, valid_score_dir, f"{kwarg["method"].__name__}.npy"))

def ananlyze_valid_label_call(imgs_path, **kwarg):
    return np.load(join(imgs_path, valid_result_dir, f"result.npy"))


def ROCplot(name, ans, data):
    fpr, tpr, thresholds = roc_curve(ans, data, pos_label=1)
    auc_score = roc_auc_score(ans, data)
    ROC_plot(name, fpr, tpr, auc_score)
    print("AUC: {:.4f}".format(auc_score))


