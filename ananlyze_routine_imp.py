from os import listdir, makedirs
import shutil
import cv2
import numpy as np
from os.path import join,exists
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from display import pr_img, roc_img
from utils import call_for_imgss, compute_f1_score, compute_mean_precision, get_imgs_paths
from stiv_compute_routine_imp import root,img_dir
from valid_compute_imp import valid_score_methods,valid_score_dir,valid_result_dir,valid_label_file

ananlyze_result_dir = "result"

def ananlyze_valid_wrong_call(imgs_path):
    if not exists(join(imgs_path, valid_result_dir, valid_label_file)):
        print(f"{imgs_path} not exists valid_label")
        return
    print(imgs_path)
    ans = np.load(join(imgs_path, valid_result_dir, valid_label_file))
    for met in valid_score_methods:
        ress = np.load(
            join(imgs_path, valid_result_dir, f"{met.__name__}.npy")
        )

        for i,file in enumerate(listdir(join(imgs_path, img_dir))):
            _res_path = None
            if ans[i] == 1 and ress[i] == 0:# TN
                _res_path = join(
                    ananlyze_result_dir, "valid_wrong", "tn", met.__name__
                )
            if ans[i] == 0 and ress[i] == 1:# FP
                _res_path = join(
                    ananlyze_result_dir, "valid_wrong", "fp", met.__name__
                )  
            if ans[i] != ress[i]:#结果保存
                makedirs(_res_path, exist_ok=True)
                shutil.copy(
                    join(imgs_path, img_dir, file),
                    join(_res_path, f"{len(listdir(_res_path)):04}.jpg"),
                )
            i += 1

def ananlyze_result_wrong():
    shutil.rmtree(join(ananlyze_result_dir, "valid_wrong"), ignore_errors=True)
    path_list = get_imgs_paths(root)
    call_for_imgss(path_list,ananlyze_valid_wrong_call)

def ananlyze_valid_score_call(imgs_path, **kwarg):
    return np.load(join(imgs_path, valid_score_dir, f"{kwarg["method"].__name__}.npy"))

def ananlyze_valid_label_call(imgs_path, **kwarg):
    return np.load(join(imgs_path, valid_result_dir, f"result.npy"))    

def ananlyze_valid_ROC():
    res_path=join(ananlyze_result_dir, valid_score_dir+"_roc")
    shutil.rmtree(res_path, ignore_errors=True)
    path_list = get_imgs_paths(root)
    for met in valid_score_methods:
        
        ress = np.concatenate(call_for_imgss(path_list, ananlyze_valid_score_call, method=met))
        anss=np.concatenate(call_for_imgss(path_list, ananlyze_valid_label_call, method=met))

        ress_train, ress_test, anss_train, anss_test = train_test_split(
        ress,anss, test_size=0.2, random_state=0
    )
        
        name, ans, data=met.__name__,anss_test,ress_test
        fpr, tpr, _ = roc_curve(ans, data)
        auc_score = roc_auc_score(ans, data)
        makedirs(res_path,exist_ok=True)
        cv2.imwrite(join(res_path,name+"ROC.jpg"),roc_img(name, fpr, tpr, auc_score)) 

def ananlyze_valid_PR():
    res_path=join(ananlyze_result_dir, valid_score_dir+"_pr")
    shutil.rmtree(res_path, ignore_errors=True)
    path_list = get_imgs_paths(root)
    for met in valid_score_methods:
        ress = np.concatenate(call_for_imgss(path_list, ananlyze_valid_score_call, method=met))
        anss=np.concatenate(call_for_imgss(path_list, ananlyze_valid_label_call, method=met))
        ress_train, ress_test, anss_train, anss_test = train_test_split(ress,anss, test_size=0.2, random_state=0)
        
        name, ans, data=met.__name__,anss_test,ress_test
        precision, recall, _ = precision_recall_curve(ans, data)
        f1_score = compute_mean_precision(ans, data)
        makedirs(res_path,exist_ok=True)
        cv2.imwrite(join(res_path,name+"PR.jpg"),pr_img(name, precision, recall, f1_score)) 


