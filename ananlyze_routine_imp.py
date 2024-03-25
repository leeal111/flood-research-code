from os import listdir, makedirs
import shutil
import cv2
import numpy as np
from os.path import join,exists,basename,dirname
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from display import pr_img, roc_img
from utils import call_for_imgss, compute_f1_score, compute_mean_precision, get_imgs_paths
from stiv_compute_routine_imp import imgs_if_R2L, root,img_dir,site_img_dir
from valid_compute_imp import valid_score_methods,valid_score_dir,valid_result_dir,valid_label_file
ananlyze_result_dir = "result"

def ifFlip(img, path):
    if imgs_if_R2L(path):
        return cv2.flip(img, 1)
    else:
        return img
    
correct_result_dir = "correct_result"
correct_al_result_file = "al_result.npy"
correct_st_result_file = "st_result.npy"

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

def ananlyze_valid_result_wrong():
    shutil.rmtree(join(ananlyze_result_dir, "valid_wrong"), ignore_errors=True)
    path_list = get_imgs_paths(root)
    call_for_imgss(path_list,ananlyze_valid_wrong_call)

def ananlyze_valid_score_call(imgs_path, **kwarg):
    return np.load(join(imgs_path, valid_score_dir, f"{kwarg["method"].__name__}.npy"))

def ananlyze_valid_label_call(imgs_path, **kwarg):
    return np.load(join(imgs_path, valid_result_dir, valid_label_file))    

def ananlyze_valid_ROC(if_train_test_split=False ):
    res_path=join(ananlyze_result_dir, valid_score_dir+"_roc")
    shutil.rmtree(res_path, ignore_errors=True)
    path_list = get_imgs_paths(root)
    for met in valid_score_methods:
        
        ress = np.concatenate(call_for_imgss(path_list, ananlyze_valid_score_call, method=met))
        anss=np.concatenate(call_for_imgss(path_list, ananlyze_valid_label_call, method=met))

        if if_train_test_split==True:
            ress_train, ress_test, anss_train, anss_test = train_test_split(ress,anss, test_size=0.2, random_state=0)
            name, ans, data=met.__name__,anss_test,ress_test
        else:
            name, ans, data=met.__name__,anss,ress
        fpr, tpr, _ = roc_curve(ans, data)
        auc_score = roc_auc_score(ans, data)
        makedirs(res_path,exist_ok=True)
        cv2.imwrite(join(res_path,name+"ROC.jpg"),roc_img(name, fpr, tpr, auc_score)) 

def ananlyze_valid_PR(if_train_test_split=False ):
    res_path=join(ananlyze_result_dir, valid_score_dir+"_pr")
    shutil.rmtree(res_path, ignore_errors=True)
    path_list = get_imgs_paths(root)
    for met in valid_score_methods:
        ress = np.concatenate(call_for_imgss(path_list, ananlyze_valid_score_call, method=met))
        anss=np.concatenate(call_for_imgss(path_list, ananlyze_valid_label_call, method=met))
        
        if if_train_test_split==True:
            ress_train, ress_test, anss_train, anss_test = train_test_split(ress,anss, test_size=0.2, random_state=0)
            name, ans, data=met.__name__,anss_test,ress_test
        else:
            name, ans, data=met.__name__,anss,ress
        
        precision, recall, thos = precision_recall_curve(ans, data)
        # print(f"{name} thos: {thos}")
        f1_score = compute_mean_precision(ans, data)
        makedirs(res_path,exist_ok=True)
        cv2.imwrite(join(res_path,name+"PR.jpg"),pr_img(name, precision, recall, f1_score)) 
    
def ananlyze_correct_wrong_call(imgs_path):
    if not exists(join(imgs_path, correct_result_dir, correct_al_result_file)):
        print(f"{imgs_path} not exists correct_result")
        return
    print(imgs_path)
    al_ans = np.load(join(imgs_path, correct_result_dir, correct_al_result_file))
    st_ans = np.load(join(imgs_path, correct_result_dir, correct_st_result_file))

    for i,(img_file,site_file) in enumerate(zip(listdir(join(imgs_path, img_dir)),listdir(join(imgs_path, site_img_dir)))):
        _res_path = None
        if al_ans[i] == 1 and st_ans[i] == 0:
            _res_path = join(
                ananlyze_result_dir, "correct_wrong", "al_ans-1_st_ans-0", 
            )
        elif al_ans[i] == 0 and st_ans[i] == 1:
            _res_path = join(
                ananlyze_result_dir, "correct_wrong", "al_ans-0_st_ans-1", 
            ) 
        elif al_ans[i] == 0 and st_ans[i] == 0:
            _res_path = join(
                ananlyze_result_dir, "correct_wrong", "al_ans-0_st_ans-0", 
            ) 
        else :
            continue
        makedirs(_res_path, exist_ok=True)
        makedirs(join(_res_path,site_img_dir), exist_ok=True)
        shutil.copy(
            join(imgs_path, img_dir, img_file),
            join(_res_path, f"{len(listdir(_res_path))-1:04}.jpg"),
        )
        site_img=cv2.imread(join(imgs_path, site_img_dir, site_file))
        cv2.imwrite(join(_res_path,site_img_dir, f"{len(listdir(_res_path))-2:04}.jpg"),ifFlip(site_img,imgs_path))

def ananlyze_correct_result_wrong():
    shutil.rmtree(join(ananlyze_result_dir, "correct_wrong"), ignore_errors=True)
    path_list = get_imgs_paths(root)
    call_for_imgss(path_list,ananlyze_correct_wrong_call)

def ananlyze_correct_al_call(imgs_path):
    if not exists(join(imgs_path, correct_result_dir, correct_al_result_file)):
        print(f"{imgs_path} not exists correct_result")
        return
    print(imgs_path)
    al_ans = np.load(join(imgs_path, correct_result_dir, correct_al_result_file))
    res=[basename(dirname(imgs_path)),basename(imgs_path),len(al_ans),len(al_ans)-np.count_nonzero(al_ans == -1),np.count_nonzero(al_ans == 0),np.count_nonzero(al_ans == 1)]
    return res

def ananlyze_correct_al_result():
    shutil.rmtree(join(ananlyze_result_dir, "correct_al"), ignore_errors=True)
    path_list = get_imgs_paths(root)
    ress=call_for_imgss(path_list,ananlyze_correct_al_call)
    
    makedirs(join(ananlyze_result_dir, "correct_al"),exist_ok=True)
    with open(join(ananlyze_result_dir, "correct_al","res.txt"),'w') as file:
        location=""
        count=[-1,-1,-1,  -1]
        total_count=[1,1,1,1]
        for res in ress:
            if res[0]!=location:
                file.write(f"{location} {count[0]} {count[1]} {count[2]} {count[3]} {count[3]/count[1]*100:.2f}\n")
                total_count[0]+=count[0]
                total_count[1]+=count[1]
                total_count[2]+=count[2]
                total_count[3]+=count[3]
                count=[0,0,0,  0]
                location=res[0]
            count[0]+=res[2]
            count[1]+=res[3]
            count[2]+=res[4]
            count[3]+=res[5]
        total_count[0]+=count[0]
        total_count[1]+=count[1]
        total_count[2]+=count[2]
        total_count[3]+=count[3]
        file.write(f"{location} {count[0]} {count[1]} {count[2]} {count[3]} {count[3]/count[1]*100:.2f}\n")
        file.write(f"total {total_count[0]} {total_count[1]} {total_count[2]} {total_count[3]} {total_count[3]/total_count[1]*100:.2f}\n")

    
def ananlyze_correct_compare_call(imgs_path):
    if not exists(join(imgs_path, correct_result_dir, correct_al_result_file)):
        print(f"{imgs_path} not exists correct_result")
        return
    print(imgs_path)
    al_ans = np.load(join(imgs_path, correct_result_dir, correct_al_result_file))
    st_ans = np.load(join(imgs_path, correct_result_dir, correct_st_result_file))
    res=[basename(dirname(imgs_path)),basename(imgs_path),len(al_ans)-np.count_nonzero(al_ans == -1),np.count_nonzero(al_ans == 1),np.count_nonzero(st_ans == 1),np.count_nonzero(al_ans == 1)-np.count_nonzero(st_ans == 1)]
    return res

def ananlyze_correct_compare_result():
    shutil.rmtree(join(ananlyze_result_dir, "correct_compare"), ignore_errors=True)
    path_list = get_imgs_paths(root)
    ress=call_for_imgss(path_list,ananlyze_correct_compare_call)
    
    makedirs(join(ananlyze_result_dir, "correct_compare"),exist_ok=True)
    with open(join(ananlyze_result_dir, "correct_compare","res.txt"),'w') as file:
        location=""
        count=[-1,-1,-1,  -1]
        total_count=[1,1,1,1]
        for res in ress:
            if res[0]!=location:
                file.write(f"{location} {count[0]} {count[1]} {count[2]} {count[3]} {count[3]/count[0]*100:.2f}\n")
                total_count[0]+=count[0]
                total_count[1]+=count[1]
                total_count[2]+=count[2]
                total_count[3]+=count[3]
                count=[0,0,0,  0]
                location=res[0]
            count[0]+=res[2]
            count[1]+=res[3]
            count[2]+=res[4]
            count[3]+=res[5]
        total_count[0]+=count[0]
        total_count[1]+=count[1]
        total_count[2]+=count[2]
        total_count[3]+=count[3]  
        file.write(f"{location} {count[0]} {count[1]} {count[2]} {count[3]} {count[3]/count[0]*100:.2f}\n")
        file.write(f"total {total_count[0]} {total_count[1]} {total_count[2]} {total_count[3]} {total_count[3]/total_count[0]*100:.2f}\n")
