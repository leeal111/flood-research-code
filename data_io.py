import cv2
import numpy as np
from os.path import join, exists
from os import listdir, makedirs

from ananlyze_routine_imp import ananlyze_result_dir
from stiv_compute_routine_imp import root, sum_data_dir, ifft_img_dir
from utils import call_for_imgss, get_imgs_data, get_imgs_paths
from valid_compute_imp import valid_result_dir, valid_label_file, valid_score_dir

data_save_path = join(ananlyze_result_dir, "data")


def data_out_call(imgs_path, **kwarg):
    if not exists(join(imgs_path, valid_result_dir, valid_label_file)):
        print(f"{imgs_path} not exists valid_label")
        return
    return get_imgs_data(imgs_path, kwarg["data_path"])


def label_out_call(imgs_path, **kwarg):
    if not exists(join(imgs_path, valid_result_dir, valid_label_file)):
        print(f"{imgs_path} not exists valid_label")
        return
    return np.load(join(imgs_path, valid_result_dir, valid_label_file))


def data_out(path_list):
    datas = call_for_imgss(
        path_list,
        data_out_call,
        data_path=ifft_img_dir,
    )
    datas = np.concatenate(datas)
    sumlists = call_for_imgss(
        path_list,
        data_out_call,
        data_path=sum_data_dir,
    )
    sumlists = np.concatenate(sumlists)
    labels = call_for_imgss(path_list, label_out_call)
    labels = np.concatenate(labels)

    print(f"数据数：{len(labels)}")
    print(f"正数据数：{np.sum(np.array(labels)==1)}")
    print(f"负数据数：{np.sum(np.array(labels)==0)}")
    makedirs(data_save_path, exist_ok=True)
    np.save(join(data_save_path, "datas.npy"), np.array(datas))
    np.save(join(data_save_path, "labels.npy"), np.array(labels))
    np.save(join(data_save_path, "sumlists.npy"), np.array(sumlists))


def data_in(path_list):
    nn_scores = np.load(join(data_save_path, "nn_score.npy"))
    current_img_index = 0
    for imgs_path in path_list:
        # 需要首先完成数据整理
        if not exists(join(imgs_path, valid_result_dir, valid_label_file)):
            print(f"{imgs_path} not exists valid_result")
            continue
        lindex = current_img_index
        current_img_index += len(listdir(join(imgs_path, sum_data_dir)))
        rindex = current_img_index
        np.save(
            join(imgs_path, valid_score_dir, f"nn_ifftimg_result.npy"),
            np.array(nn_scores[lindex:rindex]),
        )


if __name__ == "__main__":
    path_list = get_imgs_paths(root)
    data_out(path_list)
    # data_in(path_list)
