from os import listdir, makedirs
import shutil

import numpy as np
from stiv_compute_routine import root, img_dir
from os.path import join, exists
from valid_compute_imp import (
    valid_result_methods,
    valid_score_methods,
)
from values import (
    valid_result_dir,
    valid_label_file,
    valid_score_dir,
    ananlyze_result_dir,
)


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
            for met in valid_result_methods + valid_score_methods:
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
