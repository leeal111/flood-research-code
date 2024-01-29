import cv2
import numpy as np
from os.path import join, exists
from os import listdir, makedirs

from stiv_compute_routine import root, sum_data_dir, ifft_img_dir
from values import valid_result_dir, valid_label_file, valid_score_dir

train_data_dir = "datatrain"


def data_collect():
    ifftimgs = []
    sumlists = []
    labels = []
    for dir1 in listdir(root):
        for dir2 in listdir(join(root, dir1)):
            imgDir_path = join(root, dir1, dir2)

            # 需要首先完成数据整理
            if not exists(join(imgDir_path, valid_result_dir, valid_label_file)):
                print(f"{imgDir_path} not exists valid_label")
                continue

            current_img_index = 0
            for file in listdir(imgDir_path):
                if not file.endswith("jpg"):
                    continue
                _list_path = join(
                    imgDir_path, sum_data_dir, f"{current_img_index:04}.npy"
                )
                sum_list = np.load(_list_path)
                sumlists.append(sum_list)

                _ifft_path = join(
                    imgDir_path,
                    ifft_img_dir,
                    f"{current_img_index:04}.jpg",
                )
                ifft_img = cv2.imread(_ifft_path)
                image = ifft_img
                height, width, _ = image.shape
                center_x = width // 2
                center_y = height // 2
                crop_size = 2**9
                start_x = center_x - crop_size // 2
                start_y = center_y - crop_size // 2
                cropped_image = image[
                    start_y : start_y + crop_size, start_x : start_x + crop_size
                ]
                ifftimgs.append(cropped_image)
                current_img_index += 1

            labels += np.load(
                join(imgDir_path, valid_result_dir, valid_label_file)
            ).tolist()
    print(f"数据数：{len(labels)}")
    print(f"正数据数：{np.sum(np.array(labels)==1)}")
    print(f"负数据数：{np.sum(np.array(labels)==0)}")
    makedirs(train_data_dir, exist_ok=True)
    np.save(join(train_data_dir, "sumlists.npy"), np.array(sumlists))
    np.save(join(train_data_dir, "ifftimgs.npy"), np.array(ifftimgs))
    np.save(join(train_data_dir, "labels.npy"), np.array(labels))


def result_extract():
    nn_scores = np.load("datatrain\\nn_score.npy")
    current_img_index = 0
    for dir1 in listdir(root):
        for dir2 in listdir(join(root, dir1)):
            imgDir_path = join(root, dir1, dir2)

            # 需要首先完成数据整理
            if not exists(join(imgDir_path, valid_result_dir, valid_label_file)):
                print(f"{imgDir_path} not exists valid_result")
                continue
            lindex = current_img_index
            for file in listdir(imgDir_path):
                if not file.endswith("jpg"):
                    continue
                current_img_index += 1
            rindex = current_img_index
            np.save(
                join(imgDir_path, valid_score_dir, f"nn_ifftimg_result.npy"),
                np.array(nn_scores[lindex:rindex]),
            )


if __name__ == "__main__":
    result_extract()
    # data_collect()
