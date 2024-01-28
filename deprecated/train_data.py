import cv2
import numpy as np
from os.path import join, exists
from os import listdir
from key_value import kvs

root = kvs.root

ifftimgs = []
sumlists = []
labels = []
result_names = [
    "v1_list_score",
    "svm_list_result",
    "signal_noise_radio_list_score",
]
results = [[] for _ in result_names]

for dir1 in listdir(root):
    for dir2 in listdir(join(root, dir1)):
        imgDir_path = join(root, dir1, dir2)

        # 需要首先完成数据整理
        if not exists(join(imgDir_path, kvs.validResDir, kvs.validRealFileName)):
            print(f"{imgDir_path} not exists valid_result")
            continue

        current_img_index = 0
        for file in listdir(imgDir_path):
            if not file.endswith("jpg"):
                continue
            list_path = kvs.sumlistDir
            _list_path = join(imgDir_path, list_path, f"{current_img_index:04}.npy")
            sum_list = np.load(_list_path)
            sumlists.append(sum_list)

            _ifft_path = join(
                imgDir_path, "result_sotabase", "06_ifft", f"{current_img_index:04}.jpg"
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
            join(imgDir_path, kvs.validResDir, kvs.validRealFileName)
        ).tolist()

        for i, name in enumerate(result_names):
            results[i] += np.load(
                join(imgDir_path, kvs.validResDir, f"{name}.npy")
            ).tolist()

print(f"数据数：{len(labels)}")
print(f"正数据数：{np.sum(np.array(labels)==1)}")
print(f"负数据数：{np.sum(np.array(labels)==0)}")
np.save(join(kvs.trainDataResDir, "sumlists.npy"), np.array(sumlists))
np.save(join(kvs.trainDataResDir, "ifftimgs.npy"), np.array(ifftimgs))
np.save(join(kvs.trainDataResDir, "labels.npy"), np.array(labels))
for i, name in enumerate(result_names):
    np.save(join(kvs.trainDataResDir, f"{name}.npy"), np.array(results[i]))
