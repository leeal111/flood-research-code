from os import listdir, makedirs
import shutil
import cv2

import numpy as np
from utils import call_for_imgss, get_imgs_paths
from os.path import exists, join, isdir, normpath
from valid_compute_imp import valid_label_file, valid_result_dir
from stiv_compute_routine_imp import root


def get_noisy_img(img):
    pass


def add_white_noise(image, amount=0.02):
    # 在图像中添加椒盐噪声
    noisy_image = np.copy(image)
    h, w, _ = noisy_image.shape
    num_pixels = int(amount * h * w)

    # 将随机选取的像素设置为最小值或最大值
    h_coords = np.random.randint(0, h, (num_pixels, 1))
    w_coords = np.random.randint(0, w, (num_pixels, 1))
    noisy_image[h_coords, w_coords] = [255, 255, 255]  # 设置为白色噪声

    return noisy_image


def add_gaussian_noise(image, mean=0, stddev=0.5):
    # 生成与图像大小相同的高斯噪声
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)

    # 将噪声与图像相加
    noisy_image = cv2.add(image, noise)

    return noisy_image


def generate_noisy_dataset_call(imgs_path, **kwarg):
    src = kwarg["src"]
    dst = kwarg["dst"]
    method = kwarg["noisy_method"]
    dst_path = imgs_path.replace(src, dst)
    makedirs(dst_path)

    valid_label_file_path = join(imgs_path, valid_result_dir, valid_label_file)
    assert exists(valid_label_file_path), f"{imgs_path} not exist valid_label_file"
    dst_valid_result_path = join(dst_path, valid_result_dir)
    makedirs(dst_valid_result_path)
    shutil.copy(valid_label_file_path, dst_valid_result_path)

    for file in listdir(imgs_path):
        src_file_path = join(imgs_path, file)
        if isdir(src_file_path):
            continue
        image = cv2.imread(src_file_path)
        noisy_image = method(image)
        cv2.imwrite(join(dst_path, file), noisy_image)


def generate_noisy_dataset(src, dst, noisy_method):
    assert not exists(dst)
    src_list = get_imgs_paths(src)
    call_for_imgss(
        src_list,
        generate_noisy_dataset_call,
        src=src,
        dst=dst,
        noisy_method=noisy_method,
    )


def test_add_noise():
    image = cv2.imread(normpath(r"test\0010.jpg"))
    noisy_image = add_white_noise(image)
    # noisy_image = add_gaussian_noise(image)
    cv2.imshow("Original Image", image)
    cv2.imshow("Noisy Image", noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src_root = "data_base"
    generate_noisy_dataset(src_root, "data_white_noise", noisy_method=add_white_noise)
    generate_noisy_dataset(
        src_root, "data_gaussian_noise", noisy_method=add_gaussian_noise
    )
    # test_add_noise()
