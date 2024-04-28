from os.path import basename, dirname, join, exists, normpath
from os import listdir, remove
import shutil

import cv2
import numpy as np

from display import line_chart_img


def crop_img(img, crop_size=2**9):
    image = img
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    start_x = center_x - crop_size // 2
    start_y = center_y - crop_size // 2
    cropped_image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    return cropped_image


def get_imgs_datas(
    imgs_path,
    datas_dir,
):
    datas = []
    datas_path = join(imgs_path, datas_dir)
    for file in listdir(datas_path):
        data_path = join(datas_path, file)
        if file.endswith("jpg"):
            data = cv2.imread(data_path)
        elif file.endswith("npy"):
            data = np.load(data_path)
        else:
            print("get_imgs_data error")
            data = []
        datas.append(data)
    return datas


def get_imgs_data(
    imgs_path,
    data_dir,
):
    data_path = join(imgs_path, data_dir)
    if not exists(data_path):
        print("get_imgs_data error:{imgs_path} img_data not exists")
        return None
    if data_dir.endswith("jpg"):
        data = cv2.imread(data_path)
    elif data_dir.endswith("npy"):
        data = np.load(data_path)
    else:
        print("get_imgs_data error:{imgs_path} wrong tailfix")
        data = None
    return data


def call_for_imgss(imgs_paths, call_func, *arg, **kwarg):
    ress = []
    for imgs_path in imgs_paths:
        res = call_func(imgs_path, *arg, **kwarg)
        if res is None:
            continue
        ress.append(res)
    return ress


def get_imgs_paths(root):
    path_list = []
    for dir1 in listdir(root):
        for dir2 in listdir(join(root, dir1)):
            imgs_path = join(root, dir1, dir2)
            path_list.append(imgs_path)
    return path_list


def imgs_if_R2L(imgs_path):
    dic = {
        "ddh": False,
        "jx": True,
        "mc": False,
        "ah": True,
        "fj": True,
        "gx": False,
        "jxdx": True,
        "yc": False,
        "hd": True,
        "ys": True,
    }
    imgs_path_ = imgs_path
    while True:
        dir1 = basename(imgs_path)
        if dir1 in dic:
            return dic[dir1]

        if imgs_path == dirname(imgs_path):
            print(f"{imgs_path_}: unknown ifRightToLeft")
            return False
        else:
            imgs_path = dirname(imgs_path)


def imgss_del_call(imgs_path, del_dir):
    if del_dir == "" or del_dir == None:
        del_dir = "shield"
    del_path = join(imgs_path, del_dir)
    shutil.rmtree(
        del_path,
        ignore_errors=True,
    )
    if exists(del_path):
        remove(del_path)


def img_strength():
    name = "00061"
    img_path = normpath(rf"C:\Users\leeal\Desktop\sucai\{name}.jpg")
    image = cv2.imread(normpath(img_path), 0)

    # 定义阈值和增强值
    # threshold = 15
    # enhancement_value = 100
    threshold = 30
    enhancement_value = 100

    # 遍历图像的每个像素
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            # 获取当前像素的值
            pixel_value = image[y, x]

            # 判断当前像素的值是否大于阈值
            if pixel_value > threshold:
                # 增强当前像素的值
                enhanced_value = min(pixel_value + enhancement_value, 255)

                # 更新增强后的像素值
                image[y, x] = enhanced_value

    new_img = image
    cv2.imshow("img", new_img)
    cv2.waitKey(0)
    cv2.imwrite(normpath(r"C:\Users\leeal\Desktop\out.jpg"), new_img)

if __name__ == "__main__":
    # img_strength()
    list_str = "9.80616e+08 1.64359e+09 3.01448e+09 3.77477e+09 4.10932e+09 4.25065e+09 4.3834e+09 4.45643e+09 4.21144e+09 4.01638e+09 4.17618e+09 4.56382e+09 4.93322e+09 5.08948e+09 5.2818e+09 5.64076e+09 6.17215e+09 6.73695e+09 7.32199e+09 8.06782e+09 8.80017e+09 9.63527e+09 1.06172e+10 1.21698e+10 1.4218e+10 1.64617e+10 1.89513e+10 2.18014e+10 2.54009e+10 2.95733e+10 3.32858e+10 3.46431e+10 3.29896e+10 2.92193e+10 2.52251e+10 2.15298e+10 1.80014e+10 1.48977e+10 1.24817e+10 1.07976e+10 9.40666e+09 8.21399e+09 7.391e+09 6.96923e+09 6.99125e+09 7.30003e+09 7.57161e+09 7.4627e+09 6.9525e+09 6.30898e+09 5.83506e+09 5.51372e+09 5.29835e+09 5.03122e+09 4.71934e+09 4.58988e+09 4.61505e+09 4.49449e+09 4.09487e+09 3.71857e+09 3.62078e+09 3.69011e+09 3.64192e+09 3.42972e+09 3.01342e+09 2.59247e+09 2.47843e+09 2.70731e+09 3.14198e+09 3.50842e+09 3.70303e+09 3.77264e+09 3.78337e+09 3.7206e+09 3.5719e+09 3.43669e+09 3.42854e+09 3.45969e+09 3.33063e+09 3.07068e+09 2.90891e+09 3.01909e+09 3.15445e+09 3.00398e+09 2.68716e+09 2.44471e+09 2.4764e+09 2.44579e+09 2.21667e+09 1.35718e+09 "
    num_list = [float(num) for num in list_str.split()]
    cv2.imshow("img",line_chart_img(num_list))
    cv2.waitKey(0)
    
