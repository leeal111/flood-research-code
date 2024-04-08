import tkinter as tk
import cv2
from os.path import join, exists, isdir, normpath
from os import listdir, makedirs
from PIL import Image, ImageTk
import numpy as np
from values import (
    ananlyze_result_dir,
    correct_result_dir,
    correct_al_result_file,
    correct_st_result_file,
    correct_al_result_file_u,
    correct_st_result_file_u,
    stiv_result_dir,
    hw_img_dir,
    sti_res_dir,
    valid_label_file,
    img_dir,
    valid_result_dir,
)

from utils import get_imgs_paths, imgs_if_R2L

root = normpath(r"test\stiv_routine\root")
st_path = hw_img_dir
al_path = sti_res_dir
or_path = img_dir
valid_result_path = join(valid_result_dir, valid_label_file)
valid_result_path_u = join(valid_result_dir, f"signal_noise_radio_list_score.npy")
path_list = get_imgs_paths(root)

current_dir_index = 0
imgs_path = ""
al_imgs = []
al_tkimgs = []
st_imgs = []
st_tkimgs = []
current_img_index = 0
img_num = 0
al_ress = []
st_ress = []
al_ress_u = []
st_ress_u = []
valid_result = []
origin_imgs = []
valid_data_u = []


def button1_click():
    # 初始化变量
    global imgs_path
    global current_dir_index

    if current_dir_index == len(path_list):
        print(f"not imgs more!")
        return

    imgs_path = path_list[current_dir_index]
    while 1:
        # 检查是否计算以及是否已经标注
        if not exists(join(imgs_path, stiv_result_dir)):
            print(f"{imgs_path} not exists stiv_result")
            current_dir_index += 1
            if current_dir_index == len(path_list):
                print(f"not imgs more!")
                return
            imgs_path = path_list[current_dir_index]
            continue
        if exists(join(imgs_path, correct_result_dir, correct_al_result_file_u)):
            print(f"{imgs_path} exists al_result")
            current_dir_index += 1
            if current_dir_index == len(path_list):
                print(f"not imgs more!")
                return
            imgs_path = path_list[current_dir_index]
            continue
        break
    print(imgs_path)

    # 读取原始图片origin_imgs
    global origin_imgs
    origin_imgs = []
    _or_path = join(imgs_path, or_path)
    for file in listdir(_or_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(join(_or_path, file))
        origin_imgs.append(img.copy())

    # 读取算法图片al_imgs、al_tkimgs
    global al_imgs
    global al_tkimgs
    al_imgs = []
    al_tkimgs = []
    _al_path = join(imgs_path, al_path)
    for file in listdir(_al_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(join(_al_path, file))
        al_imgs.append(img.copy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        al_tkimgs.append(img)

    # 读取站点图片st_imgs、st_tkimgs
    global st_imgs
    global st_tkimgs
    st_imgs = []
    st_tkimgs = []
    _st_path = join(imgs_path, st_path)
    for file in listdir(_st_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(join(_st_path, file))
        if imgs_if_R2L(imgs_path):
            img = cv2.flip(img, 1)
        st_imgs.append(img.copy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        st_tkimgs.append(img)

    global current_img_index
    global img_num
    global al_ress_u
    global st_ress_u
    global valid_data
    global valid_data_u
    global al_ress
    global st_ress
    current_img_index = -1
    img_num = len(st_tkimgs)
    al_ress_u = []
    st_ress_u = []
    valid_data = np.load(join(imgs_path, valid_result_path))
    valid_data_u = np.load(join(imgs_path, valid_result_path_u))
    al_ress = np.load(join(imgs_path, correct_result_dir, correct_al_result_file))
    st_ress = np.load(join(imgs_path, correct_result_dir, correct_st_result_file))
    print(valid_data)
    print(valid_data_u)
    print(al_ress)
    print(st_ress)
    nextImg()


def nextImg():
    global current_img_index
    current_img_index += 1
    if current_img_index == img_num:
        return
    while True:
        if valid_data_u[current_img_index] == valid_data[current_img_index]:
            al_ress_u.append(al_ress[current_img_index])
            st_ress_u.append(st_ress[current_img_index])
            print(current_img_index, "not change")
        elif valid_data_u[current_img_index] == 0:
            al_ress_u.append(-1)
            st_ress_u.append(-1)
            print(current_img_index, "invalid")
        else:
            break
        current_img_index += 1
        if current_img_index == img_num:
            return
    print(current_img_index)
    label1.configure(image=st_tkimgs[current_img_index])
    label2.configure(image=al_tkimgs[current_img_index])


def button2_click():
    if current_img_index == img_num:
        return
    al_ress_u.append(1)
    st_ress_u.append(1)
    nextImg()


def button3_click():
    if current_img_index == img_num:
        return
    al_ress_u.append(1)
    st_ress_u.append(0)
    nextImg()


def button4_click():
    if current_img_index == img_num:
        return
    al_ress_u.append(0)
    st_ress_u.append(1)
    nextImg()


def button5_click():
    if current_img_index == img_num:
        return
    al_ress_u.append(0)
    st_ress_u.append(0)
    nextImg()


def button6_click():
    global current_dir_index

    if current_img_index != img_num:
        return

    makedirs(join(imgs_path, correct_result_dir), exist_ok=True)
    np.save(join(imgs_path, correct_result_dir, correct_al_result_file_u), al_ress_u)
    np.save(join(imgs_path, correct_result_dir, correct_st_result_file_u), st_ress_u)

    current_dir_index += 1


def button7_click():
    global current_dir_index
    if current_img_index == img_num:
        return
    _res_path = join(ananlyze_result_dir, "correct_example")
    makedirs(_res_path, exist_ok=True)
    _site_res_path = join(ananlyze_result_dir, "correct_example", hw_img_dir)
    makedirs(_site_res_path, exist_ok=True)

    index = 0
    for i, file in enumerate(listdir(_res_path)):
        if isdir(join(_res_path), file):
            continue
        index = i + 1
    cv2.imwrite(join(_res_path, f"{index:04}.jpg"), origin_imgs[current_img_index])
    cv2.imwrite(join(_site_res_path, f"{index:04}.jpg"), st_imgs[current_img_index])
    print(f"{index:04}.jpg")


def on_key_press(event):
    if event.char == "a":
        button1_click()
    if event.char == "s":
        button6_click()
    if event.char == "q":
        button2_click()
    if event.char == "w":
        button3_click()
    if event.char == "e":
        button4_click()
    if event.char == "r":
        button5_click()
    if event.char == "d":
        button7_click()


# 创建UI
window = tk.Tk()
label1 = tk.Label(window)
label2 = tk.Label(window)
label1.pack(side=tk.LEFT)
label2.pack(side=tk.LEFT)
button1 = tk.Button(window, text="a:执行下一批次", command=button1_click)
button2 = tk.Button(window, text="q:都对", command=button2_click)
button3 = tk.Button(window, text="w:算法对", command=button3_click)
button4 = tk.Button(window, text="e:站点对", command=button4_click)
button5 = tk.Button(window, text="r:全错", command=button5_click)
button6 = tk.Button(window, text="s:结果保存", command=button6_click)
button7 = tk.Button(window, text="d:收藏", command=button7_click)
button1.pack()
button2.pack()
button3.pack()
button4.pack()
button5.pack()
button6.pack()
window.bind("<KeyPress>", on_key_press)
window.mainloop()
