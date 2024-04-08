from os.path import join, exists, normpath
from os import listdir, makedirs
import numpy as np
import tkinter as tk
import cv2
from PIL import Image, ImageTk

from utils import get_imgs_paths
from values import (
    stiv_result_dir,
    ifft_res_dir,
    sti_res_dir,
    img_dir,
    valid_result_dir,
    valid_label_file,
    ananlyze_result_dir,
    valid_example_dir,
)

root = normpath(r"test\stiv_routine\root")

path_list = get_imgs_paths(root)

st_path = ifft_res_dir
al_path = sti_res_dir
or_path = img_dir

current_dir_index = 0
imgs_path = ""
al_imgs = []
al_tkimgs = []
st_imgs = []
st_tkimgs = []
current_img_index = 0
img_num = 0
ress = []
origin_imgs = []


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
        if exists(join(imgs_path, valid_result_dir, valid_label_file)):
            print(f"{imgs_path} exists valid_label")
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
        st_imgs.append(img.copy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        st_tkimgs.append(img)

    global current_img_index
    global img_num
    global ress
    current_img_index = -1
    img_num = len(st_tkimgs)
    ress = []
    nextImg()


def nextImg():
    global current_img_index
    current_img_index += 1
    print(f"img {current_img_index}")
    if current_img_index == img_num:
        return

    label1.configure(image=st_tkimgs[current_img_index])
    label2.configure(image=al_tkimgs[current_img_index])


def button2_click():
    if current_img_index == img_num:
        return
    ress.append(0)

    nextImg()


def button3_click():
    if current_img_index == img_num:
        return
    ress.append(1)

    nextImg()


def button4_click():
    global current_dir_index

    if current_img_index != img_num:
        return

    makedirs(join(imgs_path, valid_result_dir), exist_ok=True)
    np.save(join(imgs_path, valid_result_dir, valid_label_file), ress)

    current_dir_index += 1


def button5_click():
    global current_dir_index
    if current_img_index == img_num:
        return
    _res_path = join(ananlyze_result_dir, valid_example_dir)
    makedirs(_res_path, exist_ok=True)
    index = 0
    for i, _ in enumerate(listdir(_res_path)):
        index = i + 1
    cv2.imwrite(join(_res_path, f"{index:04}.jpg"), origin_imgs[current_img_index])
    print(f"img {current_img_index} save to valid_example {index:04}.jpg")


def on_key_press(event):
    if event.char == "a":
        button1_click()
    if event.char == "s":
        button4_click()
    if event.char == "q":
        button2_click()
    if event.char == "w":
        button3_click()
    if event.char == "d":
        button5_click()


window = tk.Tk()
label1 = tk.Label(window)
label2 = tk.Label(window)
button1 = tk.Button(window, text="a:执行下一批次", command=button1_click)
button2 = tk.Button(window, text="q:无效", command=button2_click)
button3 = tk.Button(window, text="w:有效", command=button3_click)
button4 = tk.Button(window, text="s:结果保存", command=button4_click)
button5 = tk.Button(window, text="d:收藏", command=button5_click)

label1.pack(side=tk.LEFT)
label2.pack(side=tk.LEFT)

button1.pack()
button2.pack()
button3.pack()
button4.pack()
button5.pack()

window.bind("<KeyPress>", on_key_press)

# 运行窗口的主循环
window.mainloop()
