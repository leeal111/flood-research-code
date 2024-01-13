from os.path import join, exists
from os import listdir, makedirs
import numpy as np
import tkinter as tk
import cv2
from PIL import Image, ImageTk

from valid_compute_imp import validScoreMethods
from stiv_compute_routine_imp import ifRight2LeftForLoc
from key_value import kvs

root = kvs.root
res_path = kvs.validResDir
scoreMethods = validScoreMethods

# 依赖的前置操作
st_path = kvs.sumlistPicDir
al_path = kvs.stivResPicDir

dir_list = []
for dir1 in listdir(root):
    for dir2 in listdir(join(root, dir1)):
        imgDir_path = join(root, dir1, dir2)

        # 需要首先完成数据整理以及算法结果输出
        if not exists(join(imgDir_path, al_path)):
            print(f"{imgDir_path} not exists al_result")
            continue

        # 检查是否存在人工标注结果，存在则忽略
        if exists(join(imgDir_path, res_path, kvs.validRealFileName)):
            continue
        dir_list.append(imgDir_path)

current_dir_index = 0
imgDir_path = ""
al_imgs = []
al_tkimgs = []
st_imgs = []
st_tkimgs = []
origin_imgs = []
current_img_index = 0
img_num = 0
ress = []


def ifFlip(img, path):
    if ifRight2LeftForLoc(path) == True:
        return cv2.flip(img, 1)
    else:
        return img


def button1_click():
    # 初始化变量
    global imgDir_path
    imgDir_path = dir_list[current_dir_index]
    print(imgDir_path)

    # 读取原始图片origin_imgs
    global origin_imgs
    origin_imgs = []
    for file in listdir(imgDir_path):
        if not file.endswith(".jpg"):
            continue
        img = ifFlip(cv2.imread(join(imgDir_path, file)), imgDir_path)
        origin_imgs.append(img.copy())

    # 读取算法图片al_imgs、al_tkimgs
    global al_imgs
    global al_tkimgs
    al_imgs = []
    al_tkimgs = []
    _al_path = join(imgDir_path, al_path)
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
    _st_path = join(imgDir_path, st_path)
    for file in listdir(_st_path):
        if not file.endswith(".jpg"):
            continue
        img = ifFlip(cv2.imread(join(_st_path, file)), imgDir_path)
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
    if current_img_index == img_num:
        return

    label1.configure(image=st_tkimgs[current_img_index])
    label2.configure(image=al_tkimgs[current_img_index])

    for i, met in enumerate(scoreMethods):
        scores[i].set(str(met(imgDir_path, current_img_index)))


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

    makedirs(join(imgDir_path, res_path), exist_ok=True)
    np.save(join(imgDir_path, res_path, f"result.npy"), ress)

    current_dir_index += 1


window = tk.Tk()
label1 = tk.Label(window)
label2 = tk.Label(window)
button1 = tk.Button(window, text="执行下一批次", command=button1_click)
button2 = tk.Button(window, text="无效", command=button2_click)
button3 = tk.Button(window, text="有效", command=button3_click)
button4 = tk.Button(window, text="结果保存", command=button4_click)


label1.pack(side=tk.LEFT)
label2.pack(side=tk.LEFT)

scores = []
text_labels = []
for met in scoreMethods:
    score = tk.StringVar()
    score.set(f"{met.__name__}")
    text_label = tk.Label(window, textvariable=score)
    text_label.pack()
    scores.append(score)
    text_labels.append(text_label)

button1.pack()
button2.pack()
button3.pack()
button4.pack()

# 运行窗口的主循环
window.mainloop()
