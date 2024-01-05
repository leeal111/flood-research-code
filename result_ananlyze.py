import tkinter as tk
import cv2
import os
from PIL import Image, ImageTk
import method.stiv as method
import shutil

root = "valid"  # valid temp
resPath = "ananlyze" + ""
st = "st"
al = "al"
st_path = "hwMot"
al_path = os.path.join("result_sotabase", "10_STIRES")
type_list = [
    "invalid",
    "total_right",
    "al_right",
    "st_right",
    "total_wrong",
    "wrong_imgs",
]


def ifFlip(img, path):
    if root != "temp":
        return img

    dir1 = os.path.basename(os.path.dirname(path))
    if dir1 == "ddh":
        img = img
    elif dir1 == "jx":
        img = cv2.flip(img, 1)
    elif dir1 == "mc":
        img = img
    elif dir1 == "ah":
        img = cv2.flip(img, 1)
    elif dir1 == "fj":
        img = cv2.flip(img, 1)
    elif dir1 == "gx":
        img = img
    else:
        pass

    return img


def nextImg():
    global current_img_index
    global al_tkimgs
    global st_tkimgs

    if len(st_tkimgs) - 1 == current_img_index:
        return
    current_img_index += 1
    label1.configure(image=al_tkimgs[current_img_index])
    label2.configure(image=st_tkimgs[current_img_index])


def imgSave(type_name):
    global totol_num
    totol_num += 1
    img_path = os.path.join(dir_list[current_dir_index], resPath, type_name)
    cv2.imwrite(
        os.path.join(
            img_path,
            st,
            f"{current_img_index:04}.jpg",
        ),
        st_imgs[current_img_index],
    )
    cv2.imwrite(
        os.path.join(img_path, al, f"{current_img_index:04}.jpg"),
        al_imgs[current_img_index],
    )

    if type_name == type_list[3] or type_name == type_list[4]:
        cv2.imwrite(
            os.path.join(
                dir_list[current_dir_index],
                resPath,
                type_list[5],
                f"{current_img_index:04}.jpg",
            ),
            origin_imgs[current_img_index],
        )

    if root == "temp":
        if type_name != type_list[0]:
            cv2.imwrite(
                os.path.join(new_imgDir_path, f"{current_img_index:04}.jpg"),
                origin_imgs[current_img_index],
            )
            cv2.imwrite(
                os.path.join(new_imgDir_path, "hwMot", f"{current_img_index:04}.jpg"),
                st_imgs[current_img_index],
            )


def button1_click():
    global current_dir_index
    global current_img_index
    global al_imgs
    global al_tkimgs
    global st_imgs
    global st_tkimgs
    global dir_list
    global img_num
    global al_cor_num
    global st_cor_num
    global valid_num
    global totol_num
    global origin_imgs
    global new_imgDir_path
    origin_imgs = []
    al_imgs = []
    al_tkimgs = []
    st_imgs = []
    st_tkimgs = []
    valid_num = 0
    al_cor_num = 0
    st_cor_num = 0
    totol_num = 0
    current_img_index = 0

    imgDir_path = dir_list[current_dir_index]
    _resPath = os.path.join(imgDir_path, resPath)
    os.makedirs(_resPath)
    for type_ in type_list:
        if type_ == "wrong_imgs":
            os.makedirs(os.path.join(_resPath, type_))
            continue
        os.makedirs(os.path.join(_resPath, type_, st))
        os.makedirs(os.path.join(_resPath, type_, al))
    if root == "temp":
        new_imgDir_path = imgDir_path.replace("temp", "valid")
        os.makedirs(new_imgDir_path)
        os.makedirs(os.path.join(new_imgDir_path, "hwMot"))
    print(imgDir_path)
    for file in os.listdir(imgDir_path):
        if not file.endswith(".jpg"):
            continue
        img = ifFlip(cv2.imread(os.path.join(imgDir_path, file)), imgDir_path)
        origin_imgs.append(img.copy())

    _al_path = os.path.join(imgDir_path, al_path)
    for file in os.listdir(_al_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(_al_path, file))
        al_imgs.append(img.copy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        al_tkimgs.append(img)

    _st_path = os.path.join(imgDir_path, st_path)
    for file in os.listdir(_st_path):
        if not file.endswith(".jpg"):
            continue
        img = ifFlip(cv2.imread(os.path.join(_st_path, file)), imgDir_path)
        st_imgs.append(img.copy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        st_tkimgs.append(img)
    img_num = len(st_tkimgs)

    label1.configure(image=al_tkimgs[current_img_index])
    label2.configure(image=st_tkimgs[current_img_index])


def button2_click():
    if totol_num == img_num:
        return
    imgSave(type_list[0])
    nextImg()


def button3_click():
    if totol_num == img_num:
        return
    imgSave(type_list[1])
    global al_cor_num
    global st_cor_num
    global valid_num
    al_cor_num += 1
    st_cor_num += 1
    valid_num += 1
    nextImg()


def button4_click():
    if totol_num == img_num:
        return
    imgSave(type_list[2])
    global al_cor_num
    global st_cor_num
    global valid_num
    al_cor_num += 1
    valid_num += 1
    nextImg()


def button5_click():
    global al_cor_num
    global st_cor_num
    global valid_num
    global totol_num
    global current_dir_index
    if totol_num != img_num:
        return
    with open(
        os.path.join(dir_list[current_dir_index], resPath, "res.txt"), "w"
    ) as file:
        path = dir_list[current_dir_index]
        # path = os.path.dirname(path)
        path = os.path.basename(path)
        file.write(f"{path} {totol_num} {valid_num} {st_cor_num} {al_cor_num}\n")

    current_dir_index += 1


def button6_click():
    if totol_num == img_num:
        return
    imgSave(type_list[3])
    global al_cor_num
    global st_cor_num
    global valid_num
    st_cor_num += 1
    valid_num += 1
    nextImg()


def button7_click():
    if totol_num == img_num:
        return
    imgSave(type_list[4])
    global al_cor_num
    global st_cor_num
    global valid_num
    valid_num += 1
    nextImg()


# 创建UI
window = tk.Tk()
label1 = tk.Label(window)
label2 = tk.Label(window)
label1.pack(side=tk.LEFT)
label2.pack(side=tk.LEFT)
button1 = tk.Button(window, text="执行下一批次", command=button1_click)
button2 = tk.Button(window, text="无效", command=button2_click)
button3 = tk.Button(window, text="都对", command=button3_click)
button4 = tk.Button(window, text="算法对", command=button4_click)
button5 = tk.Button(window, text="结果保存", command=button5_click)
button6 = tk.Button(window, text="站点对", command=button6_click)
button7 = tk.Button(window, text="全错", command=button7_click)
button1.pack()
button2.pack()
button3.pack()
button4.pack()
button6.pack()
button7.pack()
button5.pack()

dir_list = []
for dir1 in os.listdir(root):
    for dir2 in os.listdir(os.path.join(root, dir1)):
        imgDir_path = os.path.join(root, dir1, dir2)
        if resPath in os.listdir(imgDir_path):
            continue
        num_img = 0
        for file in os.listdir(imgDir_path):
            if file.endswith("jpg"):
                num_img += 1
        if num_img == 0:
            continue
        dir_list.append(imgDir_path)
print(dir_list)


al_imgs = []
al_tkimgs = []
st_imgs = []
st_tkimgs = []
origin_imgs = []
current_dir_index = 0
current_img_index = 0

valid_num = 0
al_cor_num = 0
st_cor_num = 0
totol_num = 0
img_num = 0

new_imgDir_path = ""
# 运行主循环
window.mainloop()
