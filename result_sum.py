import pandas as pd
import os
import cv2
import shutil

savePath = "corr_res"
root = "temp"
resPath = "ananlyze"

os.makedirs(savePath, exist_ok=True)
for loc in os.listdir(root):
    total_num = 0
    st_cor_num = 0
    al_cor_num = 0
    with open(os.path.join(savePath, loc + ".txt"), "w") as file:
        for dir in os.listdir(os.path.join(root, loc)):
            data_path = os.path.join(root, loc, dir, resPath, "res.txt")
            if not os.path.exists(data_path):
                continue
            with open(data_path, "r") as resfile:
                line = resfile.readline()
                total_num += int(line.split()[2])
                st_cor_num += int(line.split()[3])
                al_cor_num += int(line.split()[4])
                file.write(line)
    if total_num == 0:
        print(f"{loc}无有效图片 ")
    else:
        print(
            f"{loc}有效图片共{total_num}张，其中站点和算法准确率分别为 {(st_cor_num/total_num*100):.1f}% 和 {(al_cor_num/total_num*100):.1f}% "
        )


for loc in os.listdir(root):
    wrong_imgs = []
    for dir in os.listdir(os.path.join(root, loc)):
        imgWrong_path = os.path.join(root, loc, dir, resPath, "wrong_imgs")
        if not os.path.exists(imgWrong_path):
            continue
        for file in os.listdir(imgWrong_path):
            if not file.endswith("jpg"):
                continue
            wrong_imgs.append(cv2.imread(os.path.join(imgWrong_path, file)))
    save_loc_path = os.path.join(savePath, "wrong", loc)

    shutil.rmtree(save_loc_path, ignore_errors=True)
    os.makedirs(save_loc_path)
    for idx, img in enumerate(wrong_imgs):
        cv2.imwrite(
            os.path.join(
                save_loc_path,
                f"{idx:04}.jpg",
            ),
            wrong_imgs[idx],
        )


def dataColl(type_name):
    for loc in os.listdir(root):
        save_loc_path = os.path.join(savePath, type_name, loc)
        shutil.rmtree(save_loc_path, ignore_errors=True)
        st_path = os.path.join(save_loc_path, "st")
        st_imgs = []
        al_path = os.path.join(save_loc_path, "al")
        al_imgs = []
        os.makedirs(st_path)
        os.makedirs(al_path)
        for dir in os.listdir(os.path.join(root, loc)):
            img_path = os.path.join(root, loc, dir, resPath, type_name)
            if not os.path.exists(img_path):
                continue
            imgWrong_path = os.path.join(img_path, "al")
            for file in os.listdir(imgWrong_path):
                if not file.endswith("jpg"):
                    continue
                al_imgs.append(cv2.imread(os.path.join(imgWrong_path, file)))

            imgWrong_path = os.path.join(img_path, "st")
            for file in os.listdir(imgWrong_path):
                if not file.endswith("jpg"):
                    continue
                st_imgs.append(cv2.imread(os.path.join(imgWrong_path, file)))

        for idx, img in enumerate(st_imgs):
            cv2.imwrite(
                os.path.join(
                    st_path,
                    f"{idx:04}.jpg",
                ),
                st_imgs[idx],
            )
        for idx, img in enumerate(al_imgs):
            cv2.imwrite(
                os.path.join(
                    al_path,
                    f"{idx:04}.jpg",
                ),
                al_imgs[idx],
            )
