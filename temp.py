import os
import shutil
from os.path import join
from os import makedirs, listdir
import cv2

root = "temp"
type_list = [
    "invalid",
    "total_right",
    "al_right",
    "st_right",
    "total_wrong",
    "wrong_imgs",
]
st = "st"
al = "al"


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


for dir in listdir(root):
    for dir2 in listdir(join(root, dir)):
        img_Dir = join(root, dir, dir2)
        shutil.rmtree(join(img_Dir, "ananlyze", "wrong_imgs"), ignore_errors=True)
        imgs = []
        for file in os.listdir(img_Dir):
            if not file.endswith("jpg"):
                continue
            imgs.append(ifFlip(cv2.imread(os.path.join(img_Dir, file)), img_Dir))
        _resPath = os.path.join(img_Dir, "ananlyze")
        os.makedirs(_resPath, exist_ok=True)
        for type_ in type_list:
            if type_ == "wrong_imgs":
                os.makedirs(os.path.join(_resPath, type_))
                continue
            os.makedirs(os.path.join(_resPath, type_, st), exist_ok=True)
            os.makedirs(os.path.join(_resPath, type_, al), exist_ok=True)
        for file in listdir(join(_resPath, type_list[3], st)):
            idx = int(os.path.splitext(file)[0])
            cv2.imwrite(
                os.path.join(
                    _resPath,
                    type_list[5],
                    f"{idx:04}.jpg",
                ),
                imgs[idx],
            )
        for file in listdir(join(_resPath, type_list[4], st)):
            idx = int(os.path.splitext(file)[0])
            cv2.imwrite(
                os.path.join(
                    _resPath,
                    type_list[5],
                    f"{idx:04}.jpg",
                ),
                imgs[idx],
            )
