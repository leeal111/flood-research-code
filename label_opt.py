from os.path import normpath, join
import numpy as np


def reLabel(mode, imgDir_path, current_img_index):
    valid_p = join(imgDir_path, kvs.validResDir, kvs.validRealFileName)
    corre_s_p = join(imgDir_path, kvs.correctResDir, kvs.st_result_resName)
    corre_a_p = join(imgDir_path, kvs.correctResDir, kvs.al_result_resName)
    valid = np.load(valid_p)
    corre_s = np.load(corre_s_p)
    corre_a = np.load(corre_a_p)

    if mode == -1:
        valid[current_img_index] = 0
        corre_a[current_img_index] = -1
        corre_s[current_img_index] = -1
    else:
        valid[current_img_index] = 1
        if mode == 0:
            corre_a[current_img_index] = 0
            corre_s[current_img_index] = 0
        elif mode == 1:
            corre_a[current_img_index] = 1
            corre_s[current_img_index] = 0
        elif mode == 2:
            corre_a[current_img_index] = 0
            corre_s[current_img_index] = 1
        elif mode == 3:
            corre_a[current_img_index] = 1
            corre_s[current_img_index] = 1

    np.save(valid_p, valid)
    np.save(corre_s_p, corre_s)
    np.save(corre_a_p, corre_a)


if __name__ == "__main__":
    mode = 0  # -1：无效，0：全错, 1：算法对, 2：站点对, 3：都对
    imgDir_path = normpath(r"C:\BaseDir\code\flood\data\fj\20231213_103431")
    current_img_index = 0
    reLabel(mode, imgDir_path, current_img_index)
