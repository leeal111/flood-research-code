import cv2
from stiv_compute_routine_imp import imgs_if_R2L


correct_result_dir = "correct_result"
correct_al_result_file = "al_result.npy"
correct_st_result_file = "st_result.npy"


def ifFlip(img, path):
    if imgs_if_R2L(path):
        return cv2.flip(img, 1)
    else:
        return img
