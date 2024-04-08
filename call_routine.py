from os.path import normpath
from stiv_routine_imp import stiv_call
from utils import call_for_imgss, get_imgs_paths, imgss_del_call


# 配置参数
test_mode = 2  # 0:img 1:imgs 2:imgss
call_func = stiv_call[test_mode]  # imgss_del_call
img_path = normpath(r"test\stiv\sti007.jpg")
imgs_path = normpath(r"test\stiv_routine\20240327_133648")
imgss_path = normpath(r"test\stiv_routine\root")

##############################################
if test_mode == 0:
    call_func(img_path, if_R2L=False)

elif test_mode == 1:
    call_func(imgs_path, if_R2L=False)

elif test_mode == 2:
    call_for_imgss(get_imgs_paths(imgss_path), call_func, del_dir=None)
