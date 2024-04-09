from os.path import normpath
from stiv_routine_imp import stiv_call
from utils import call_for_imgss, get_imgs_paths, imgss_del_call
from valid_routine_imp import valid_call
from ananlyze_routine_imp import (
    ananlyze_valid_wrong_call,
    ananlyze_correct_wrong_call,
    ananlyze_compare_img_call,
)

# 配置参数
test_mode = 2  # 0:img 1:imgs 2:imgss
# call_func = valid_call[test_mode]
call_func = ananlyze_compare_img_call
img_path = normpath(r"test\stiv\sti007.jpg")
imgs_path = normpath(r"test\stiv_routine\20240115_131849")
imgss_path = normpath(r"data\data_base")

##############################################
if test_mode == 0:
    call_func(img_path, if_R2L=False)

elif test_mode == 1:
    call_func(imgs_path, if_R2L=False)

elif test_mode == 2:
    call_for_imgss(get_imgs_paths(imgss_path), call_func, del_dir=None)
