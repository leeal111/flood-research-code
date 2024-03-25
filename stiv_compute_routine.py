from stiv_compute_routine_imp import *
from os.path import normpath
from utils import call_for_imgss, get_imgs_paths

# 配置参数
test_mode = 4  # 1:img 2:imgs 3:imgs&speed 4:imgss 5：imgssdelete
img_path = normpath(r"test\sti010.jpg")
imgs_path = normpath(r"C:\BaseDir\code\flood\result\correct_wrong\al_ans-1_st_ans-0")


##############################################
if test_mode == 1:
    img_test(img_path, if_R2L=False)

elif test_mode == 2:
    imgs_test(imgs_path, if_R2L=False)

elif test_mode == 3:
    imgs_test_with_speed(imgs_path, if_R2L=False)

elif test_mode == 4:
    paths = get_imgs_paths(root)
    call_for_imgss(paths, stiv_row_call)
    call_for_imgss(paths, stiv_compute_call)

elif test_mode == 5:
    paths = get_imgs_paths(root)
    call_for_imgss(paths, stiv_del_call, del_path=join("valid_score"))
else:
    print("Unknown method")
    exit()
