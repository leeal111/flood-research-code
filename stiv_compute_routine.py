import os
from key_value import kvs
from stiv import *
from stiv_compute_routine_imp import *

# 配置参数
ifSavePro = True
ifRight2Left = False
TestMode = 4  # 0:video 1:imgs 2:img 3:imgs&spedd 4:imgss
video_path = os.path.normpath()
imgDir_path = os.path.normpath(r"")
img_path = os.path.normpath(r"")

##############################################
stiv = STIV(ifSavePro, ifRight2Left, methodName=kvs.stivMethod)
if TestMode == 0:
    pass

elif TestMode == 1:
    ImgsTest(imgDir_path, stiv)

elif TestMode == 2:
    ImgTest(img_path, stiv)

elif TestMode == 3:
    ImgsTestWithSpeed(imgDir_path, stiv)

elif TestMode == 4:
    root = kvs.root
    for dir1 in os.listdir(root):
        for dir2 in os.listdir(os.path.join(root, dir1)):
            imgDir_path = os.path.join(root, dir1, dir2)
            rowData2MyData(imgDir_path)

            # 如果已经测试过了那么就不测试了
            # if os.path.basename(resPath(imgDir_path, stiv)) in os.listdir(imgDir_path):
            #     continue

            stiv.ifRightToLeft = ifRight2LeftForLoc(imgDir_path)

            if os.path.exists(
                os.path.join(imgDir_path, "hwMot", "flow_speed_evaluation_result.csv")
            ):
                ImgsTestWithSpeed(imgDir_path, stiv)
            else:
                ImgsTest(imgDir_path, stiv)

elif TestMode == 5:
    root = kvs.root
    for dir1 in os.listdir(root):
        for dir2 in os.listdir(os.path.join(root, dir1)):
            imgDir_path = os.path.join(root, dir1, dir2)
            del_path = os.path.join(imgDir_path, "valid_result", "nothing.npy")
            # shutil.rmtree(
            #     del_path,
            #     ignore_errors=True,
            # )
            if os.path.exists(del_path):
                os.remove(del_path)

else:
    print("Unknown method")
    exit()
