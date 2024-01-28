import os
from os.path import join
from stiv import *
from stiv_compute_routine_imp import *

root = "data"
stiv_method_name = "sotabase"
stiv_result_dir = os.path.basename(resPath("_", stiv_method_name))

ifft_res_dir = join(stiv_result_dir, "09_IFFTRES")
sti_res_dir = join(stiv_result_dir, "11_STIRES")
img_dir = join(stiv_result_dir, "00_ORIGIN")
sum_data_dir = join("result_sotabase", "10_sumlist")
ifft_img_dir = join("result_sotabase", "06_ifft")


def main():
    # 配置参数
    ifSavePro = True  # 是否在测试数据路径下保存中间结果
    ifRight2Left = False  # 河流流向是否是从右往左
    TestMode = 1  # 0:video 1:imgs 2:img 3:imgs&speed 4:imgss 5：imgssdelete
    video_path = os.path.normpath(r"")
    imgDir_path = os.path.normpath(
        r"C:\BaseDir\code\flood\result\valid_wrong\fp\nn_ifftimg_result"
    )
    img_path = os.path.normpath(r"")

    ##############################################
    stiv = STIV(ifSavePro, ifRight2Left, methodName=stiv_method_name)
    if TestMode == 0:
        pass

    elif TestMode == 1:
        ImgsTest(imgDir_path, stiv)

    elif TestMode == 2:
        ImgTest(img_path, stiv)

    elif TestMode == 3:
        if os.path.exists(
            os.path.join(imgDir_path, "hwMot", "flow_speed_evaluation_result.csv")
        ):
            ImgsTestWithSpeed(imgDir_path, stiv)
        else:
            ImgsTest(imgDir_path, stiv)

    elif TestMode == 4:
        for dir1 in os.listdir(root):
            for dir2 in os.listdir(os.path.join(root, dir1)):
                imgDir_path = os.path.join(root, dir1, dir2)
                rowData2MyData(imgDir_path)

                # 如果已经测试过了那么就不测试了
                if os.path.basename(
                    resPath(imgDir_path, stiv.methodName)
                ) in os.listdir(imgDir_path):
                    continue

                stiv.ifRightToLeft = ifRight2LeftForLoc(imgDir_path)

                if os.path.exists(
                    os.path.join(
                        imgDir_path, "hwMot", "flow_speed_evaluation_result.csv"
                    )
                ):
                    ImgsTestWithSpeed(imgDir_path, stiv)
                else:
                    ImgsTest(imgDir_path, stiv)

    elif TestMode == 5:
        for dir1 in os.listdir(root):
            for dir2 in os.listdir(os.path.join(root, dir1)):
                imgDir_path = os.path.join(root, dir1, dir2)
                del_path = os.path.join(imgDir_path, "valid_result", "result_.npy")
                # shutil.rmtree(
                #     del_path,
                #     ignore_errors=True,
                # )
                if os.path.exists(del_path):
                    os.remove(del_path)

    else:
        print("Unknown method")
        exit()


if __name__ == "__main__":
    main()
