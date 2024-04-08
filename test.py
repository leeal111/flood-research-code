import shutil
from utils import *
from os.path import normpath, join
from os import listdir

root = normpath(r"data\hd")
for dir in listdir(root):
    dir_path = join(root, dir)
    shutil.copytree(
        join(dir_path, "result_sotabase", "0_10_STIRES"), join(dir_path, "Result")
    )
    shutil.rmtree(join(dir_path, "correct_result"))
    shutil.rmtree(join(dir_path, "result_sotabase"))
    shutil.rmtree(join(dir_path, "valid_result"))
    shutil.rmtree(join(dir_path, "valid_score"))
