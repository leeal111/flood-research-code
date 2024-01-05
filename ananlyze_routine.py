from ananlyze_routine_imp import *
from key_value import kvs

res_Path = kvs.ananlyzeResDir
root = kvs.root
TestMode = 0

makedirs(res_Path, exist_ok=True)
if TestMode == 0:
    valid_method_result_eval(res_Path, root)
else:
    print("Unknown method")
