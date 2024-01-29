from utils import call_for_imgss, get_imgs_paths
from valid_compute_imp import *
from stiv_compute_routine_imp import root

path_list = get_imgs_paths(root)
call_for_imgss(path_list, valid_score_call)

valid_thos = [
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
]

call_for_imgss(path_list, valid_result_call, valid_thos=valid_thos)
