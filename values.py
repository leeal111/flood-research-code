from os.path import join

# stiv_routine_imp
stiv_method_name = "sotabase"
stiv_result_dir = "result" + ("_" if stiv_method_name != "" else "") + stiv_method_name
xxx_img_dir = "xxxMot"
xxx_others_dir = "cop"
xxx_csv_name = "flow_speed_evaluation_result.csv"
xxx_mot_prefix = "STI_MOT"
xxx_img_prefix = "sti"
stiv_csv_name = "stiv_result.xlsx"
stiv_real_name = "REALRES"

ifft_res_dir = join(stiv_result_dir, "0_10_IFFTRES")
sti_res_dir = join(stiv_result_dir, "0_11_STIRES")
img_dir = join(stiv_result_dir, "0_00_ORIGIN")
sum_data_dir = join(stiv_result_dir, "1_00_sumlist")
ifft_img_dir = join(stiv_result_dir, "0_06_ifft")

# valid_routine_imp
valid_threshold = 0.3
valid_score_dir = "valid_score"
valid_result_dir = "valid_result"
valid_label_file = "result.npy"
valid_example_dir = "valid_example"

# ananlyze_routine_imp
ananlyze_result_dir = "result"

# label_correct
correct_result_dir = "correct_result"
correct_al_result_file = "al_result.npy"
correct_st_result_file = "st_result.npy"
correct_example_dir = "correct_example"
correct_al_result_file_u = "al_result_u.npy"
correct_st_result_file_u = "st_result_u.npy"
