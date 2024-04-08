from os.path import join

# stiv_routine_imp
stiv_method_name = "sotabase"
stiv_result_dir = "result" + ("_" if stiv_method_name != "" else "") + stiv_method_name
hw_img_dir = "hwMot"
hw_others_dir = "cop"
hw_csv_name = "flow_speed_evaluation_result.csv"
hw_mot_prefix = "STI_MOT"
hw_img_prefix = "sti"
stiv_csv_name = "stiv_result.xlsx"
stiv_real_name = "REALRES"

# valid_routine_imp
valid_threshold = 0.3


ifft_res_dir = join(stiv_result_dir, "0_09_IFFTRES")
sti_res_dir = join(stiv_result_dir, "0_10_STIRES")
img_dir = join(stiv_result_dir, "0_00_ORIGIN")
sum_data_dir = join(stiv_result_dir, "1_00_sumlist")
ifft_img_dir = join(stiv_result_dir, "0_06_ifft")
