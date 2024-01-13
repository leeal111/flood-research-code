import json
from os.path import join


def stringify_dict(dictionary):
    json_str = json.dumps(dictionary)
    json_str = json_str.strip("{}").replace('"', "")
    json_str = json_str.replace(": ", "-")
    json_str = json_str.replace(", ", "_")
    return json_str


class key_value:
    def __init__(self) -> None:
        self.root = "data"
        self.stivMethod = "sotabase"
        self.validResDir = "valid_result"
        self.validScoDir = "valid_score"
        self.sumlistPicDir = join("result_sotabase", "07_sum")
        self.stivResPicDir = join("result_sotabase", "11_STIRES")
        self.validRealFileName = "result.npy"
        self.sumlistDir = join("result_sotabase", "10_sumlist")
        self.svmResDir = join("result", "valid_model")
        self.svmDic = {"kernel": "linear"}
        self.svmModelName = stringify_dict(self.svmDic)  # "search"
        self.correctResDir = "correct_result"
        self.sitePicDir = "hwMot"
        self.al_result_resName = "al_result.npy"
        self.st_result_resName = "st_result.npy"
        self.ananlyzeResDir = "result"


kvs = key_value()
