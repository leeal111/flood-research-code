from os.path import join


class key_value:
    def __init__(self) -> None:
        self.root = "data"
        self.stivMethod = 
        self.validResDir = 
        self.validScoDir = 
        self.sumlistPicDir = join("result_sotabase", "07_sum")
        self.stivResPicDir = 
        self.validRealFileName = 
        self.sumlistDir = 
        self.svmResDir = 
        self.correctResDir = "correct_result"
        self.sitePicDir = "hwMot"
        self.al_result_resName = "al_result.npy"
        self.st_result_resName = "st_result.npy"
        self.ananlyzeResDir = 
        self.trainDataResDir = "train_data"
        self.nnResPath = join(self.ananlyzeResDir, "nn_infer_result", "result.npy")


kvs = key_value()
