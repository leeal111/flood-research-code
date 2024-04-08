from os import makedirs
from os.path import join
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV
import json
import time
from display import line_chart_img, lines_chart_img

from utils import compute_mean_precision, compute_precision

train_data_dir = "result\\data"
ananlyze_result_dir = "result"


# SVM的数据归一化
def svm_process(sumlist):
    sumlist = sumlist / np.max(sumlist)
    return sumlist


# 参数形成字符串
def stringify_dict(dictionary):
    json_str = json.dumps(dictionary)
    json_str = json_str.strip("{}").replace('"', "")
    json_str = json_str.replace(": ", "-")
    json_str = json_str.replace(", ", "_")
    return json_str


mode = 1  # 0:参数 1:搜索
svm_dic = {"kernel": "linear"}  # svm参数
svm_model_dir = join("result", "valid_model")
model_name = stringify_dict(svm_dic) if mode == 0 else "GridSearch"


def main():
    global model_name
    res_path = svm_model_dir

    X = np.load(join(train_data_dir, "sumlists.npy"))
    Y = np.load(join(train_data_dir, "labels.npy"))
    X = [svm_process(x) for x in X]
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), np.array(Y), test_size=0.2, random_state=0
    )

    if mode == 0:
        # 训练
        classifier = svm.SVC(**svm_dic)
        classifier.fit(X_train, y_train)

        # 测试
        prediction = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        print("准确率：", accuracy)

        # 保存
        makedirs(res_path, exist_ok=True)
        joblib.dump(classifier, join(res_path, model_name + ".joblib"))
    else:
        # 网格搜索
        svr = svm.SVC()
        searchStr="coef0"
        param_grid = {
    'C': [0.01,0.1,1], #惩罚参数, 0.01, 0.1, 1, 10, 100, 1000
    'kernel': ['poly'], #核函数, '', 'sigmoid'
    'degree': [2,3,4], #多项式核函数的次数
    # 'gamma': ['scale'], #高斯核函数的带宽, 'auto', 0.001, 0.01, 0.1, 1
    'coef0': [-0.5,0.0, 0.5], #多项式核函数和sigmoid核函数的常数项
    'shrinking': [True],
    'probability': [True]
}
        clf = GridSearchCV(svr, param_grid,n_jobs=-1)
        stime = time.time()
        clf.fit(np.array(X_train), np.array(y_train))
        etime = time.time()
        # 保存
        best_model = clf.best_estimator_
        best_para=clf.best_params_
        results = clf.cv_results_
        
        makedirs(join(ananlyze_result_dir, "svm_train"), exist_ok=True)
        with open(join(ananlyze_result_dir, "svm_train", "result.txt"), "w") as f:
            for mean_score, params in zip(results["mean_test_score"], results["params"]):
                f.write(f"Mean score: {mean_score}, Parameters: {params}\n")
            f.write(f"total time: {etime-stime}, total nums: {len(results["mean_test_score"])}\n")
            f.write(f"best paras: {best_para}\n")
        # cv2.imwrite(join(ananlyze_result_dir, "svm_train", "result.jpg"),line_chart_img(results["mean_test_score"],param_grid[searchStr])) 
        
        makedirs(res_path, exist_ok=True)
        joblib.dump(best_model, join(res_path, model_name + ".joblib"))


if __name__ == "__main__":
    main()
