from os import makedirs
from os.path import join
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV
import json
from nn_train import train_data_dir


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


mode = 1
svm_dic = {"kernel": "linear"}
svm_model_dir = join("result", "valid_model")
model_name = stringify_dict(svm_dic) if mode == 0 else "GridSearch"


def main():
    global model_name
    # SVM训练要点：
    # 数据预处理：特征缩放、特征选择、数据平衡
    # 核函数选择：线性核、多项式核和高斯核（径向基函数核）
    # 超参数调优：正则化参数C、核函数参数，交叉验证或网格搜索
    # 特征工程：
    res_path = svm_model_dir

    X = np.load(join(train_data_dir, "sumlists.npy"))
    Y = np.load(join(train_data_dir, "labels.npy"))
    X = [svm_process(x) for x in X]
    pos_num = np.sum(np.array(Y) > 0)
    neg_num = np.sum(np.array(Y) < 0)
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
        parameters = {
            "C": [0.001, 0.003, 0.006, 0.009, 0.01, 0.04, 0.08, 0.1, 0.2, 0.5, 1.0],
            "kernel": ("rbf",),
            "gamma": [0.001, 0.005, 0.1, 0.15, 0.20, 0.23, 0.27],
            "decision_function_shape": ["ovo", "ovr"],
            # "class_weight": [{-1: pos_num, 1: neg_num}],
        }
        clf = GridSearchCV(svr, parameters)
        clf.fit(np.array(X), np.array(Y))

        # 保存
        best_model = clf.best_estimator_
        best_params = clf.best_params_
        results = clf.cv_results_
        for mean_score, params in zip(results["mean_test_score"], results["params"]):
            if mean_score < 0.8:
                continue
            print(f"Mean score: {mean_score}, Parameters: {params}")
        makedirs(res_path, exist_ok=True)
        joblib.dump(best_model, join(res_path, model_name + ".joblib"))


if __name__ == "__main__":
    main()
