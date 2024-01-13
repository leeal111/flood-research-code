from os import listdir, makedirs
from os.path import join, exists
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV

from key_value import kvs, stringify_dict


def svm_process(sumlist):
    sumlist = sumlist / np.max(sumlist)
    return sumlist


def main():
    # SVM训练要点：
    # 数据预处理：特征缩放、特征选择、数据平衡
    # 核函数选择：线性核、多项式核和高斯核（径向基函数核）
    # 超参数调优：正则化参数C、核函数参数，交叉验证或网格搜索
    # 特征工程：
    root = kvs.root
    st_path = kvs.sumlistDir
    valid_path = kvs.validResDir
    svm_dic = kvs.svmDic
    res_path = kvs.svmResDir
    modelName = kvs.svmModelName
    X = []
    Y = []
    for dir1 in listdir(root):
        for dir2 in listdir(join(root, dir1)):
            imgDir_path = join(root, dir1, dir2)

            # 需要首先完成人工标注结果
            if not exists(join(imgDir_path, valid_path, "result.npy")):
                print(f"{imgDir_path} not exists valid_result")
                continue

            for file in listdir(join(imgDir_path, st_path)):
                if not file.endswith("npy"):
                    continue
                sumlist = np.load(join(imgDir_path, st_path, file))
                sumlist = svm_process(sumlist)
                X.append(sumlist)
            for item in np.load(join(imgDir_path, valid_path, "result.npy")):
                if item == 0:
                    Y.append(-1)
                elif item == 1:
                    Y.append(1)
                else:
                    print("Unknown classindex!")

    pos_num = np.sum(np.array(Y) > 0)
    neg_num = np.sum(np.array(Y) < 0)
    print(f"{pos_num} {neg_num}")
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), np.array(Y), test_size=0.2, random_state=42
    )
    print("训练集大小：", X_train.shape)
    print("测试集大小：", X_test.shape)

    # 训练
    # classifier = svm.SVC(**svm_dic)
    # classifier.fit(X_train, y_train)

    # # 计算准确率
    # prediction = classifier.predict(X_test)
    # accuracy = accuracy_score(y_test, prediction)
    # print("准确率：", accuracy)
    svr = svm.SVC()
    parameters = {
        "C": [0.001, 0.003, 0.006, 0.009, 0.01, 0.04, 0.08, 0.1, 0.2, 0.5, 1.0],
        "kernel": (
            "linear",
            "rbf",
        ),
        "gamma": [0.001, 0.005, 0.1, 0.15, 0.20, 0.23, 0.27],
        "decision_function_shape": ["ovo", "ovr"],
        # "class_weight": [{-1: pos_num, 1: neg_num}],
    }
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    best_params = clf.best_params_
    modelName = "search"
    results = clf.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        if mean_score < 0.8:
            continue
        print(f"Mean score: {mean_score}, Parameters: {params}")

    if modelName != "":
        makedirs(res_path, exist_ok=True)
        joblib.dump(best_model, join(res_path, modelName + ".joblib"))


if __name__ == "__main__":
    main()
