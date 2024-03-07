import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def normalize_img(data):
    # USage：对于任意数据的np数组，归一化到0-255区域。已经在0-255则仅转换为np.uint8

    ma = np.max(data)
    mi = np.min(data)
    if ma != mi:
        data = (data - mi) * (255 / (ma - mi))
    else:
        if ma > 255:
            v = 255
        elif ma < 0:
            v = 0
        else:
            v = ma
        data.fill(v)
    img = data.astype(np.uint8)

    return img


def line_chart_img(list, x_label=None):
    # USage：折线图图像数据

    x = range(len(list))
    if x_label != None:
        plt.xticks(x, x_label)
    plt.plot(x, list)
    max_index = np.argmax(list)
    max_value = list[max_index]
    plt.annotate(
        f"           ({max_value:.2f})",
        xy=(x[max_index], max_value),
        xytext=(x[max_index], max_value),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )
    fig = plt.gcf()
    fig.canvas.draw()
    rgba_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.clf()
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGB2BGR)


def tile_matrix_img(imgs):
    # USage：显示一个图像矩阵或者返回拼接好的图像矩阵

    return normalize_img(np.vstack([np.hstack(x) for x in imgs]), True)


def add_angle_img(img, angle, noPrintNum=0):
    #  返回带角度线的图

    maxr = min(img.shape) // 2
    h = img.shape[0] // 2
    w = img.shape[1] // 2
    img_c = np.copy(img)
    for r in range(maxr):
        if r < noPrintNum:
            continue
        img_c[
            h + int(r * math.sin(angle / 360 * 2 * math.pi)),
            w + int(r * math.cos(angle / 360 * 2 * math.pi)),
        ] = 255
    for r in range(maxr):
        if r < noPrintNum:
            continue
        img_c[
            h - int(r * math.sin(angle / 360 * 2 * math.pi)),
            w - int(r * math.cos(angle / 360 * 2 * math.pi)),
        ] = 255

    return img_c


def roc_img(name, fpr, tpr, auc_score):
    titles = {
        "v1_list_score": "平稳程度",
        "signal_peek_value_list_score": "总体能量值",
        "signal_noise_radio_list_score": "峰值信噪比",
        "nn_ifftimg_result": "神经网络",
        "svm_list_result": "支持向量机",
    }
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.plot(fpr, tpr, label="ROC curve (mean precision = %0.2f)" % auc_score)
    plt.plot([0, 1], [0, 1], "k--")  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("假阳率", fontsize=20)
    plt.ylabel("真阳率", fontsize=20)
    plt.title(titles[name], fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    fig = plt.gcf()
    fig.canvas.draw()
    rgba_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.clf()
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGB2BGR)


def pr_img(name, precision, recall, f1_score):
    titles = {
        "v1_list_score": "平稳程度",
        "signal_peek_value_list_score": "总体能量值",
        "signal_noise_radio_list_score": "峰值信噪比",
        "nn_ifftimg_result": "神经网络",
        "svm_list_result": "支持向量机",
    }
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.plot(recall, precision, label="PR curve (AP = %0.2f)" % f1_score)
    plt.plot([1, 0], [0, 1], "k--")  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.title(titles[name], fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    fig = plt.gcf()
    fig.canvas.draw()
    rgba_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.clf()
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGB2BGR)


def lines_chart_img(lists, names, x_label=None):
    # USage：折线图图像数据

    x = range(len(lists[0]))
    if x_label != None:
        plt.xticks(x, x_label)
    for i, l in enumerate(lists):
        plt.plot(x, l, label=names[i])
        max_index = np.argmax(l)
        max_value = l[max_index]
        plt.annotate(
            f"           ({max_value:.2f})",
            xy=(x[max_index], max_value),
            xytext=(x[max_index], max_value),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
        )
    # plt.xlabel("Recall", fontsize=20)
    # plt.ylabel("Precision", fontsize=20)
    # plt.title(titles[name], fontsize=20)
    plt.legend()
    plt.show()
    fig = plt.gcf()
    fig.canvas.draw()
    rgba_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.clf()
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGB2BGR)
