import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def toImg(data):
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


def listImg(list):
    # USage：折线图图像数据

    x = range(len(list))

    plt.plot(range(len(list)), list)
    max_index = np.argmax(list)
    max_value = list[max_index]
    plt.annotate(
        f"max value: ({x[max_index]:.2f}, {max_value:.2f})",
        xy=(x[max_index], max_value),
        xytext=(x[max_index] + 1 - 10, max_value),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )
    fig = plt.gcf()
    fig.canvas.draw()
    rgba_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.clf()
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGB2BGR)


def imgMtricImg(imgs):
    # USage：显示一个图像矩阵或者返回拼接好的图像矩阵

    return toImg(np.vstack([np.hstack(x) for x in imgs]), True)


def img_add_angle(img, angle, noPrintNum=0):
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


def ROC_plot(name, titles, fpr, tpr, auc_score):
    titles = {
        "v1_list_score": "平稳程度",
        "v2_list_score": "v2_list_score",
        "signal_peek_value_list_score": "signal_peek_value_list_score",
        "signal_noise_radio_list_score": "峰值信噪比",
        "nn_ifftimg_result": "神经网络",
    }
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % auc_score)
    plt.plot([0, 1], [0, 1], "k--")  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("假阳率", fontsize=20)
    plt.ylabel("真阳率", fontsize=20)
    plt.title(titles[name], fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.show()
