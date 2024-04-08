import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "SimHei",
]  # 设置中文字体族

fig_size = (8, 6)
font_size = 20


def normalize_img(data):

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


def plt_fig_img(fig):
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return bgr_image


def line_chart_img(y, x=None):
    if x is None:
        x = range(len(y))
    fig, ax = plt.subplots(figsize=fig_size)
    plot_line(y, x, ax, name="1")
    return plt_fig_img(fig)


def plot_line(y, x, ax, name):
    ax.plot(x, y, label=name)
    max_index = np.argmax(y)
    max_value = y[max_index]
    y_range = ax.get_ylim()
    plt.annotate(
        f"(index:{max_index} value:{max_value:.1e})",
        xy=(x[max_index], max_value),
        xytext=(x[max_index], max_value + (y_range[1] - y_range[0]) / 50),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )


def tile_matrix_img(imgs):
    return np.vstack([np.hstack(x) for x in imgs])


def add_angle_img(img, angle, central_not_print_num=0):
    maxr = min(img.shape) // 2
    h = img.shape[0] // 2
    w = img.shape[1] // 2
    img_c = np.copy(img)
    for r in range(maxr):
        if r < central_not_print_num:
            continue
        img_c[
            h + int(r * math.sin(angle / 360 * 2 * math.pi)),
            w + int(r * math.cos(angle / 360 * 2 * math.pi)),
        ] = 255
        img_c[
            h - int(r * math.sin(angle / 360 * 2 * math.pi)),
            w - int(r * math.cos(angle / 360 * 2 * math.pi)),
        ] = 255

    return img_c


titles = {
    "smoothness": "平稳程度",
    "signal_peek_value": "总体能量值",
    "signal_noise_radio": "峰值信噪比",
    "nn": "神经网络",
    "svm": "支持向量机",
}


def pr_img(name, precision, recall, f1_score):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(recall, precision, label=f"PR curve (AP = {f1_score:0.2f})")
    ax.plot([1, 0], [0, 1], "k--")
    ax.xlim([0.0, 1.0])
    ax.ylim([0.0, 1.0])
    ax.xlabel("Recall", fontsize=font_size)
    ax.ylabel("Precision", fontsize=font_size)
    ax.title(titles[name], fontsize=font_size)
    ax.legend(loc="lower right", fontsize=font_size)
    return plt_fig_img(fig)


def lines_chart_img(ys, names, x=None):
    if x is None:
        x = range(len(ys[0]))
    fig, ax = plt.subplots(figsize=fig_size)
    for i, y in enumerate(ys):
        plot_line(y, x, ax, names[i])
    ax.legend()
    return plt_fig_img(fig)


if __name__ == "__main__":
    y = [5, 1, 4, 2, 3]
    y2 = [1, 2, 3, 4, 5]
    img = lines_chart_img([y, y2], ["1", "2"])
    imgs = [[img, img], [img, img]]

    cv2.imshow("test", img)
    cv2.waitKey(0)
