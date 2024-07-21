import math
import os
import cv2
import numpy as np

from display import add_angle_img, line_chart_img, normalize_img


class MethodFlag:
    std = True
    clr = True
    fft = True
    fftclr = True
    fftpow = True
    fftcrop = True
    ifft = True
    ifftclr = True

    def __init__(self, std, clr, fft, fftclr, fftpow, fftcrop, ifft, ifftclr):
        self.std = std
        self.clr = clr
        self.fft = fft
        self.fftclr = fftclr
        self.fftpow = fftpow
        self.fftcrop = fftcrop
        self.ifft = ifft
        self.ifftclr = ifftclr


def std_filter(sti):
    return (sti - np.mean(sti, axis=0)) / (np.std(sti, axis=0) + 1e-8)


def sobel(image):
    img_clr = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
    img_clr[img_clr < 0] = 0
    return img_clr


def abs_FFT_shift(image):
    return np.abs(np.fft.fftshift(np.fft.fft2(image)))


def low_freq_filter(image):
    image[:, image.shape[1] // 2] = 0
    image[image.shape[0] // 2] = 0
    return image


def vertical_delete(image):
    if image.shape[1] % 2 == 0:
        image_l = image[:, 1 : image.shape[1] // 2]
    else:
        image_l = image[:, 0 : image.shape[1] // 2]
    image_v = image[:, image.shape[1] // 2]
    image_r = image[:, image.shape[1] // 2 + 1 : image.shape[1]]
    image_r_flip = cv2.flip(image_r, 1)
    min_value = np.minimum(image_l, image_r_flip)
    image_l_res = image_l - min_value
    image_r_res = cv2.flip(image_r_flip - min_value, 1)

    image_v = np.zeros_like(image_v)
    return np.hstack([image_l_res, image_v[:, np.newaxis], image_r_res])


def img_pow(image, powNum=None):
    maxv = np.max(image)
    powNum = np.log(255) / np.log(maxv) if powNum == None else powNum
    return pow(image, powNum)


def img_crop(image, min_distance=40, min_angle=5, pixel_num=1000):
    # 获取图像中心点坐标
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    # 初始化集合，包含中心点坐标
    selected_coord = {(center_y, center_x)}
    image_c = image.copy()
    top_indices = np.argsort(image_c.ravel())[-pixel_num:]
    top_coords = np.unravel_index(top_indices, image_c.shape)
    top_coords = zip(top_coords[0][::-1], top_coords[1][::-1])

    not_coord = set()
    for top_coord in top_coords:
        local_min_dis = 10**10
        for s_coord in selected_coord:
            cur_min_dis = abs(top_coord[0] - s_coord[0]) + abs(
                top_coord[1] - s_coord[1]
            )
            if local_min_dis > cur_min_dis:
                local_min_dis = cur_min_dis

        # 角度判断
        coords = [c for c in selected_coord if c[0] > center_y]
        total = sum([image[y, x] for y, x in coords])
        mean_x, mean_y = sum(x * image[y, x] for y, x in coords) / (total + 1), sum(
            y * image[y, x] for y, x in coords
        ) / (total + 1)
        # image[int(mean_y), int(mean_x)] = np.max(image)
        mean_tan = math.atan2(mean_y - center_y, mean_x - center_x) / math.pi * 180
        local_tan = (
            math.atan2(top_coord[0] - center_y, top_coord[1] - center_x) / math.pi * 180
        )
        local_tan = local_tan + 180 if local_tan < 0 else local_tan
        local_min_angle = abs(local_tan - mean_tan)

        if local_min_dis < min_distance or local_min_angle < min_angle:
            selected_coord.add(top_coord)
        else:
            # print(local_min_angle)
            not_coord.add(top_coord)

    # 将未被选中的像素置为0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (y, x) in not_coord:
                image[y, x] = 0  # np.max(image)
    return image


def xycrd2polarcrd(
    img, res, theta, precision, rangeV=0, rangedivR=2, central_zero_num=0
):
    maxr = int(min(img.shape) / rangedivR)
    maxa = int(2 * theta / precision)
    h = img.shape[0] // 2
    w = img.shape[1] // 2

    dst = np.zeros([maxa, maxr])
    for a in range(maxa):
        angle = (res - theta) + (a * precision)
        for r in range(maxr):
            h0 = h + int(r * math.sin(angle / 180 * math.pi))
            w0 = w + int(r * math.cos(angle / 180 * math.pi))
            dst[a, r] += img[h0, w0]
            for i in range(rangeV):
                dangle = (i + 1) * precision
                dst[a, r] += (
                    img[
                        h + int(r * math.sin((angle - dangle) / 180 * math.pi)),
                        w + int(r * math.cos((angle - dangle) / 180 * math.pi)),
                    ]
                    + img[
                        h + int(r * math.sin((angle + dangle) / 180 * math.pi)),
                        w + int(r * math.cos((angle + dangle) / 180 * math.pi)),
                    ]
                )

    dst[:, :central_zero_num] = 0

    return dst


def search(img_feature, res, theta, precision):
    polar = xycrd2polarcrd(
        img_feature,
        res,
        theta,
        precision,
        rangeV=1,
        rangedivR=2.5,
        central_zero_num=20,
    )
    sum_list = np.sum(polar, axis=1)
    res = (res - theta) + (np.argmax(sum_list) * precision)
    return res, polar, sum_list


def process_blocks(image, block_size=(8, 8), func=None):
    # 获取图像的尺寸
    height, width = image.shape

    # 计算图像可以被分成多少个块
    num_rows = height // block_size[0]
    num_cols = width // block_size[1]

    # 创建一个用于存储处理后的块的矩阵
    processed_blocks = np.zeros(
        (num_rows, num_cols, block_size[0], block_size[1]), dtype=float
    )

    # 遍历每个块并应用指定的函数
    for i in range(num_rows):
        for j in range(num_cols):
            # 计算当前块的坐标
            start_row = i * block_size[0]
            end_row = start_row + block_size[0]
            start_col = j * block_size[1]
            end_col = start_col + block_size[1]

            # 获取当前块并应用函数
            block = image[start_row:end_row, start_col:end_col]
            if func is not None:
                processed_block = func(block)
            else:
                processed_block = block

            # 将处理后的块存储在输出矩阵中
            processed_blocks[i, j] = processed_block

    # 将处理后的块拼接回原始图像大小
    processed_image = np.zeros_like(image)
    for i in range(num_rows):
        for j in range(num_cols):
            # 计算当前块的坐标
            start_row = i * block_size[0]
            end_row = start_row + block_size[0]
            start_col = j * block_size[1]
            end_col = start_col + block_size[1]

            # 将处理后的块放回原始图像位置
            processed_image[start_row:end_row, start_col:end_col] = processed_blocks[
                i, j
            ]

    return processed_image


class STIV:
    def __init__(self, if_eval=True) -> None:
        self.eval = if_eval
        self.proImgs = {}
        self.proDatas = {}
        self.score = None

    def _score(self, sum_list, range_len=5):
        max_index = np.argmax(sum_list)
        total = np.max(sum_list)
        for i in range(range_len):
            i += 1
            index = max_index - i if max_index - i >= 0 else 0
            total += sum_list[index]
            index = (
                max_index + i if max_index + i < len(sum_list) else len(sum_list) - 1
            )
            total += sum_list[index]
        self.score = total / np.sum(sum_list)

    def _img_process(self, img):

        # # 消除不同位置的竖直亮度差异。
        # img_std = std_filter(img.copy())

        # # 提取倾斜特征，使得fft的特征更明显，同时也是归一化
        # # l = 8 * (2**0)
        # # print(l)
        # # img_clr = process_blocks(img_std.copy(), block_size=(l, l), func=sobel)
        # img_clr = sobel(img_std)

        # # 傅里叶变换
        # img_fft = abs_FFT_shift(img_clr.copy())
        # low_freq_filter(img_fft)

        # # 过滤由于partSobel产生的噪声
        # img_fft_clr = vertical_delete(img_fft)

        # # 幂运算
        # img_fft_pow = img_pow(img_fft_clr, 2)

        # # 仅取中心部分
        # img_fft_crop = img_crop(img_fft_pow)

        # # 傅里叶变换并取幅值
        # img_fe = abs_FFT_shift(img_fft_crop.copy())
        # low_freq_filter(img_fe)

        # # 更严格的取向判断
        # img_fe_clr = vertical_delete(img_fe)

        # if self.eval:
        #     self.proImgs["ORIGIN"] = img
        #     self.proImgs["std"] = normalize_img(img_std)
        #     self.proImgs["clr"] = normalize_img(img_clr)
        #     self.proImgs["fft"] = normalize_img(img_fft)
        #     self.proImgs["fftclr"] = normalize_img(img_fft_clr)
        #     self.proImgs["fftcrop"] = normalize_img(img_fft_crop)
        #     self.proImgs["ifft"] = normalize_img(img_fe)
        #     self.proImgs["ifftclr"] = normalize_img(img_fe_clr)

        flag = MethodFlag(
            **{
                "std": True,
                "clr": True,
                "fft": True,
                "fftclr": True,
                "fftpow": True,
                "fftcrop": False,
                "ifft": True,
                "ifftclr": False,
            }
        )

        if self.eval:
            self.proImgs["ORIGIN"] = img

        if flag.std:
            img = std_filter(img.copy())
            if self.eval:
                self.proImgs["std"] = normalize_img(img)
        if flag.clr:
            img = sobel(img)
            if self.eval:
                self.proImgs["clr"] = normalize_img(img)
        if flag.fft:
            img = abs_FFT_shift(img.copy())
            low_freq_filter(img)
            if self.eval:
                self.proImgs["fft"] = normalize_img(img)
        if flag.fftclr:
            img = vertical_delete(img)
            if self.eval:
                self.proImgs["fftclr"] = normalize_img(img)
        if flag.fftpow:
            img = img_pow(img, 2)
            if self.eval:
                self.proImgs["fftpow"] = normalize_img(img)
        if flag.fftcrop:
            img = img_crop(img)
            if self.eval:
                self.proImgs["fftcrop"] = normalize_img(img)
        if flag.ifft:
            img = abs_FFT_shift(img.copy())
            low_freq_filter(img)
            if self.eval:
                self.proImgs["ifft"] = normalize_img(img)
        if flag.ifftclr:
            img = vertical_delete(img)
            if self.eval:
                self.proImgs["ifftclr"] = normalize_img(img)
        return img

    def sti2angle(self, img, if_R2L=False):
        self.proImgs = {}
        self.proDatas = {}
        self.score = None

        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if if_R2L:
            img = cv2.flip(img, 1)

        fea = self._img_process(img)

        ## 开始搜寻结果 ##
        res, _, sum_list = search(fea, res=45, theta=45, precision=1)
        self._score(sum_list)

        if res < 2:
            res = 2
        if res > 88:
            res = 88

        res, _, _ = search(fea, res, theta=2, precision=0.1)

        # result = res if res <= 90 else 180 - res
        result = 90 - res if res <= 90 else res - 90

        if self.eval:
            self.proImgs["sum"] = line_chart_img(sum_list)
            self.proDatas["sumlist"] = sum_list.copy()
            self.proImgs["FFTRES"] = add_angle_img(self.proImgs["fft"], -result, 200)
            self.proImgs["IFFTRES"] = add_angle_img(
                self.proImgs["ifft"], 90 - result, 200
            )
            self.proImgs["STIRES"] = add_angle_img(self.proImgs["ORIGIN"], 90 - result)

        return result

    # def sti2angle_FFT(self, img, if_R2L=False):
    #     self.proImgs = {}
    #     self.proDatas = {}
    #     self.score = None

    #     if len(img.shape) > 2:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     if if_R2L:
    #         img = cv2.flip(img, 1)

    #     # img_std = std_filter(img.copy())

    #     # img_clr = part_sobel(img_std.copy(), 1000)

    #     img_fft = abs_FFT_shift(img.copy())
    #     img_fe = np.log(img_fft)
    #     low_freq_filter(img_fft)
    #     low_freq_filter(img_fe)

    #     # img_fft_clr = vertical_delete(img_fft)

    #     res, _, sum_list = self._search(img_fe, res=135, theta=45, precision=1)
    #     self.proImgs["sum"] = line_chart_img(sum_list)
    #     self.proDatas["sumlist"] = sum_list.copy()

    #     if res < 92:
    #         res = 92
    #     if res > 178:
    #         res = 178

    #     res, _, _ = self._search(img_fe, res=135, theta=2, precision=0.1)

    #     result = res if res <= 90 else 180 - res
    #     if self.eval:
    #         # self.proImgs["std"] =  normalize_img(img_std)
    #         # self.proImgs["clr"] =  normalize_img(img_clr)
    #         self.proImgs["fft"] = normalize_img(img_fft)
    #         # self.proImgs["fftclr"] =  normalize_img(img_fft_clr)
    #         self.proImgs["fea"] = normalize_img(img_fe)
    #         self.proImgs["sum"] = line_chart_img(sum_list)
    #         self.proImgs["FFTRES"] = add_angle_img(self.proImgs["fea"], -result, 200)

    #     return result


if __name__ == "__main__":
    stiv = STIV()
    print(stiv.sti2angle(cv2.imread(os.path.normpath(r"test\stiv\sti007.jpg")), True))
