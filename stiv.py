import cv2
import numpy as np
import math

from display import *


def std_filter(sti):
    # USage：
    # in ------
    #   sti: 原始的sti，元素为cv2格式图片
    # out -----
    #   sti: 经过标准滤波的sti
    return (sti - np.mean(sti, axis=0)) / (np.std(sti, axis=0) + 1e-8)


def xycrd2polarcrd(
    img, res=45, theta=45, precision=1, rangeV=0, rangedivR=2, zeroNum=0
):
    # USage：
    # in ------
    #   img: 直角坐标fft图
    # out -----
    #   dst: 极坐标fft图

    # (res-theta)+(np.argmax(sum_list)*precision)
    maxr = int(min(img.shape) / rangedivR)
    # maxr = 100 if maxr > 100 else maxr
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

    dst[:, :zeroNum] = 0

    return dst


def list2score(sum_list):
    # 计算峰值比例
    sum_list = sum_list / np.max(sum_list)
    maxIndex = np.argmax(sum_list)
    diff = np.diff(sum_list)
    lsum = np.sum(np.abs(diff[:maxIndex])) / (np.max(sum_list) - sum_list[0])
    rsum = np.sum(np.abs(diff[maxIndex : len(diff)])) / (
        np.max(sum_list) - sum_list[len(sum_list) - 1]
    )
    score = 2 / (lsum + rsum)

    return score


def lowFreqFilter(image):
    image[:, image.shape[1] // 2] = 0
    image[image.shape[0] // 2] = 0
    return image


def absFFTshift(image):
    return np.abs(np.fft.fftshift(np.fft.fft2(image)))


def verticalDelete(image):
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


def imgPow(image, powNum=None):
    img_clr = verticalDelete(image)
    maxv = np.max(img_clr)
    powNum = np.log(255) / np.log(maxv) if powNum == None else powNum
    return pow(image, powNum)


def partSobel(image, pixelNum):
    index_list = list(range(0, image.shape[0], pixelNum))
    index_list_end = [x + pixelNum for x in index_list]

    if index_list_end[len(index_list_end) - 1] > image.shape[0]:
        index_list_end[len(index_list_end) - 1] = image.shape[0]

    img = np.zeros_like(image)
    for startIndex, endIndex in zip(index_list, index_list_end):
        img_clr_part = cv2.Sobel(image[startIndex:endIndex], cv2.CV_64F, 1, 1, ksize=3)
        img_clr_part[img_clr_part < 0] = 0
        img[startIndex:endIndex] = img_clr_part
    return img


def imgCrop(image, rangedivR=10):
    h, w = image.shape[:2]
    l = min(h, w) // rangedivR

    # 计算图像中心坐标
    center_x = w // 2
    center_y = h // 2

    # 计算正方形的左上角和右下角坐标
    x1 = center_x - l // 2
    y1 = center_y - l // 2
    x2 = center_x + l // 2
    y2 = center_y + l // 2

    # 创建剪裁掩膜
    mask = np.zeros((h, w), dtype=np.uint8)
    # mask[y1:y2, x1:x2] = 255
    mask[center_y:y2, x1:center_x] = 255
    mask[y1:center_y, center_x:x2] = 255
    # 将图像与掩膜相乘，将正方形之外的像素置为 0
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


class STIV:
    def __init__(self) -> None:
        self.proImgs = {}
        self.proDatas = {}

    def sti2angle_IFFT(self, img):
        # 消除不同位置的竖直亮度差异。可以改进的地方为以突变边界消除（即50像素）
        # 作用：可以突出高亮度区域将低亮度区域的特征淹没了
        img_std = std_filter(img.copy())

        # 提取倾斜特征，使得fft的特征更明显，同时也是归一化
        img_clr = partSobel(img_std.copy(), 1000)

        # 傅里叶变换
        img_fft = absFFTshift(img_clr.copy())
        lowFreqFilter(img_fft)

        # 过滤由于partSobel产生的噪声
        img_fft_clr = verticalDelete(img_fft)

        # 幂运算
        img_fft_pow = pow(img_fft_clr, 2)

        # 仅取中心部分
        img_fft_crop = imgCrop(img_fft_pow)

        # 傅里叶变换并取幅值
        img_fe = absFFTshift(img_fft_crop.copy())
        lowFreqFilter(img_fe)

        # 更严格的取向判断
        img_fe_clr = verticalDelete(img_fe)

        ## 开始搜寻结果 ##
        img_fe_ = img_fe_clr
        polar = xycrd2polarcrd(img_fe_, rangeV=1, rangedivR=2.5, zeroNum=20)
        sum_list = np.sum(polar, axis=1)
        res = np.argmax(sum_list)

        if res < 2:
            res = 2
        if res > 88:
            res = 88

        theta = 2
        precision = 0.1
        polar_2 = xycrd2polarcrd(
            img_fe_, res, theta, precision, rangeV=1, rangedivR=2.5, zeroNum=20
        )
        sum_list_2 = np.sum(polar_2, axis=1)

        res = (res - theta) + (np.argmax(sum_list_2) * precision)

        result = 90 - res if res <= 90 else res - 90

        self.proImgs["std"] = normalize_img(img_std)
        self.proImgs["clr"] = normalize_img(img_clr)
        self.proImgs["fft"] = normalize_img(img_fft)
        self.proImgs["fftclr"] = normalize_img(img_fft_clr)
        self.proImgs["fftcrop"] = normalize_img(img_fft_crop)

        self.proImgs["ifft"] = normalize_img(img_fe)

        self.proImgs["sum"] = line_chart_img(sum_list)
        self.proImgs["FFTRES"] = add_angle_img(self.proImgs["fft"], -result, 200)
        self.proImgs["IFFTRES"] = add_angle_img(self.proImgs["ifft"], 90 - result, 200)

        self.proDatas["sumlist"] = sum_list.copy()

        return result

    def sti2angle_FFT(self, img):
        # 消除不同位置的竖直亮度差异。可以改进的地方为以突变边界消除（即50像素）
        # 作用：可以突出高亮度区域将低亮度区域的特征淹没了
        # img_std = std_filter(img.copy())

        # 提取倾斜特征，使得fft的特征更明显，同时也是归一化
        # img_clr = partSobel(img_std.copy(), 1000)

        # 傅里叶变换
        img_fft = absFFTshift(img.copy())
        lowFreqFilter(img_fft)

        # 过滤由于partSobel产生的噪声
        # img_fft_clr = verticalDelete(img_fft)

        img_fe_ = np.log(
            absFFTshift(img.copy())
        )  # verticalDelete(np.log(absFFTshift(img.copy())))
        lowFreqFilter(img_fe_)
        rangeV_ = 0
        rangedivR_ = 2
        zeroNum_ = 10

        ## 开始搜寻结果 ##
        res = 135
        theta = 45
        precision = 1
        polar = xycrd2polarcrd(
            img_fe_,
            res,
            theta,
            precision,
            rangeV=rangeV_,
            rangedivR=rangedivR_,
            zeroNum=zeroNum_,
        )
        sum_list = np.sum(polar, axis=1)
        res = (res - theta) + (np.argmax(sum_list) * precision)

        if res < 92:
            res = 92
        if res > 178:
            res = 178

        theta = 2
        precision = 0.1
        polar_2 = xycrd2polarcrd(
            img_fe_,
            res,
            theta,
            precision,
            rangeV=rangeV_,
            rangedivR=rangedivR_,
            zeroNum=zeroNum_,
        )
        sum_list_2 = np.sum(polar_2, axis=1)

        res = (res - theta) + (np.argmax(sum_list_2) * precision)

        result = res if res <= 90 else 180 - res

        # self.proImgs["std"] =  toImg(img_std)
        # self.proImgs["clr"] =  toImg(img_clr)
        self.proImgs["fft"] = normalize_img(img_fft)
        # self.proImgs["fftclr"] =  toImg(img_fft_clr)

        self.proImgs["fe"] = normalize_img(np.log(absFFTshift(img.copy())))
        self.proImgs["sum"] = line_chart_img(sum_list)
        self.proImgs["FFTRES"] = add_angle_img(self.proImgs["fe"], -result, 200)
        return result

    def sti2angle(self, img):
        # 变灰度图
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.proImgs["ORIGIN"] = img
        res = self.sti2angle_IFFT(img.copy())
        self.proImgs["STIRES"] = add_angle_img(img, 90 - res)

        return res, self.proImgs, self.proDatas