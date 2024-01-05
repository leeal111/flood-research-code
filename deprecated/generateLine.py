# -*- coding: utf-8 -*-
# 已知深度求像素位置

import cv2
import numpy as np
import math


def worldToPixel(order, flood_height, line):
    order = int(order)

    # zoom_order = 40(预置点位1）,35（预置点位2）,20（预置点位3）,8（预置点位4）,2（预置点位5）
    fx_range = [88839.564, 70899.7147, 48170.7513, 21158.3803, 6478.44226, 6478.44226]
    cx_range = [1919.85631, 1920.25843, 1920.11447, 1929.9872, 1917.03337, 1917.03337]
    fy_range = [103354.361, 71936.4665, 49556.6291, 21351.6852, 6463.06821, 6463.06821]
    cy_range = [1080.20694, 1079.59805, 1080.80066, 1084.90231, 1082.77481, 1082.77481]

    k1_range = [
        -5.59444899,
        -4.36936102,
        -0.52838596,
        0.607502233,
        0.028468785,
        0.028468785,
    ]
    k2_range = [
        -0.003608052,
        -0.01609312,
        -0.01141387,
        -25.6117614,
        1.60334895,
        1.60334895,
    ]
    p1_range = [
        0.0173209945,
        0.0059306887,
        0.02625075,
        0.00000470704571,
        -0.00566590458,
        -0.00566590458,
    ]
    p2_range = [
        0.0130225839,
        -0.00116745487,
        0.00695651,
        0.000327414736,
        -0.00305205746,
        -0.00305205746,
    ]
    k3_range = [
        -0.0000278552301,
        -0.141406815,
        0.00486861,
        -0.370078468,
        -13.2985282,
        -13.2985282,
    ]

    # 俯仰角
    theta_range = [2.53, 3.63, 5.23, 8.58, 19.23, 31.88]

    # 相机坐标系到像素坐标系的转换矩阵
    # print('---请输入内参矩阵！---')
    fx = fx_range[order]
    cx = cx_range[order]
    fy = fy_range[order]
    cy = cy_range[order]
    # print('---请输入畸变系数！---')
    k1 = k1_range[order]
    k2 = k2_range[order]
    p1 = p1_range[order]
    p2 = p2_range[order]
    k3 = k3_range[order]
    theta = theta_range[order]

    mat_intri = np.mat([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # 畸变系数
    coff_dis = np.mat([k1, k2, p1, p2, k3])

    w = 3840  # 宽度
    h = 2160  # 高度
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mat_intri, coff_dis, (w, h), 1, (w, h)
    )
    fy = newcameramtx[1][1]
    theta = theta * math.pi / 180

    # 547.4766是球机架设的十字架中心的高程，0.4是十字架中心到球机光心的距离
    # Y 是球机相对水平面的高度
    Y = 547.4766 - flood_height - 0.4

    r = {}
    # 预置点位
    r["order"] = order

    fi = math.atan(line / Y)
    y1 = fy * math.tan(math.pi / 2 - theta - fi) + 1080
    # 测速线的像素纵坐标
    r["y_pixel"] = y1

    l = Y / math.cos(fi)
    Z = l * 750 / fx
    # 750个像素对应的真实世界长度
    r["realLength_750"] = Z
    return r


if __name__ == "__main__":
    # with open("temp/generateLine.txt", "r") as file:
    #     line = file.readline().strip()  # 读取文件的一行并去除首尾空格
    # # print(line)
    # # 将字符串拆分为单个值
    # values = line.split()
    # # print(values)
    # # 将字符串值转换为双精度浮点数
    # value1 = float(values[0])
    # value2 = float(values[1])
    # value3 = float(values[2])

    result = []

    # 水位高程
    flood_height = 535  # value1

    # 右水边距，来自水文站查表所得，会改变
    right_flood_distance = 20  # value2
    # 左水边距，来自水文站查表所得，会改变
    left_flood_distance = 260  # value3

    # 测速线间隔
    line_interval = 5

    # 球机到右水边的深度
    camera_to_right_flood = right_flood_distance - 21.07
    # 球机到左水边的深度
    camera_to_left_flood = left_flood_distance - 21.07
    # 水面宽度
    flood_width = camera_to_left_flood - camera_to_right_flood

    # 测速线深度
    line_range = np.arange(camera_to_right_flood, camera_to_left_flood, line_interval)
    line_range = line_range[1 : len(line_range) - 1]

    order = 5
    for line in line_range:
        item = worldToPixel(order, flood_height, line)
        if item["y_pixel"] < 0:
            if order == 0:
                print(f"error y_pixel {item['y_pixel']}")
                item["y_pixel"] = 0
                continue
            order = order - 1
            item = worldToPixel(order, flood_height, line)
        elif item["y_pixel"] > 2160:
            print(f"error y_pixel {item['y_pixel']}")
            item["y_pixel"] = 2159
        result.append(item)

    # for line in line_range:
    #     if line >= 13.0 and line < 25.0:
    #         order = 5
    #     elif line >= 25.0 and line < 60.0:
    #         order = 4
    #     elif line >= 60.0 and line < 100.0:
    #         order = 3
    #     elif line >= 100.0 and line < 140.0:
    #         order = 2
    #     elif line >= 140.0 and line < 200.0:
    #         order = 1
    #     elif line >= 200.0 and line < 285.0:
    #         order = 0
    #     result.append(worldToPixel(order, flood_height, line))

    with open("temp/generateLine_result.txt", "w") as file:
        for item in result:
            # 将数组中的每个元素格式化为字符串
            array_str = " ".join(
                [str(item["order"]), str(item["y_pixel"]), str(item["realLength_750"])]
            )
            # print(array_str)
            # 将数组字符串写入文件
            file.write(array_str + "\n")
