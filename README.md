# 复杂场景下基于雷视融合的流速测量技术

## 简介

本项目针对水利行业对非接触式测流/测水位等需求，重点面向中型河道、生态流量监测两大关键领域，研究基于视频+雷视融合流速、水位测量技术及装备，解决常规使用的转子流速仪法、声学法、雷达法等仪器设备，仅能测量点、线或局部水面流速，且施测的时效性和对高洪期、浅水域、强紊动、高含沙及易感潮水流等复杂流态的适宜性严重受限的难题。

## 硬件环境介绍

* jetson agx
* 水位计
* 相机
* 灯光

## 快速开始

1.环境安装

```Shell
pip install -r requirements.txt
```

2.构建、编译

```Shell
sh make.sh
```

3.输入

* deep.txt：河流断面数据
* preset.txt：预置点位数据

4.开始运行

```shell
# 开始读取视频、记录视频并生成sti图
./flood
```

5.输出

* 程序会以在“/data/”下以当前时间为文件夹名称创建文件夹，并在其中保存输出的视频、sti图和算法计算的流速等。

  ![result.png](./result.png)

  * result.png：将结果可视化
  * result.txt：包含水位、全站仪近远岸距离、测速线的位置信息、测速结果
  * *.mp4：相机拍摄的视频
  * *.jpg：sti图
  * *_loc.jpg：测速线可视化结果

* 程序运行的状态可在命令行窗口查看。

## 代码详细讲解

|              文件路径              |           功能介绍           |
| :--------------------------------: | :--------------------------: |
|           ./src/main.cpp           |           程序入口           |
|     ./src/STIVDeepLearning.cpp     | 在c++代码中调用深度学习算法  |
|      ./src/STIVTradition.cpp       |   在c++代码中调用传统算法    |
|     ./src/WaterLevelDevice.cpp     |   控制水位计、读取水位高度   |
|       ./src/LightDevice.cpp        |           控制灯光           |
|      ./src/FlowAlgorithm.cpp       |         测流算法基类         |
|    ./src/CameraCalibration.cpp     |           相机标定           |
|       ./src/CamaraDevice.cpp       | 控制相机移动、录制视频等功能 |
|       ./python/deeplearning/       |      深度学习算法源代码      |
|        ./python/tradition/         |        传统算法源代码        |
| ./python/corrVideoAndGetSTIPath.py |     矫正视频并生成STI图      |
|      ./python/generateLine.py      |        确定测速线坐标        |
|        ./python/lightOff.py        |         控制灯光关闭         |
|        ./python/lightOn.py         |         控制灯光打开         |
|       ./python/resultPlot.py       |          可视化代码          |
