# MTCNN state-of-the-art face detection method

## Update 2017.09.24

* 增加了跨平台编译的能力

* 适配至最新版caffe

* 增加了轻量级版本，便于移植到android平台

## 概述

[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)是[Kaipeng Zhang](https://kpzhang93.github.io/)等人提出的多任务级联卷积神经网络进行人脸检测的方法，是迄今为止开放源码的效果最好的人脸检测器之一，[在fddb上有100个误报时的检出率高达90%以上](https://github.com/imistyrain/fddb-windows)，作者提供的版本为[matlab版](https://github.com/kpzhang93/MTCNN_face_detection_alignment),它最终的效果如图所示：

![](https://i.imgur.com/FbglxoX.jpg)


## 运行方法

1.编译最新版[caffe](https://github.com/BVLC/caffe)，这个网上已有很多[教程](http://blog.csdn.net/akashaicrecorder/article/details/71016942),恕不赘述

2.打开MTCNN.sln，把MTCNN设为启动项。

3.编译mxnet的windows版，参考[mxnet VS2015编译
](https://github.com/imistyrain/mxnet-oneclick/blob/master/mxnet%20VS2015%E7%BC%96%E8%AF%91.pdf)，然后打开MTCNN.sln,把MTCNNPy设为启动项.加载此工程需要安装VS python的插件[PTVS 2.2.6 VS 2015](https://github.com/Microsoft/PTVS/releases/v2.2.6)

本机测试环境为VS2015,Cuda8.0,CuDnn5.1,python2.7

## 参考


*  [Win10+VS2015 caffe环境搭建](http://blog.csdn.net/akashaicrecorder/article/details/71016942)

* [fddb-windows](https://github.com/imistyrain/fddb-windows)