# MTCNN-VS

MTCNN对应的VS工程

[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)是[Kaipeng Zhang](https://kpzhang93.github.io/)等人提出的多任务级联卷积神经网络进行人脸检测的方法，作者提供的版本为matlab版，地址：[https://github.com/kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment),它最终的效果如图所示：

![](http://i.imgur.com/A5sVHXh.png)

工程化应用亟需C++版本，由Yanfu Ren首先放出了它实现的版本，地址为[https://github.com/DaFuCoding/MTCNN_Caffe](https://github.com/DaFuCoding/MTCNN_Caffe)，这个版本存在两个问题：

1：和作者实现的matlab版本有偏差，原因待查

2：把代码和caffe库放在了一块，这对于从来没有使用过caffe的是莫大的福音，但是已经装过caffe的却没必要下载整个库

此外，yuanyang基于也实现了基于python的mxnet版本，这些都为开源实现做出了有益的贡献。

这个project的目的不是重复去实现那些已经实现的功能，而是吸收已有成果，便于程化应用。

运行方法：

1.编译caffe-windows版，参考[https://github.com/happynear/caffe-windows](https://github.com/happynear/caffe-windows)，特别注意[FAQ](https://github.com/happynear/caffe-windows/blob/master/FAQ.md)中提到的，防止出现重复注册的error：

Another method is to add layer_factory.cpp and force_link.cpp to your own project, to let the compiler know the existence of the layers and solvers. However, when using this method, the layers will be registered twice and you will get an error in include/caffe/layer_factory.hpp line 68. To fix this error, you can just remove line 68-69 or use if(registry.count(type) > 0) continue; to replace the CHECK statement.

2.打开MTCNN.sln，把MTCNN设为启动项。

3.编译mxnet的windows版，参考[mxnet VS2013编译
](https://github.com/imistyrain/mxnet-mr/blob/master/mxnet%20VS2013%E7%BC%96%E8%AF%91.pdf)，然后打开MTCNN.sln,把MTCNNPy设为启动项

本机测试环境为VS2013,Cuda7.0,CuDnn4.0,python2.7

参考：

1.[https://github.com/kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

2.[https://github.com/DaFuCoding/MTCNN_Caffe](https://github.com/DaFuCoding/MTCNN_Caffe)

3.[https://github.com/pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)