# 【Github项目文档】DeepLearningTutorial项目说明

**Deep Learning,Leaning deep,Have fun!**
# 介绍
如果你是深度学习/卷积神经网络的初学者，且对图像分类、目标检测、分割等CV相关领域感兴趣，请继续<br />**↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓**<br />刚刚入门DL，CV，CNN？或者了解各种理论后仍不知从何下手 ？是不是对于各个网络模型的代码实现一脸懵逼？如果是，那么这个项目就是为你准备的。**Talk is cheap,show me the code!本项目致力于图像分类网络(经典CNN)、目标检测、实例分割等一切CV相关领域的论文/网络解读 + 代码构建 + 模型训练**(在1.和2.部分)；在第3.学习资源部分里分享深度学习，计算机视觉相关的文章、视频公开课、开源框架、项目和平台等和一切**深度学习相关的优秀资源**；第4部分是tensorflow和pytorch上的**公开数据集**<br />好东西要共享，Ideas worth spreading！项目不定期更新。<br />**目录如下：**

- [介绍](https://github.com/Flowingsun007/DeepLearningTutorial#%E4%BB%8B%E7%BB%8D)
- [1.图像分类Image Classification](https://github.com/Flowingsun007/DeepLearningTutorial#1图像分类image-classification)
- [2.目标检测Object Detection](https://github.com/Flowingsun007/DeepLearningTutorial#2目标检测object-detection)
  - [2.1 One-stage](https://github.com/Flowingsun007/DeepLearningTutorial#21-one-stage)
  - [2.2 Two-stage](https://github.com/Flowingsun007/DeepLearningTutorial#22-two-stage)
  - [2.3 资源分享](https://github.com/Flowingsun007/DeepLearningTutorial#23-资源分享)
    - [2.3.1 知乎](https://github.com/Flowingsun007/DeepLearningTutorial#231-知乎)
    - [2.3.2 论文](https://github.com/Flowingsun007/DeepLearningTutorial#232-论文)
    - [2.3.3 代码实战](https://github.com/Flowingsun007/DeepLearningTutorial#233-代码实战)
- [3.学习资源](https://github.com/Flowingsun007/DeepLearningTutorial#3%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%BA%90)
  - [3.1 机器学习](https://github.com/Flowingsun007/DeepLearningTutorial#31-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)
    - [3.1.1 入门概念](https://github.com/Flowingsun007/DeepLearningTutorial#311-%E5%85%A5%E9%97%A8%E6%A6%82%E5%BF%B5)
    - [3.1.2 公开课](https://github.com/Flowingsun007/DeepLearningTutorial#312-%E5%85%AC%E5%BC%80%E8%AF%BE)
    - [3.1.3 学习资源](https://github.com/Flowingsun007/DeepLearningTutorial#313-%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%BA%90)
    - [3.1.4 竞赛平台](https://github.com/Flowingsun007/DeepLearningTutorial#314-%E7%AB%9E%E8%B5%9B%E5%B9%B3%E5%8F%B0)
  - [3.2 深度学习](https://github.com/Flowingsun007/DeepLearningTutorial#32-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0)
    - [3.2.1 入门概念](https://github.com/Flowingsun007/DeepLearningTutorial#321-%E5%85%A5%E9%97%A8%E6%A6%82%E5%BF%B5)
    - [3.2.2 视频公开课](https://github.com/Flowingsun007/DeepLearningTutorial#322-%E8%A7%86%E9%A2%91%E5%85%AC%E5%BC%80%E8%AF%BE)
    - [3.2.3 学习资源](https://github.com/Flowingsun007/DeepLearningTutorial#323-%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%BA%90)
      - [书PDF](https://github.com/Flowingsun007/DeepLearningTutorial#书PDF)
      - [卷积神经网络CNN](https://github.com/Flowingsun007/DeepLearningTutorial#卷积神经网络CNN)
      - [目标检测Object Detection](https://github.com/Flowingsun007/DeepLearningTutorial#目标检测ObjectDetection)
    - [3.2.4  开源工具](https://www.yuque.com/zhaoluyang/ai/vgn4pv#hgkH7)
      - [深度学习框架](https://github.com/Flowingsun007/DeepLearningTutorial#%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6)
      - [支撑工具](https://github.com/Flowingsun007/DeepLearningTutorial#%E6%94%AF%E6%92%91%E5%B7%A5%E5%85%B7)
      - [其他资源](https://github.com/Flowingsun007/DeepLearningTutorial#%E5%85%B6%E4%BB%96%E8%B5%84%E6%BA%90)
  - [3.3 计算机视觉](https://github.com/Flowingsun007/DeepLearningTutorial#33-%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)
    - [3.3.1 入门概念](https://github.com/Flowingsun007/DeepLearningTutorial#331-%E5%85%A5%E9%97%A8%E6%A6%82%E5%BF%B5)
    - [3.3.2 公开课](https://github.com/Flowingsun007/DeepLearningTutorial#332-%E5%85%AC%E5%BC%80%E8%AF%BE)
    - [3.3.3 学习资源](https://github.com/Flowingsun007/DeepLearningTutorial#333-%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%BA%90)
- [4.公开数据集](https://github.com/Flowingsun007/DeepLearningTutorial#4%E5%85%AC%E5%BC%80%E6%95%B0%E6%8D%AE%E9%9B%86)
  - [4.1 Pytorch提供](https://github.com/Flowingsun007/DeepLearningTutorial#41-Pytorch%E6%8F%90%E4%BE%9B)
  - [4.2 Tensorflow提供](https://github.com/Flowingsun007/DeepLearningTutorial#42-Tensorflow%E6%8F%90%E4%BE%9B)

---

# 1.图像分类Image Classification
| 项目✓ | 论文✓ | 网络✓ | 模型训练✓ |
| :---: | :---: | :---: | :---: |
| **LeNet** | [1998](https://ieeexplore.ieee.org/document/726791?reload=true&arnumber=726791)            [论文解读](https://zhuanlan.zhihu.com/p/34311419) | [LeNet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/network/LeNet.py) | [train_lenet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/train_lenet.py) |
| **AlexNet** | [2012-PDF](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)    [论文解读](https://zhuanlan.zhihu.com/p/107660669) | [AlexNet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/network/AlexNet.py) | [train_alexnet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/train_alexnet.py) |
| **Network in Network** | [2013-PDF](http://arxiv.org/pdf/1312.4400)    [论文解读](https://zhuanlan.zhihu.com/p/108235295) | [NetworkInNetwork.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/network/NetworkInNetwork.py) | [train_nin.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/train_nin.py) |
| **VGG** | [2014-PDF](https://arxiv.org/pdf/1409.1556.pdf)    [论文解读](https://zhuanlan.zhihu.com/p/107884876) | [VGG.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/network/VGG.py) | [train_vgg.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/train_vgg.py) |
| **GoogLeNet** | [2014-PDF](https://arxiv.org/pdf/1409.4842)    [论文解读](https://zhuanlan.zhihu.com/p/108414921) | [GoogLeNet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/network/GoogLenet.py) | [train_googlenet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/train_googlenet.py) |
| **ResNet** | [2015-PDF](https://arxiv.org/pdf/1512.03385.pdf)    [论文解读](https://zhuanlan.zhihu.com/p/108708768) | [ResNet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/network/ResNet.py) | [train_resnet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/train_resnet.py) |
| **DenseNet** | [2016-PDF](https://arxiv.org/pdf/1608.06993.pdf)    [论文解读](https://zhuanlan.zhihu.com/p/109269085) | [DenseNet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/network/DenseNet.py) | [train_densenet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/train_densenet.py) |
| **ShuffleNet** | [2017-PDF](https://arxiv.org/pdf/1707.01083)    [论文解读](https://zhuanlan.zhihu.com/p/32304419) | [shuffleNet.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/ShuffleNet.py) | ✗ |
| **ShuffleNetV2** | [2018-PDF](https://arxiv.org/pdf/1807.11164)    [论文解读](https://zhuanlan.zhihu.com/p/48261931) | [ShuffleNetV2.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/shufflenet_v2.py) | ✗ |
| **MobileNet** | [V1](https://arxiv.org/abs/1704.04861)   [V2](https://128.84.21.199/pdf/1801.04381.pdf)   [V3](https://arxiv.org/pdf/1905.02244.pdf)  [论文解读](https://zhuanlan.zhihu.com/p/70703846) | [MobileNetV3.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/network/MobileNetV3.py) | [train_mobilenet.py](https://github.com/Flowingsun007/DeepLearningTutorial/blob/master/ImageClassification/train_mobilenet.py) |


---

# 2.目标检测Object Detection
## 2.1 One-stage
| 项目 | 论文 | 网络 | 模型训练 |
| :---: | :---: | :---: | :---: |
| **YoloV1** | [CVPR'16](http://arxiv.org/abs/1506.02640)  [论文解读](https://zhuanlan.zhihu.com/p/32525231) | ☐ | [官方-darknet](https://pjreddie.com/darknet/yolov1/)    [tensorflow](https://github.com/gliese581gg/YOLO_tensorflow) |
| **SSD** | [ECCV'16](http://arxiv.org/abs/1512.02325)  [论文解读](https://zhuanlan.zhihu.com/p/33544892) | ☐ | [官方-caffe](https://github.com/weiliu89/caffe/tree/ssd)  [tensorflow](https://github.com/balancap/SSD-Tensorflow)  [pytorch](https://github.com/amdegroot/ssd.pytorch) |
| **YoloV2** | [CVPR'17](https://arxiv.org/pdf/1612.08242.pdf)  [论文解读](https://zhuanlan.zhihu.com/p/35325884) | ☐ | [官方-darknet](https://pjreddie.com/darknet/yolov2/)    [tf](https://github.com/hizhangp/yolo_tensorflow)    [tf](https://github.com/KOD-Chen/YOLOv2-Tensorflow)   [pytorch](https://github.com/longcw/yolo2-pytorch) |
| **RetinaNet** | [ICCV'17](https://arxiv.org/pdf/1708.02002.pdf)   [论文解读](https://zhuanlan.zhihu.com/p/68786098) | ☐ | [官方-keras](https://github.com/fizyr/keras-retinanet) |
| **YoloV3** | [arXiV'18](https://arxiv.org/abs/1804.02767)  [论文翻译](https://zhuanlan.zhihu.com/p/37201615) | ☐ | [官方-darknet](https://github.com/pjreddie/darknet)    [tf](https://github.com/mystic123/tensorflow-yolo-v3)    [tf2.0](https://github.com/Flowingsun007/DeepLearningTutorial/tree/master/ObjectDetection/Yolo)    [pytorch](https://github.com/eriklindernoren/PyTorch-YOLOv3) |
| **NAS-FPN** | [CVPR'19](https://arxiv.org/abs/1904.07392)  [论文解读](https://zhuanlan.zhihu.com/p/97230695) | ☐ | ☐ |
| **EfficientNet** | [arXiV'19](https://arxiv.org/pdf/1911.09070v1.pdf)  [论文解读](https://zhuanlan.zhihu.com/p/104790514) | ☐ | [官方-tensorflow](https://github.com/google/automl/tree/master/efficientdet) |

## 2.2 Two-stage
| 项目 | 论文 | 网络 | 模型训练 |
| :---: | :---: | :---: | :---: |
| **R-CNN** | [CVPR'14](https://arxiv.org/pdf/1311.2524.pdf)  [论文解读+翻译](https://zhuanlan.zhihu.com/p/115060099) | ☐ | [官方-caffe](https://github.com/rbgirshick/rcnn) |
| **Fast R-CNN** | [ICCV'15](https://arxiv.org/pdf/1504.08083.pdf)   [解读1](https://zhuanlan.zhihu.com/p/79054417)  [解读2](https://zhuanlan.zhihu.com/p/60968116) | ☐ | [官方-caffe](https://github.com/rbgirshick/fast-rcnn) [tensorflow](https://github.com/zplizzi/tensorflow-fast-rcnn) |
| **Faster R-CNN** | [NIPS'15](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)   [解读1](https://zhuanlan.zhihu.com/p/82185598)  [解读2](https://zhuanlan.zhihu.com/p/61202658) | ☐ | [官方-caffe](https://github.com/rbgirshick/py-faster-rcnn)   [tensorflow](https://github.com/endernewton/tf-faster-rcnn)   [pytorch](https://github.com/jwyang/faster-rcnn.pytorch) |
| **FPN** | [CVPR'17](https://arxiv.org/abs/1612.03144)   [解读1](https://zhuanlan.zhihu.com/p/62604038) [解读2](https://zhuanlan.zhihu.com/p/62604038) | ☐ | [caffe](https://github.com/unsky/FPN) |
| **Mask R-CNN** | [ICCV'17](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)    [解读1](https://zhuanlan.zhihu.com/p/37998710)  [解读2](https://zhuanlan.zhihu.com/p/65321082) | ☐ | [官方-caffe2](https://github.com/facebookresearch/Detectron)   [tf](https://github.com/matterport/Mask_RCNN)   [tf](https://github.com/CharlesShang/FastMaskRCNN)   [pytorch](https://github.com/multimodallearning/pytorch-mask-rcnn) |
| **ThunderNet** | [ICCV'19](https://arxiv.org/pdf/1903.11752.pdf)    [论文解读](https://zhuanlan.zhihu.com/p/61113865) | ☐ | ☐ |

## 2.3 资源分享
### 2.3.1 知乎

- [基于深度学习的目标检测算法综述（一）](https://zhuanlan.zhihu.com/p/40047760)
- [基于深度学习的目标检测算法综述（二）](https://zhuanlan.zhihu.com/p/40020809)
- [基于深度学习的目标检测算法综述（三）](https://zhuanlan.zhihu.com/p/40102001)
- [干货 | 目标检测入门，看这篇就够了（已更完）](https://zhuanlan.zhihu.com/p/34142321)
- [51 个深度学习目标检测模型汇总，论文、源码一应俱全！](https://zhuanlan.zhihu.com/p/55519131)
- [two/one-stage,anchor-based/free目标检测发展及总结：一文了解目标检测](https://zhuanlan.zhihu.com/p/100823629)
### 2.3.2 论文
**【论文合集】**

- [目标检测相关论文deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)
- [目标检测发展、论文综述](https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html)
- [干货 | 目标检测入门，看这篇就够了（已更完）](https://zhuanlan.zhihu.com/p/34142321)
- [awesome-object-detection](https://github.com/amusi/awesome-object-detection)
- [【目标检测论文解读】ObjectDetection—R-CNN](https://zhuanlan.zhihu.com/p/115060099)
- [【目标检测论文解读】ObjectDetection—Fast R-CNN](https://zhuanlan.zhihu.com/p/121658700)
- [【目标检测论文解读】ObjectDetection—Faster R-CNN](https://zhuanlan.zhihu.com/p/121676212)
- [【目标检测论文解读】ObjectDetection—YoloV3论文+代码+资源合集](https://zhuanlan.zhihu.com/p/122229193)

**【发展综述】**

- [**Object Detection in 20 Years: A Survey**](https://arxiv.org/abs/1905.05055)
- [**A Survey of Deep Learning-based Object Detection**](https://arxiv.org/abs/1907.09408)
- **[Imbalance Problems in Object Detection: A Review](https://arxiv.org/abs/1909.00169)**
- [**Recent Advances in Deep Learning for Object Detection**](https://arxiv.org/abs/1908.03673)
- [**《Deep Learning for Generic Object Detection: A Survey》**](https://arxiv.org/abs/1809.02165)
- [**《Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks》**](https://arxiv.org/abs/1809.03193)
### 2.3.3 代码实战
#### 目标检测ObjectDetection
- [【github】Detectron2——facebook开源目标检测框架](https://github.com/facebookresearch/detectron2)
- [【github】mmdetection——商汤科技+香港中文大学开源目标检测框架](https://github.com/open-mmlab/mmdetection)
- [【github】TensorFlow2.0-Examples](https://github.com/YunYang1994/TensorFlow2.0-Examples)
- [【github】awesome-object-detection](https://github.com/amusi/awesome-object-detection)
- [【目标检测实战】Darknet—yolov3模型训练（VOC数据集)](https://zhuanlan.zhihu.com/p/92141879)
- [【目标检测实战】Pytorch—SSD模型训练（VOC数据集）](https://zhuanlan.zhihu.com/p/92154612)



---

# 3.学习资源
## 3.1 机器学习
### 3.1.1 入门概念

- [机器学习温和指南](http://link.zhihu.com/?target=https%3A//www.csdn.net/article/2015-09-08/2825647)
- [有趣的机器学习：最简明入门指南](http://link.zhihu.com/?target=http%3A//blog.jobbole.com/67616/)
- [一个故事说明什么是机器学习](http://link.zhihu.com/?target=https%3A//www.cnblogs.com/subconscious/p/4107357.html)
- [cstghitpku：干货|机器学习超全综述！](https://zhuanlan.zhihu.com/p/46320419)
- [机器学习该怎么入门？](https://www.zhihu.com/question/20691338)
- [如何系统入门机器学习？](https://www.zhihu.com/question/266127835)
- [机器学习该怎么入门？](https://www.zhihu.com/question/20691338)
### 3.1.2 公开课

- **加州理工学院**[Learning from data(费曼奖得主Yaser Abu-Mostafa教授)](http://work.caltech.edu/lectures.html)
- **谷歌** [Google 制作的节奏紧凑、内容实用的机器学习简介课程](https://developers.google.com/machine-learning/crash-course/)
- **林軒田**[[機器學習基石]Machine Learning Foundations——哔哩哔哩](https://www.bilibili.com/video/av1624332?p=2)

**网易**<br />[![](https://camo.githubusercontent.com/eea35fe0589e9d1b64c41af9bf73be0dbf162201/68747470733a2f2f63646e2e6e6c61726b2e636f6d2f79757175652f302f323032302f706e672f3231363931342f313538343432353633383431312d35386561636636342d646366352d343333322d393435622d6637393366343562346637302e706e6723616c69676e3d6c65667426646973706c61793d696e6c696e65266865696768743d323530266f726967696e4865696768743d323530266f726967696e57696474683d3435302673697a653d30267374617475733d646f6e65267374796c653d6e6f6e652677696474683d343530#align=left&display=inline&height=250&originHeight=250&originWidth=450&status=done&style=none&width=450)](https://study.163.com/course/introduction/1004570029.htm)<br />[吴恩达机器学习](https://study.163.com/course/introduction/1004570029.htm)<br />网易杭州研究院<br />Google Brain 和百度大脑的发起人、Coursera 创始人吴恩达（Andrew Ng）亲授，在全球有百万选课量，主要讲述人工智能中基础的机...[查看详情](https://study.163.com/course/introduction/1004570029.htm)<br />中文教学的优质课程加上贴近生活的案例，你将在学习AI的道路上跑得更快！<br />[![](https://camo.githubusercontent.com/f09e216a5474a81adf2212e5a5e1900385fb0218/68747470733a2f2f63646e2e6e6c61726b2e636f6d2f79757175652f302f323032302f706e672f3231363931342f313538343432353633383532352d30666138366238632d643039372d346335632d383835642d3634313630386232346562302e706e6723616c69676e3d6c65667426646973706c61793d696e6c696e65266865696768743d323439266f726967696e4865696768743d323530266f726967696e57696474683d3435302673697a653d30267374617475733d646f6e65267374796c653d6e6f6e652677696474683d343439#align=left&display=inline&height=250&originHeight=250&originWidth=450&status=done&style=none&width=450)](https://study.163.com/course/introduction/1208946807.htm)<br />[李宏毅机器学习中文课程](https://study.163.com/course/introduction/1208946807.htm)<br />网易云课堂IT互联网<br />来自台湾大学李宏毅老师的课程，以精灵宝可梦作为课程案例，生动地为你讲解机器学习。同时，他还设计了六项作业和一项期末项目，...[查看详情](https://study.163.com/course/introduction/1208946807.htm)<br />[机器学习及其深层与结构化](https://study.163.com/course/introduction/1208991809.htm)<br />网易云课堂IT互联网<br />台湾大学李宏毅老师在《机器学习》基础上提供的《机器学习及其深度与结构化》课程，为你深入解析深度学习与结构学习。[查看详情](https://study.163.com/course/introduction/1208991809.htm)<br />[李宏毅线性代数中文课程](https://study.163.com/course/introduction/1208956807.htm)<br />网易云课堂IT互联网<br />来自台湾大学李宏毅老师的课程，专为对人工智能感兴趣，但是数学基础薄弱的同学设计，让你深刻理解数学概念，学会在人工智能应用...[查看详情](https://study.163.com/course/introduction/1208956807.htm)<br />[机器学习前沿技术](https://study.163.com/course/introduction/1209400866.htm)<br />网易云课堂IT互联网<br />机器学习的下一步是什么？机器能不能知道“我不知道”、“我为什么知道”，机器的错觉，终身学习<br />[查看详情](https://study.163.com/course/introduction/1209400866.htm)
### 3.1.3 学习资源
**【书】**

- [周志华《机器学习》公式推导在线阅读](https://datawhalechina.github.io/pumpkin-book/#/)

**【知乎】**

- [机器学习科研的十年](https://zhuanlan.zhihu.com/p/74249758)
- [机器学习最好的课程是什么？](https://www.zhihu.com/question/37031588/answer/723461499)
- [**吴恩达机器学习笔记整理**](https://zhuanlan.zhihu.com/p/75173557)
- **第一周**[单变量线性回归和损失函数、梯度下降的概念](https://zhuanlan.zhihu.com/p/73363177)
- **第二周**[多变量线性回归和特征缩放、学习率](https://zhuanlan.zhihu.com/p/73403012)
- **第三周**[分类问题逻辑回归和过拟合、正则化](https://zhuanlan.zhihu.com/p/73404297)
- **第四周**[神经元、神经网络和前向传播算法](https://zhuanlan.zhihu.com/p/73665825)
- **第五周**[神经网络、反向传播算法和随机初始化](https://zhuanlan.zhihu.com/p/74167352)
- **第六周**[应用机器学习的建议和系统设计](https://zhuanlan.zhihu.com/p/75326539)
- **第七周**[支持向量机SVM和核函数](https://zhuanlan.zhihu.com/p/74764135)
- **第八周**[聚类K-Means算法、降维和主成分分析](https://zhuanlan.zhihu.com/p/74902766)
- **第九周**[异常检测和高斯分布、推荐系统和协同过滤](https://zhuanlan.zhihu.com/p/75036754)
- **第十周**[大规模机器学习和随机梯度下降算法](https://zhuanlan.zhihu.com/p/75171589)
- [【机器学习理论】—mAP 查全率 查准率 IoU ROC PR曲线 F1值](https://zhuanlan.zhihu.com/p/92495276)
- [SVM教程：支持向量机的直观理解](https://zhuanlan.zhihu.com/p/40857202)
- [支持向量机(SVM)是什么意思？](https://www.zhihu.com/question/21094489/answer/86273196)

**【Github】**

- [Machine-Learning-Tutorials](https://github.com/ujjwalkarn/Machine-Learning-Tutorials)
- [李航《统计学习方法》——代码实现](https://github.com/fengdu78/lihang-code)
### 3.1.4 竞赛平台

- [Kaggle](https://www.kaggle.com/competitions)
- [阿里天池](https://tianchi.aliyun.com/home?spm=5176.12281949.0.0.493e2448ifo8Vz)
- [Kesci 和鲸社区](https://www.kesci.com/)
- [百度AI Studio](https://aistudio.baidu.com/aistudio/competition)
## 3.2 深度学习
### 3.2.1 入门概念

- [深度学习如何入门？](https://www.zhihu.com/question/26006703/answer/129209540)
- [有哪些优秀的深度学习入门书籍？需要先学习机器学习吗？](https://www.zhihu.com/question/36675272)
- [CNN（卷积神经网络）是什么？有何入门简介或文章吗？](https://www.zhihu.com/question/52668301)
- [从应用的角度来看，深度学习怎样快速入门？](https://www.zhihu.com/question/343407265/answer/830912894)
- [普通程序员如何正确学习人工智能方向的知识？](https://www.zhihu.com/question/51039416)
- [有哪些优秀的深度学习入门书籍？需要先学习机器学习吗？](https://www.zhihu.com/question/36675272/answer/603847513)
- [给妹纸的深度学习教学(0)——从这里出发](https://zhuanlan.zhihu.com/p/28462089)

**【梯度下降、深度神经网络、反向传播】**
- [【深度学习理论】一文搞透梯度下降Gradient descent](https://zhuanlan.zhihu.com/p/144478956)
- [【深度学习理论】纯公式手推+代码撸——神经网络的反向传播+梯度下降](https://zhuanlan.zhihu.com/p/145538299)
- [【深度学习理论】一文搞透pytorch中的tensor、autograd、反向传播和计算图](https://zhuanlan.zhihu.com/p/145353262)
- [神经网络为什么可以（理论上）拟合任何函数？](https://www.zhihu.com/question/268384579/answer/540793202)
- [道理我都懂，但是神经网络反向传播时的梯度到底怎么求？](https://zhuanlan.zhihu.com/p/25202034)

### 3.2.2 视频公开课
**3Blue1Brown**

- [【S301】But what is a Neural Network 什么是神经网络？](https://zhuanlan.zhihu.com/p/104263315)
- [【S302】Gradient descent, how neural networks learn 梯度下降，神经网络如何学习](https://zhuanlan.zhihu.com/p/104263315)
- [【S303】What is backpropagation really doing 反向传播是如何起作用的](https://zhuanlan.zhihu.com/p/104263315)
- [【S304】Backpropagation calculus 反向传播公式推导](https://zhuanlan.zhihu.com/p/104263315)[<br />](https://zhuanlan.zhihu.com/p/104263315)

**斯坦福**

- [斯坦福2017季CS224n深度学习自然语言处理课程](https://www.bilibili.com/video/av13383754/?from=search&seid=13189649321373413789)
- [斯坦福大学公开课 ：机器学习课程-吴恩达](http://open.163.com/special/opencourse/machinelearning.html)

**Coursera**

- [Machine Learning | Coursera](https://www.coursera.org/learn/machine-learning)

**李宏毅**<br />官方主页：[Hung-yi Lee](http://speech.ee.ntu.edu.tw/~tlkagk/talk.html)

- **YouTube Channel teaching Deep Learning and Machine Learning** ([link](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists))
- [李宏毅深度学习(2016)—哔哩哔哩](https://www.bilibili.com/video/av9770190/?from=search&seid=17240241049019116161)
- [李宏毅深度学习(2017)—哔哩哔哩](https://www.bilibili.com/video/av9770302/?from=search&seid=9981051227372686627)
- **Tutorial for Generative Adversarial Network (GAN)**([slideshare](https://www.slideshare.net/tw_dsconf/ss-78795326),[pdf](http://speech.ee.ntu.edu.tw/~tlkagk/slide/Tutorial_HYLee_GAN.pdf),[ppt](http://speech.ee.ntu.edu.tw/~tlkagk/slide/Tutorial_HYLee_GAN.pptx))
- **Tutorial for General Deep Learning Technology**([slideshare](http://www.slideshare.net/tw_dsconf/ss-62245351),[pdf](http://speech.ee.ntu.edu.tw/~tlkagk/slide/Tutorial_HYLee_Deep.pdf),[ppt](http://speech.ee.ntu.edu.tw/~tlkagk/slide/Tutorial_HYLee_Deep.pptx))

**网易**<br />[![](https://camo.githubusercontent.com/f44b5ac541b5f36064d483e5d97bf7a8f9070ecb/68747470733a2f2f63646e2e6e6c61726b2e636f6d2f79757175652f302f323032302f706e672f3231363931342f313538343432353633383437302d34653536643638642d393964632d343136312d383437662d3666353931303237363636302e706e6723616c69676e3d6c65667426646973706c61793d696e6c696e65266865696768743d323439266f726967696e4865696768743d323530266f726967696e57696474683d3435302673697a653d30267374617475733d646f6e65267374796c653d6e6f6e652677696474683d343439#align=left&display=inline&height=250&originHeight=250&originWidth=450&status=done&style=none&width=450)](https://study.163.com/course/introduction/1003842018.htm)<br />[Hinton机器学习与神经网络中文课](https://study.163.com/course/introduction/1003842018.htm)<br />AI研习社<br />多伦多大学教授 Geoffrey Hinton，众所周知的神经网络发明者，亲自为你讲解机器学习与神经网络相关课程。[查看详情](https://study.163.com/course/introduction/1003842018.htm)

[![](https://camo.githubusercontent.com/24037d321427d11dff63e86942e438f08621e000/68747470733a2f2f63646e2e6e6c61726b2e636f6d2f79757175652f302f323032302f706e672f3231363931342f313538343432353633383535362d38303335633839302d393131352d346165372d383631342d3134643061303838343030362e706e6723616c69676e3d6c65667426646973706c61793d696e6c696e65266865696768743d323439266f726967696e4865696768743d323530266f726967696e57696474683d3435302673697a653d30267374617475733d646f6e65267374796c653d6e6f6e652677696474683d343439#align=left&display=inline&height=250&originHeight=250&originWidth=450&status=done&style=none&width=450)](https://study.163.com/course/introduction/1004336028.htm)<br />[牛津大学xDeepMind 自然语言处理](https://study.163.com/course/introduction/1004336028.htm)<br />大数据文摘<br />由牛津大学人工智能实验室，与创造了 AlphaGo 传奇的谷歌 DeepMind 部门合作的课程，主要讲述利用深度学习实现自然语言处理（NLP...[查看详情](https://study.163.com/course/introduction/1004336028.htm)<br />[![](https://camo.githubusercontent.com/705d2c9b7ded47c9a48c498f7cc058854f106e18/68747470733a2f2f63646e2e6e6c61726b2e636f6d2f79757175652f302f323032302f706e672f3231363931342f313538343432353633383439312d36303131613762352d373565632d346338622d626534362d6433383138663762393463652e706e6723616c69676e3d6c65667426646973706c61793d696e6c696e65266865696768743d323439266f726967696e4865696768743d323530266f726967696e57696474683d3435302673697a653d30267374617475733d646f6e65267374796c653d6e6f6e652677696474683d343439#align=left&display=inline&height=250&originHeight=250&originWidth=450&status=done&style=none&width=450)](https://study.163.com/course/introduction/1004938039.htm)<br />[MIT6.S094深度学习与自动驾驶](https://study.163.com/course/introduction/1004938039.htm)<br />大数据文摘<br />由麻省理工大学（MIT）推出的自动驾驶课程 6.S094 ，主要讲述自动驾驶技术，提供在线项目的实践环境，可直接修改官方网站代码，...[查看详情](https://study.163.com/course/introduction/1004938039.htm)
### 3.2.3 学习资源
#### 书PDF
[《Dive Into DeepLearning》动手学深度学习](http://zh.d2l.ai/)    [**Pytorch版**](http://tangshusen.me/Dive-into-DL-PyTorch/#/)      [**Tensorflow2.0版**](https://trickygo.github.io/Dive-into-DL-TensorFlow2.0/#/)<br />麻省理工学院出版社《[Deep Learning](http://www.deeplearningbook.org/)》
> 中文版：[exacity/deeplearningbook-chinese](https://github.com/exacity/deeplearningbook-chinese)

《[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)》
> 中文版：[https://tigerneil.gitbooks.io/neural-networks-and-deep-learning-zh/content/](https://tigerneil.gitbooks.io/neural-networks-and-deep-learning-zh/content/)

#### 卷积神经网络CNN

- [能否对卷积神经网络工作原理做一个直观的解释？](https://www.zhihu.com/question/39022858)
- [CNN 入门讲解专栏阅读顺序以及论文研读视频集合](https://zhuanlan.zhihu.com/p/33855959)
- [CNN系列模型发展简述（附github代码——已全部跑通）](https://zhuanlan.zhihu.com/p/66215918)
- [【论文解读+代码实战】CNN深度卷积神经网络-AlexNet](https://zhuanlan.zhihu.com/p/107660669)
- [【论文解读+代码实战】CNN深度卷积神经网络-VGG](https://zhuanlan.zhihu.com/p/107884876)
- [【论文解读+代码实战】CNN深度卷积神经网络-Network in Network](https://zhuanlan.zhihu.com/p/108235295)
- [【论文解读+代码实战】CNN深度卷积神经网络-GoogLeNet](https://zhuanlan.zhihu.com/p/108414921)
- [【论文解读+代码实战】CNN深度卷积神经网络-ResNet](https://zhuanlan.zhihu.com/p/108708768)
- [【论文解读+代码实战】CNN深度卷积神经网络-DenseNet](https://zhuanlan.zhihu.com/p/109269085)
### 3.2.4  开源工具
#### 深度学习框架

- [**Tensorflow**](https://tensorflow.google.cn/)
- [**Pytorch**](https://tensorflow.google.cn/)
- [**PaddlePaddle**](https://www.paddlepaddle.org.cn/)
- [**Keras**](https://keras.io/)
- [**Mxnet**](http://mxnet.incubator.apache.org/)
- [**Caffe**](http://caffe.berkeleyvision.org/)
- [**Darknet**](https://pjreddie.com/darknet/)

**Tensorflow入门**

- [Tensorflow官方Tutorials](https://tensorflow.google.cn/tutorials)
- [动手学深度学习-Tensorflow2.0版](https://trickygo.github.io/Dive-into-DL-TensorFlow2.0/#/)
- [在线pdf:《简单粗暴 TensorFlow 2》](https://tf.wiki/)
- [【github】TensorFlow-Course](https://github.com/machinelearningmindset/TensorFlow-Course)
- [【github】TensorFlow2.0-Examples](https://github.com/YunYang1994/TensorFlow2.0-Examples)
- [【github】eat_tensorflow2_in_30_days](https://github.com/lyhue1991/eat_tensorflow2_in_30_days)

**Pytorch入门**
- [Pytorch官方Tutorials](https://pytorch.org/tutorials/)
- [动手学深度学习-Pytorch版](http://tangshusen.me/Dive-into-DL-PyTorch/#/)
- [《pytorch handbook》—【github标星11.6k】](https://github.com/zergtant/pytorch-handbook)    

#### 支撑工具

- [Cuda下载——GPU通用计算框架](https://developer.nvidia.com/cuda-toolkit-archive)
- [Cudnn下载——GPU加速库](https://developer.nvidia.com/rdp/cudnn-download)
- [Nvidia Driver下载——Nvidia显卡驱动](https://www.nvidia.cn/Download/index.aspx?lang=cn#)
- [Nvidia TensorRT下载——Nvidia高性能深度学习推理加深SDK](https://developer.nvidia.com/tensorrt)
- [Anaconda——虚拟编程环境管理](https://www.anaconda.com/)

**标注软件**
- [【github】CasiaLabeler——支持实例分割车道线检测多边形标注等](https://github.com/wkentaro/labelme)
- [【github】labelme——python开发的多边形标注工具](https://github.com/wkentaro/labelme)

**模型可视化**
- [NN-SVG——在线神经网络模型画图工具](http://alexlenail.me/NN-SVG/index.html)
- [Netron——开源神经网络模型画图工具](https://github.com/lutzroeder/netron)
- [PlotNeuralNet——开源神经网络绘图工具](https://github.com/HarisIqbal88/PlotNeuralNet)

**性能优化和部署**
- [【github】torch2trt——易于使用的PyTorch到TensorRT转换器](https://github.com/NVIDIA-AI-IOT/torch2trt)
- [【github】ncnn——腾讯出品的针对移动平台优化的高性能神经网络推理框架](https://github.com/Tencent/ncnn)
- [【github】onnx——跨框架机器学习互操作性的开放标准](https://github.com/onnx/onnx)
- [【github】tensorrt——一个C ++库，用于在NVIDIA GPU和深度学习加速器上进行高性能推理。](https://github.com/NVIDIA/TensorRT)
#### 其他资源

- [FFmpeg——有关视频、图片处理的一切](http://ffmpeg.org/)
- [Spleeter——用深度学习分离音乐中的各个音轨，伴奏提取](https://github.com/deezer/spleeter)
- [GAN人脸生成——用StyleGAN换脸](https://github.com/a312863063/generators-with-stylegan2)
- [faceswap——GAN视频换脸](https://github.com/deepfakes/faceswap)
- [DeepFaceLab——基于faceswap的换脸软件](https://github.com/iperov/DeepFaceLab)

---

## 3.3 计算机视觉
### 3.3.1 入门概念
### 3.3.2 公开课
**网易**<br />[![](https://camo.githubusercontent.com/7194aa9572bff7302a413e967ddf54c0c6c6dcdd/68747470733a2f2f63646e2e6e6c61726b2e636f6d2f79757175652f302f323032302f706e672f3231363931342f313538343432353633383432302d62393930366337612d306461322d346332662d616263322d3762323537343930393033332e706e6723616c69676e3d6c65667426646973706c61793d696e6c696e65266865696768743d323439266f726967696e4865696768743d323530266f726967696e57696474683d3435302673697a653d30267374617475733d646f6e65267374796c653d6e6f6e652677696474683d343439#align=left&display=inline&height=250&originHeight=250&originWidth=450&status=done&style=none&width=450)](https://study.163.com/course/introduction/1003223001.htm)<br />[CS231n计算机视觉课程](https://study.163.com/course/introduction/1003223001.htm)<br />大数据文摘<br />谷歌 AI 中国的负责人、斯坦福大学副教授李飞飞（Fei-Fei L）亲授的 CS231n 课程，每年选课量都爆满的斯坦福王牌课程，主要讲述...[查看详情](https://study.163.com/course/introduction/1003223001.htm)
### 3.3.3 学习资源
**理论**

- OpenCV官网 [https://opencv.org/](https://opencv.org/)
- 学习网站 [https://www.learnopencv.com/](https://www.learnopencv.com/)

**代码实战**

- [【github】OpenCV官方Demo](https://github.com/opencv/opencv/tree/master/samples/cpp)
- [【CV实战】OpenCV—Hello world代码示例](https://zhuanlan.zhihu.com/p/58028543)
- [【CV实战】Ubuntu18.04源码编译安装opencv-3.4.X+测试demo](https://zhuanlan.zhihu.com/p/93356275)
- [【github】「画像処理100本ノック」中文版本！为图像处理初学者设计的 100 个问题](https://github.com/gzr2017/ImageProcessing100Wen))
---

# 4.公开数据集
## 4.1 Pytorch提供
[**torchvision.datasets**](https://pytorch.org/docs/master/torchvision/datasets.html#)

- [MNIST](https://pytorch.org/docs/master/torchvision/datasets.html#mnist)
- [Fashion-MNIST](https://pytorch.org/docs/master/torchvision/datasets.html#fashion-mnist)
- [KMNIST](https://pytorch.org/docs/master/torchvision/datasets.html#kmnist)
- [EMNIST](https://pytorch.org/docs/master/torchvision/datasets.html#emnist)
- [QMNIST](https://pytorch.org/docs/master/torchvision/datasets.html#qmnist)
- [FakeData](https://pytorch.org/docs/master/torchvision/datasets.html#fakedata)
- [COCO](https://pytorch.org/docs/master/torchvision/datasets.html#coco)
- [LSUN](https://pytorch.org/docs/master/torchvision/datasets.html#lsun)
- [ImageFolder](https://pytorch.org/docs/master/torchvision/datasets.html#imagefolder)
- [DatasetFolder](https://pytorch.org/docs/master/torchvision/datasets.html#datasetfolder)
- [ImageNet](https://pytorch.org/docs/master/torchvision/datasets.html#imagenet)
- [CIFAR](https://pytorch.org/docs/master/torchvision/datasets.html#cifar)
- [STL10](https://pytorch.org/docs/master/torchvision/datasets.html#stl10)
- [SVHN](https://pytorch.org/docs/master/torchvision/datasets.html#svhn)
- [PhotoTour](https://pytorch.org/docs/master/torchvision/datasets.html#phototour)
- [SBU](https://pytorch.org/docs/master/torchvision/datasets.html#sbu)
- [Flickr](https://pytorch.org/docs/master/torchvision/datasets.html#flickr)
- [VOC](https://pytorch.org/docs/master/torchvision/datasets.html#voc)
- [Cityscapes](https://pytorch.org/docs/master/torchvision/datasets.html#cityscapes)
- [SBD](https://pytorch.org/docs/master/torchvision/datasets.html#sbd)
- [USPS](https://pytorch.org/docs/master/torchvision/datasets.html#usps)
- [Kinetics-400](https://pytorch.org/docs/master/torchvision/datasets.html#kinetics-400)
- [HMDB51](https://pytorch.org/docs/master/torchvision/datasets.html#hmdb51)
- [UCF101](https://pytorch.org/docs/master/torchvision/datasets.html#ucf101)
- <br />

[**torchaudio.datasets**](https://pytorch.org/audio/datasets.html#)

- [COMMONVOICE](https://pytorch.org/audio/datasets.html#commonvoice)
- [LIBRISPEECH](https://pytorch.org/audio/datasets.html#librispeech)
- [VCTK](https://pytorch.org/audio/datasets.html#vctk)
- [YESNO](https://pytorch.org/audio/datasets.html#yesno)

[**torchtext.datasets**](https://pytorch.org/text/datasets.html#)

- [Language Modeling](https://pytorch.org/text/datasets.html#language-modeling)
- [Sentiment Analysis](https://pytorch.org/text/datasets.html#sentiment-analysis)
- [Text Classification](https://pytorch.org/text/datasets.html#text-classification)
- [Question Classification](https://pytorch.org/text/datasets.html#question-classification)
- [Entailment](https://pytorch.org/text/datasets.html#entailment)
- [Language Modeling](https://pytorch.org/text/datasets.html#id1)
- [Machine Translation](https://pytorch.org/text/datasets.html#machine-translation)
- [Sequence Tagging](https://pytorch.org/text/datasets.html#sequence-tagging)
- [Question Answering](https://pytorch.org/text/datasets.html#question-answering)
- [Unsupervised Learning](https://pytorch.org/text/datasets.html#unsupervised-learning)
## 4.2 Tensorflow提供

- **Audio**
  - [groove](https://tensorflow.google.cn/datasets/catalog/groove)
  - [librispeech](https://tensorflow.google.cn/datasets/catalog/librispeech)
  - [libritts](https://tensorflow.google.cn/datasets/catalog/libritts)
  - [ljspeech](https://tensorflow.google.cn/datasets/catalog/ljspeech)
  - [nsynth](https://tensorflow.google.cn/datasets/catalog/nsynth)
  - [savee](https://tensorflow.google.cn/datasets/catalog/savee)
  - [speech_commands](https://tensorflow.google.cn/datasets/catalog/speech_commands)
- **Image**
  - [abstract_reasoning](https://tensorflow.google.cn/datasets/catalog/abstract_reasoning)
  - [aflw2k3d](https://tensorflow.google.cn/datasets/catalog/aflw2k3d)
  - [arc](https://tensorflow.google.cn/datasets/catalog/arc)
  - [beans](https://tensorflow.google.cn/datasets/catalog/beans)
  - [bigearthnet](https://tensorflow.google.cn/datasets/catalog/bigearthnet)
  - [binarized_mnist](https://tensorflow.google.cn/datasets/catalog/binarized_mnist)
  - [binary_alpha_digits](https://tensorflow.google.cn/datasets/catalog/binary_alpha_digits)
  - [caltech101](https://tensorflow.google.cn/datasets/catalog/caltech101)
  - [caltech_birds2010](https://tensorflow.google.cn/datasets/catalog/caltech_birds2010)
  - [caltech_birds2011](https://tensorflow.google.cn/datasets/catalog/caltech_birds2011)
  - [cars196](https://tensorflow.google.cn/datasets/catalog/cars196)
  - [cassava](https://tensorflow.google.cn/datasets/catalog/cassava)
  - [cats_vs_dogs](https://tensorflow.google.cn/datasets/catalog/cats_vs_dogs)
  - [celeb_a](https://tensorflow.google.cn/datasets/catalog/celeb_a)
  - [celeb_a_hq](https://tensorflow.google.cn/datasets/catalog/celeb_a_hq)
  - [cifar10](https://tensorflow.google.cn/datasets/catalog/cifar10)
  - [cifar100](https://tensorflow.google.cn/datasets/catalog/cifar100)
  - [cifar10_1](https://tensorflow.google.cn/datasets/catalog/cifar10_1)
  - [cifar10_corrupted](https://tensorflow.google.cn/datasets/catalog/cifar10_corrupted)
  - [citrus_leaves](https://tensorflow.google.cn/datasets/catalog/citrus_leaves)
  - [cityscapes](https://tensorflow.google.cn/datasets/catalog/cityscapes)
  - [clevr](https://tensorflow.google.cn/datasets/catalog/clevr)
  - [cmaterdb](https://tensorflow.google.cn/datasets/catalog/cmaterdb)
  - [coil100](https://tensorflow.google.cn/datasets/catalog/coil100)
  - [colorectal_histology](https://tensorflow.google.cn/datasets/catalog/colorectal_histology)
  - [colorectal_histology_large](https://tensorflow.google.cn/datasets/catalog/colorectal_histology_large)
  - [curated_breast_imaging_ddsm](https://tensorflow.google.cn/datasets/catalog/curated_breast_imaging_ddsm)
  - [cycle_gan](https://tensorflow.google.cn/datasets/catalog/cycle_gan)
  - [deep_weeds](https://tensorflow.google.cn/datasets/catalog/deep_weeds)
  - [diabetic_retinopathy_detection](https://tensorflow.google.cn/datasets/catalog/diabetic_retinopathy_detection)
  - [div2k](https://tensorflow.google.cn/datasets/catalog/div2k)
  - [dmlab](https://tensorflow.google.cn/datasets/catalog/dmlab)
  - [downsampled_imagenet](https://tensorflow.google.cn/datasets/catalog/downsampled_imagenet)
  - [dsprites](https://tensorflow.google.cn/datasets/catalog/dsprites)
  - [dtd](https://tensorflow.google.cn/datasets/catalog/dtd)
  - [duke_ultrasound](https://tensorflow.google.cn/datasets/catalog/duke_ultrasound)
  - [emnist](https://tensorflow.google.cn/datasets/catalog/emnist)
  - [eurosat](https://tensorflow.google.cn/datasets/catalog/eurosat)
  - [fashion_mnist](https://tensorflow.google.cn/datasets/catalog/fashion_mnist)
  - [flic](https://tensorflow.google.cn/datasets/catalog/flic)
  - [food101](https://tensorflow.google.cn/datasets/catalog/food101)
  - [geirhos_conflict_stimuli](https://tensorflow.google.cn/datasets/catalog/geirhos_conflict_stimuli)
  - [horses_or_humans](https://tensorflow.google.cn/datasets/catalog/horses_or_humans)
  - [i_naturalist2017](https://tensorflow.google.cn/datasets/catalog/i_naturalist2017)
  - [image_label_folder](https://tensorflow.google.cn/datasets/catalog/image_label_folder)
  - [imagenet2012](https://tensorflow.google.cn/datasets/catalog/imagenet2012)
  - [imagenet2012_corrupted](https://tensorflow.google.cn/datasets/catalog/imagenet2012_corrupted)
  - [imagenet_resized](https://tensorflow.google.cn/datasets/catalog/imagenet_resized)
  - [imagenette](https://tensorflow.google.cn/datasets/catalog/imagenette)
  - [imagewang](https://tensorflow.google.cn/datasets/catalog/imagewang)
  - [kmnist](https://tensorflow.google.cn/datasets/catalog/kmnist)
  - [lfw](https://tensorflow.google.cn/datasets/catalog/lfw)
  - [lost_and_found](https://tensorflow.google.cn/datasets/catalog/lost_and_found)
  - [lsun](https://tensorflow.google.cn/datasets/catalog/lsun)
  - [malaria](https://tensorflow.google.cn/datasets/catalog/malaria)
  - [mnist](https://tensorflow.google.cn/datasets/catalog/mnist)
  - [mnist_corrupted](https://tensorflow.google.cn/datasets/catalog/mnist_corrupted)
  - [omniglot](https://tensorflow.google.cn/datasets/catalog/omniglot)
  - [oxford_flowers102](https://tensorflow.google.cn/datasets/catalog/oxford_flowers102)
  - [oxford_iiit_pet](https://tensorflow.google.cn/datasets/catalog/oxford_iiit_pet)
  - [patch_camelyon](https://tensorflow.google.cn/datasets/catalog/patch_camelyon)
  - [pet_finder](https://tensorflow.google.cn/datasets/catalog/pet_finder)
  - [places365_small](https://tensorflow.google.cn/datasets/catalog/places365_small)
  - [plant_leaves](https://tensorflow.google.cn/datasets/catalog/plant_leaves)
  - [plant_village](https://tensorflow.google.cn/datasets/catalog/plant_village)
  - [plantae_k](https://tensorflow.google.cn/datasets/catalog/plantae_k)
  - [quickdraw_bitmap](https://tensorflow.google.cn/datasets/catalog/quickdraw_bitmap)
  - [resisc45](https://tensorflow.google.cn/datasets/catalog/resisc45)
  - [rock_paper_scissors](https://tensorflow.google.cn/datasets/catalog/rock_paper_scissors)
  - [scene_parse150](https://tensorflow.google.cn/datasets/catalog/scene_parse150)
  - [shapes3d](https://tensorflow.google.cn/datasets/catalog/shapes3d)
  - [smallnorb](https://tensorflow.google.cn/datasets/catalog/smallnorb)
  - [so2sat](https://tensorflow.google.cn/datasets/catalog/so2sat)
  - [stanford_dogs](https://tensorflow.google.cn/datasets/catalog/stanford_dogs)
  - [stanford_online_products](https://tensorflow.google.cn/datasets/catalog/stanford_online_products)
  - [sun397](https://tensorflow.google.cn/datasets/catalog/sun397)
  - [svhn_cropped](https://tensorflow.google.cn/datasets/catalog/svhn_cropped)
  - [tf_flowers](https://tensorflow.google.cn/datasets/catalog/tf_flowers)
  - [the300w_lp](https://tensorflow.google.cn/datasets/catalog/the300w_lp)
  - [uc_merced](https://tensorflow.google.cn/datasets/catalog/uc_merced)
  - [vgg_face2](https://tensorflow.google.cn/datasets/catalog/vgg_face2)
  - [visual_domain_decathlon](https://tensorflow.google.cn/datasets/catalog/visual_domain_decathlon)
- **Object_detection**
  - [coco](https://tensorflow.google.cn/datasets/catalog/coco)
  - [kitti](https://tensorflow.google.cn/datasets/catalog/kitti)
  - [open_images_v4](https://tensorflow.google.cn/datasets/catalog/open_images_v4)
  - [voc](https://tensorflow.google.cn/datasets/catalog/voc)
  - [wider_face](https://tensorflow.google.cn/datasets/catalog/wider_face)
- **Structured**
  - [amazon_us_reviews](https://tensorflow.google.cn/datasets/catalog/amazon_us_reviews)
  - [forest_fires](https://tensorflow.google.cn/datasets/catalog/forest_fires)
  - [german_credit_numeric](https://tensorflow.google.cn/datasets/catalog/german_credit_numeric)
  - [higgs](https://tensorflow.google.cn/datasets/catalog/higgs)
  - [iris](https://tensorflow.google.cn/datasets/catalog/iris)
  - [rock_you](https://tensorflow.google.cn/datasets/catalog/rock_you)
  - [titanic](https://tensorflow.google.cn/datasets/catalog/titanic)
- **Summarization**
  - [aeslc](https://tensorflow.google.cn/datasets/catalog/aeslc)
  - [big_patent](https://tensorflow.google.cn/datasets/catalog/big_patent)
  - [billsum](https://tensorflow.google.cn/datasets/catalog/billsum)
  - [cnn_dailymail](https://tensorflow.google.cn/datasets/catalog/cnn_dailymail)
  - [gigaword](https://tensorflow.google.cn/datasets/catalog/gigaword)
  - [multi_news](https://tensorflow.google.cn/datasets/catalog/multi_news)
  - [newsroom](https://tensorflow.google.cn/datasets/catalog/newsroom)
  - [opinosis](https://tensorflow.google.cn/datasets/catalog/opinosis)
  - [reddit_tifu](https://tensorflow.google.cn/datasets/catalog/reddit_tifu)
  - [scientific_papers](https://tensorflow.google.cn/datasets/catalog/scientific_papers)
  - [wikihow](https://tensorflow.google.cn/datasets/catalog/wikihow)
  - [xsum](https://tensorflow.google.cn/datasets/catalog/xsum)
- **Text**
  - [c4](https://tensorflow.google.cn/datasets/catalog/c4)
  - [cfq](https://tensorflow.google.cn/datasets/catalog/cfq)
  - [civil_comments](https://tensorflow.google.cn/datasets/catalog/civil_comments)
  - [cos_e](https://tensorflow.google.cn/datasets/catalog/cos_e)
  - [definite_pronoun_resolution](https://tensorflow.google.cn/datasets/catalog/definite_pronoun_resolution)
  - [eraser_multi_rc](https://tensorflow.google.cn/datasets/catalog/eraser_multi_rc)
  - [esnli](https://tensorflow.google.cn/datasets/catalog/esnli)
  - [gap](https://tensorflow.google.cn/datasets/catalog/gap)
  - [glue](https://tensorflow.google.cn/datasets/catalog/glue)
  - [imdb_reviews](https://tensorflow.google.cn/datasets/catalog/imdb_reviews)
  - [librispeech_lm](https://tensorflow.google.cn/datasets/catalog/librispeech_lm)
  - [lm1b](https://tensorflow.google.cn/datasets/catalog/lm1b)
  - [math_dataset](https://tensorflow.google.cn/datasets/catalog/math_dataset)
  - [movie_rationales](https://tensorflow.google.cn/datasets/catalog/movie_rationales)
  - [multi_nli](https://tensorflow.google.cn/datasets/catalog/multi_nli)
  - [multi_nli_mismatch](https://tensorflow.google.cn/datasets/catalog/multi_nli_mismatch)
  - [natural_questions](https://tensorflow.google.cn/datasets/catalog/natural_questions)
  - [qa4mre](https://tensorflow.google.cn/datasets/catalog/qa4mre)
  - [scan](https://tensorflow.google.cn/datasets/catalog/scan)
  - [scicite](https://tensorflow.google.cn/datasets/catalog/scicite)
  - [snli](https://tensorflow.google.cn/datasets/catalog/snli)
  - [squad](https://tensorflow.google.cn/datasets/catalog/squad)
  - [super_glue](https://tensorflow.google.cn/datasets/catalog/super_glue)
  - [tiny_shakespeare](https://tensorflow.google.cn/datasets/catalog/tiny_shakespeare)
  - [trivia_qa](https://tensorflow.google.cn/datasets/catalog/trivia_qa)
  - [wikipedia](https://tensorflow.google.cn/datasets/catalog/wikipedia)
  - [xnli](https://tensorflow.google.cn/datasets/catalog/xnli)
  - [yelp_polarity_reviews](https://tensorflow.google.cn/datasets/catalog/yelp_polarity_reviews)
- **Translate**
  - [flores](https://tensorflow.google.cn/datasets/catalog/flores)
  - [para_crawl](https://tensorflow.google.cn/datasets/catalog/para_crawl)
  - [ted_hrlr_translate](https://tensorflow.google.cn/datasets/catalog/ted_hrlr_translate)
  - [ted_multi_translate](https://tensorflow.google.cn/datasets/catalog/ted_multi_translate)
  - [wmt14_translate](https://tensorflow.google.cn/datasets/catalog/wmt14_translate)
  - [wmt15_translate](https://tensorflow.google.cn/datasets/catalog/wmt15_translate)
  - [wmt16_translate](https://tensorflow.google.cn/datasets/catalog/wmt16_translate)
  - [wmt17_translate](https://tensorflow.google.cn/datasets/catalog/wmt17_translate)
  - [wmt18_translate](https://tensorflow.google.cn/datasets/catalog/wmt18_translate)
  - [wmt19_translate](https://tensorflow.google.cn/datasets/catalog/wmt19_translate)
  - [wmt_t2t_translate](https://tensorflow.google.cn/datasets/catalog/wmt_t2t_translate)
- **Video**
  - [bair_robot_pushing_small](https://tensorflow.google.cn/datasets/catalog/bair_robot_pushing_small)
  - [moving_mnist](https://tensorflow.google.cn/datasets/catalog/moving_mnist)
  - [robonet](https://tensorflow.google.cn/datasets/catalog/robonet)
  - [starcraft_video](https://tensorflow.google.cn/datasets/catalog/starcraft_video)
  - [ucf101](https://tensorflow.google.cn/datasets/catalog/ucf101)
