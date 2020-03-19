## 介绍
YoloV3的tensorflow2.0实现，代码实现主要参考：[https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3)。基于以上做了部分改动和注释。其中，功能测试部分，使用YOLOV3官方模型权重即可对图片，视频进行目标检测；自定义训练部分，则根据VOC数据集进行模型训练,loss可视化,验证和测试。
## 功能测试
### 依赖项

- requirements.txt
- yolov3.weights
```shell
pip3 install -r ./resource/requirements.txt
# yolov3.weights是YOLO原作者在COCO数据集上训练的模型权重，测试需要下载并放入weight文件夹下
wget https://pjreddie.com/media/files/yolov3.weights
```
### 测试图片/视频
```
python test.py
```
## ![cc.png](https://cdn.nlark.com/yuque/0/2020/png/216914/1584605638622-5cd13db2-7259-4e67-aeb4-6613ef52ef16.png#align=left&display=inline&height=925&name=cc.png&originHeight=925&originWidth=1351&size=1822161&status=done&style=none&width=1351)
## 自定义训练
**训练以VOC数据集为例，自定义训练其他数据集，数据准备过程类似**<br />****数据准备**<br />下载VOC数据集
```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
在data/dataset下新建VOC文件夹，将下载好的文件解压并按如下目录结构存放
```
VOC 
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
```
构建训练和测试的图片txt:
```shell
python voc_annotation.py --data_path /your/path/to/VOC
```
运行后会在dataset下生成voc_train.txt和voc_test.txt
### 训练
训练前，在core/config.py下配置训练/测试图片txt路径，classes name，训练/测试的batch_size,学习率等参数：<br />![a.png](https://cdn.nlark.com/yuque/0/2020/png/216914/1584602602308-7f1668b2-bb00-4938-aa18-829e490ce90b.png#align=left&display=inline&height=878&name=a.png&originHeight=878&originWidth=1017&size=149383&status=done&style=none&width=1017)<br />**开始训练**
```shell
python train.py
```
### ![b.png](https://cdn.nlark.com/yuque/0/2020/png/216914/1584602823582-e2e10c80-c3a5-4484-b75d-ee2d3a127e7e.png#align=left&display=inline&height=480&name=b.png&originHeight=480&originWidth=843&size=618421&status=done&style=none&width=843)
### 可视化
```shell
tensorboard --logdir ./data
```
[![](https://cdn.nlark.com/yuque/0/2020/png/216914/1584600969598-856a0735-e00d-48f3-9256-02577d22b7ef.png#align=left&display=inline&height=297&originHeight=297&originWidth=1371&size=0&status=done&style=none&width=1371)](https://user-images.githubusercontent.com/30433053/68088727-db5a6b00-fe9c-11e9-91d6-555b1089b450.png)
### 验证
```shell
python evaluate.py
cd data/mAP
python main.py -na
```
### ![mAP.png](https://cdn.nlark.com/yuque/0/2020/png/216914/1584603544557-fbf307be-e9b1-456e-9cbb-66caf36c56e6.png#align=left&display=inline&height=470&name=mAP.png&originHeight=470&originWidth=815&size=49656&status=done&style=none&width=815)
### 测试
```shell
python test.py
```

## 其他
1.支持加载darknet训练好的模型权重，进行训练/测试。如：
```shell
utils.load_weights(model, "./weight/yolov3-voc_10000.weights")
```
2.model.sava_weights()时，只支持tf格式，存.h5会报错