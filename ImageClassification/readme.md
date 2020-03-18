# 【图像分类实战】经典CNN+分类数据集模型训练合集-tf2.0

# 介绍
本文件夹主要包括两方面内容：

- **1.利用Tensorflow2.0搭建各种经典CNN网络模型——存放于network文件夹**
- **2.利用搭建好的模型进行模型训练——各种trainxx.py(数据集使用MNIST,FASHION-MNIST,CIFAR-10)**

第1部分侧重各种经典CNN网络结构的展示和搭建，通常是使用tf.keras.sequential完成；第2部分的训练代码中，包含：Tensorflow2.0的基本用法，数据加载、模型定义(subclass;functional model;tf.keras.sequential)、模型训练(fit;自定义epoch训练)等、数据增强、可视化(matplotlib；tensorboard)等。
# 模型和数据集关系
**MNIST:**

- LeNet                            构建模型(subclass;functional api;sequential) + 模型/权重的保存和加载
- AlexNet                        数据集处理(tf.data.Dataset.from_tensor_slices) + 可视化(pyplot)

**FASHION-MNIST:**

- VGG                               自定义epoch训练(tf.GradientTape()+自定义更新optimizer+loss )
- Network in Network 可视化(tf.keras.callbacks.TensorBoard)
- GoogLeNet                  Checkpoint的使用(tf.keras.callbacks.ModelCheckpoint)

**CIFAR-10:**

- ResNet                         数据扩增(tf.keras.preprocessing.image.ImageDataGenerator)
- DenseNet                    
- MobileNetV3
# 文件夹结构
├── dataset        数据集文件夹<br />│   ├── CIFAR-10<br />│   │   └── data<br />│   ├── FASHION-MNIST<br />│   │   ├── data<br />│   │   ├── test_images<br />│   │   ├── test_label.txt<br />│   │   ├── train_images<br />│   │   └── train_label.txt<br />│   └── MNIST<br />│       ├── 8.png<br />│       ├── data<br />│       ├── test_images<br />│       ├── test_label.txt<br />│       ├── train_images<br />│       └── train_label.txt<br />├── network        网络模型文件夹<br />│   ├── AlexNet.py<br />│   ├── DenseNet.py<br />│   ├── GoogLenet.py<br />│   ├── LeNet.py<br />│   ├── MobileNetV3.py<br />│   ├── NetworkInNetwork.py<br />│   ├── ResNet.py<br />│   └── VGG.py<br />├── paper        论文文件夹                                      <br />├── readme.md<br />├── resource<br />│   ├── logs<br />│   ├── train_alexnet.png<br />│   ├── train_googlenet.png<br />│   └── train_vgg.png<br />├── train_alexnet.py<br />├── train_densenet.py<br />├── train_googlenet.py<br />├── train_lenet.py<br />├── train_mobilenet.py<br />├── train_nin.py<br />├── train_resnet.py<br />├── train_vgg.py<br />└── weight        权重文件夹

---



