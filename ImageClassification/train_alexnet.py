# 2020-03-04 Lyon
# MNIST数据集:http://yann.lecun.com/exdb/mnist/
from PIL import Image,ImageOps
import struct
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from network import AlexNet


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


class DataLoader():
    def __init__(self):
        initial_data = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = initial_data.load_data()
        # 将图片转为float32且除255进行归一化；expand_dims增加维度
        self.train_images = np.expand_dims(self.train_images.astype(np.float32)/255.0,axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32)/255.0,axis=-1)
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        # np.random.randint均匀分布，从训练集中随机产生batch_size个索引
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        # 将图片resize至合适大小，具体根据模型、图片大小、自己需求来选择，通常为224。这里为了适配论文中的网络结构，改为227(特例)
        resized_images = tf.image.resize_with_pad(self.train_images[index],227,227,)
        print('resized_images.shape >>>>>>>>>>>>> ', resized_images.shape)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        resized_images = tf.image.resize_with_pad(self.test_images[index],227,227,)
        return resized_images.numpy(), self.test_labels[index]


def visualization(history, epochs):
    """用pyplot将训练和验证结果可视化"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(16, 4)) # 宽×高
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')

    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def train_mnist(epochs, path):
    # 数据加载(由于图片resize到227×227尺寸较大，故这里只加载2万张图片,实际训练集有6万张图片)
    dataloader = DataLoader() 
    train_images, train_labels = dataloader.get_batch_train(20000) 
    # 定义优化器
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.8)
    # 模型构建和编译
    model = AlexNet.build_alexnet()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    history = model.fit(train_images, train_labels, epochs=epochs, shuffle=True, validation_split=0.1)
    model.save_weights(path)
    return history


def test_mnist(path):
    dataloader = DataLoader()
    test_images, test_labels = dataloader.get_batch_test(10000)
    model = AlexNet.build_alexnet()
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.build((1,227,227,1))
    model.load_weights(path)
    model.evaluate(test_images,  test_labels, verbose=2)


if __name__ == '__main__':

    # 模型训练
    path = './weight/10_epoch_alexnet_weight.h5'
    history = train_mnist(10, path)
    # 训练结果可视化
    visualization(history, 10)
    # 模型测试
    test_mnist(path)  # acc 99.14


