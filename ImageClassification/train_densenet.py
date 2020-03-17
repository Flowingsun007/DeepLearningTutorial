import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from network import DenseNet

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


class DataLoader():
    def __init__(self):
        # Keras dataset加载
        initial_data = tf.keras.datasets.cifar10
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = initial_data.load_data()
        self.train_images = self.train_images.astype(np.float32)/255.0
        self.test_images = self.test_images.astype(np.float32)/255.0
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        #need to resize images to input shape
        resized_images = tf.image.resize_with_pad(self.train_images[index],64,64,)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        #need to resize images to input shape
        resized_images = tf.image.resize_with_pad(self.test_images[index],64,64,)
        return resized_images.numpy(), self.test_labels[index]


def train_densenet(batch_size, epoch):
    dataLoader = DataLoader()
    # build callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint('{epoch}_epoch_densenet_weight.h5',
        save_weights_only=True,
        verbose=1,
        save_freq='epoch')
    # build model
    net = DenseNet.build_densenet()
    net.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6),loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    # 详细参数见官方文档：https://tensorflow.google.cn/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?hl=en
    data_generate = ImageDataGenerator(
        featurewise_center=False,# 将输入数据的均值设置为0
        samplewise_center=False, # 将每个样本的均值设置为0
        featurewise_std_normalization=False,  # 将输入除以数据标准差，逐特征进行
        samplewise_std_normalization=False,   # 将每个输出除以其标准差
        zca_epsilon=1e-6,        # ZCA白化的epsilon值，默认为1e-6
        zca_whitening=False,     # 是否应用ZCA白化
        rotation_range=10,        # 随机旋转的度数范围，输入为整数
        width_shift_range=0.1,   # 左右平移，输入为浮点数，大于1时输出为像素值
        height_shift_range=0.1,  # 上下平移，输入为浮点数，大于1时输出为像素值
        shear_range=0.,          # 剪切强度，输入为浮点数
        zoom_range=0.1,          # 随机缩放，输入为浮点数
        channel_shift_range=0.,  # 随机通道转换范围，输入为浮点数
        fill_mode='nearest',     # 输入边界以外点的填充方式，还有constant,reflect,wrap三种填充方式
        cval=0.,                 # 用于填充的值，当fill_mode='constant'时生效
        horizontal_flip=True,    # 随机水平翻转
        vertical_flip=False,     # 随机垂直翻转
        rescale=None,            # 重缩放因子，为None或0时不进行缩放
        preprocessing_function=None,  # 应用于每个输入的函数
        data_format='channels_last',   # 图像数据格式，默认为channels_last
        validation_split=0.0
      )
    # 引用自：https://www.jianshu.com/p/1576da1abd71

    train_images,train_labels = dataLoader.get_batch_train(60000)
    net.fit(
        data_generate.flow(train_images, train_labels, 
            batch_size=batch_size, 
            shuffle=True, 
            #save_to_dir='resource/images'
        ), 
        steps_per_epoch=len(train_images) // batch_size,
        epochs=epoch,
        callbacks=[checkpoint],
        shuffle=True)


def test_densenet(model_path, batch_size):
    dataLoader = DataLoader()
    test_images, test_labels = dataLoader.get_batch_test(batch_size)
    net = DenseNet.build_densenet()
    net.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    net.build((1,64,64,3))
    net.load_weights(model_path)
    net.evaluate(test_images, test_labels, verbose=2)


if __name__ == '__main__':
    # 训练
    train_densenet(256, 60)
    # 测试
    # test_densenet("./weight/60_epoch_densenet_weight.h5", 10000) #  lr 0.001  weight_decay 1e-6 84.40（无数据增强：76.10）  