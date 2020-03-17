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
from network import LeNet


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def read_image(filename, out_dir):
    f = open(filename,'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')

    for i in range(images):
        image = Image.new('L', (columns, rows)) # 'L'为单通道黑白图；'RGB'为三通道彩图，其他模式:https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')
        path = out_dir + '/' + str(i) + '.png'
        image.save(path)
        print('save image:%d >>> %s' % (i, path))


def read_label(filename, out_path):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    labelArr = [0] * labels
    for x in range(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
        save = open(out_path, 'w')
        save.write(','.join(map(lambda x: str(x), labelArr)))
        save.write('\n')
        save.close()
        print('save label:%d success' % x)


def transform_data():
    """MNIST数据预处理，将图片从byte转为png,将labels转为txt"""
    read_image('./dataset/MNIST/data/train-images-idx3-ubyte', './dataset/MNIST/train_images')
    read_label('./dataset/MNIST/data/train-labels-idx1-ubyte', './dataset/MNIST/train_label.txt')
    read_image('./dataset/MNIST/data/t10k-images-idx3-ubyte', './dataset/MNIST/test_images')
    read_label('./dataset/MNIST/data/t10k-labels-idx1-ubyte', './dataset/MNIST/test_label.txt')


def load_and_preprocess_image(path):
    """图像预处理，将输入图片转换成28×28的单通道、归一化的图片矩阵"""
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image /= 255.0
    return image


def data_loader(image_dir, label_path):
    """数据加载"""
    all_image_paths = list(os.listdir(image_dir))
    all_image_paths.sort(key= lambda x:int(x[:-4]))
    all_image_paths = [(image_dir + os.sep + str(path))for path in all_image_paths]    

    label_names = ['0','1','2','3','4','5','6','7','8','9']
    label_index = dict((name, index) for index, name in enumerate(label_names))  # {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

    with open(label_path, 'r', encoding='utf-8') as f:
        all_image_labels = f.read().strip('\n').split(',')
    all_image_labels = [label_index[label] for label in all_image_labels]

    print("\nFirst 10 images path: ", all_image_paths[:10])
    print("\nFirst 10 labels: ", all_image_labels[:10])

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds


def train_mnist(dir_path, label_path, epochs, save_path):
    """模型训练"""
    train_dataset = data_loader(dir_path, label_path)
    train_dataset = train_dataset.batch(60000)
    train_dataset = train_dataset.shuffle(60000)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_images, train_labels = next(iter(train_dataset))
    # 定义模型——三种方式：１．sequential；2.functional API；3.自定义子类Model
    model = LeNet.build_lenet('sequential')
    # 2. functional API  model = LeNet.build_lenet('functional')
    # 3. 自定义子类Model   model = LeNet.build_lenet('subclass')

    # 定义优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # 模型编译
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 训练
    model.fit(train_images, train_labels, epochs=epochs)
    # 模型保存——两种方式：1.只存权重 2、存包含优化器和损失函数在内的整个模型(仅支持functional和sequential类型的model)
    model.save(save_path)
    # 只存权重 model.save_weights(save_path)


def test_mnist(dir_path, label_path, model_path):
    # 方式１：直接加载已经训练好的模型
    # model= keras.models.load_model(model_path)
    # 方式２：重新构造模型，并加载训练好的权重
    model = LeNet.build_lenet('subclass')
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.build((1,28,28,1))
    model.load_weights(model_path)

    # 验证准确率
    test_dataset = data_loader(dir_path, label_path)
    test_dataset = test_dataset.batch(10000)
    test_images, test_labels = next(iter(test_dataset))
    loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    # 显示模型网络结构
    # print(model.summary())


def test_one_image(image_path, model_path):
    # 加载模型
    model= keras.models.load_model(model_path)
    # 加载需要预测的图片
    image = load_and_preprocess_image(image_path)
    # 把数据转换成float32
    image = tf.image.convert_image_dtype(image, tf.float32)
    tensor = tf.reshape(image, [1, 28, 28, 1])
    predictions = model.predict(tensor)
    # 输出分类矩阵和预测结构
    print("predictions:", predictions[0])
    print ('the predict number is : ', np.argmax(predictions[0]))


if __name__ == '__main__':
    # MNIST数据转换(预处理)
    # transform_data()

    # 模型训练
    path = './weight/10_epoch_lenet_weight.h5'
    # train_mnist('./dataset/MNIST/train_images', './dataset/MNIST/train_label.txt', 10, path)

    # 模型测试
    test_mnist('./dataset/MNIST/test_images', './dataset/MNIST/test_label.txt', path)  # acc 99.18

    # 测试单张图片
    test_one_image('./dataset/MNIST/8.png', path)


