# 2020-03-04 Lyon
import tensorflow as tf
from tensorflow import keras





class LeNetModel(tf.keras.Model):
    """构建LeNet模型——通过自定义子类化Ｍodel实现(subclassing the Model class)"""
    def __init__(self):
        super(LeNetModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu', input_shape=(28,28,1))
        self.pool = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = keras.layers.Conv2D(filters=16, kernel_size=5, activation = 'relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(120, activation='relu')
        self.dense2 = keras.layers.Dense(84, activation='relu')
        self.dense3 = keras.layers.Dense(10, activation='softmax')
        self.dropout = keras.layers.Dropout(0.25)

    def call(self, inputs, training=False):
        x = self.dense1(self.flatten(self.pool(self.conv2(self.pool(self.conv1(inputs))))))
        if training:
            x = self.dropout(self.dense2(self.dropout(x, training=training)))
        else:
            x = self.dense2(x)
        return self.dense3(x)



def build_suquential_model():
    """构建LeNet模型——通过Sequential()实现"""
    net = keras.models.Sequential([
    # Conv2D为平面卷积层，输入参数28,28,1表示输入为28*28像素的单通道图片，filter=6即表示有6卷积核，kernel_size=5表示单个卷积核尺寸为5*5
    keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu', input_shape=(28,28,1)),
    # MaxPool2D为池化层（最大池化）,池化核尺寸为2*2，步长为2，即保证了使整体输入尺寸缩小一半的效果
    keras.layers.MaxPool2D(pool_size=2, strides=2),
    keras.layers.Conv2D(filters=16, kernel_size=5, activation = 'relu'),
    keras.layers.MaxPool2D(pool_size=2, strides=2),
    # Flatten()即将上一层拍扁成一维数组，方便后面接上全连接层Dense
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(84, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation='softmax')])
    return net


def build_functional_model():
    """构建LeNet模型——functional API实现"""
    inputs = keras.layers.Input([28,28,1])
    conv1 = keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu', input_shape=(28,28,1))(inputs)
    pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(conv1)
    conv2 = keras.layers.Conv2D(filters=16, kernel_size=5, activation = 'relu')(pool1)
    flatten = keras.layers.Flatten()(conv2)
    dense1 = keras.layers.Dense(120, activation='relu')(flatten)
    dropout1 = keras.layers.Dropout(0.25)(dense1)
    dense2 = keras.layers.Dense(84, activation='relu')(dropout1)
    dropout2 = keras.layers.Dropout(0.25)(dense2)
    dense3 = keras.layers.Dense(10, activation=None)(dropout2)
    outputs = tf.nn.softmax(dense3)
    net = keras.Model(inputs=inputs, outputs=outputs)
    return net


def build_lenet(keyword='sequential'):
    if keyword=='sequential':
        return build_suquential_model()
    if keyword=='functional':
        return build_functional_model()
    if keyword=='subclass':
        return LeNetModel()









