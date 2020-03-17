import tensorflow as tf


class Inception(tf.keras.layers.Layer):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = tf.keras.layers.Conv2D(c1, kernel_size=1, activation='relu', padding='same')
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], kernel_size=1, padding='same', activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], kernel_size=3, padding='same',
                              activation='relu')
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], kernel_size=1, padding='same', activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], kernel_size=5, padding='same',
                              activation='relu')
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same', strides=1)
        self.p4_2 = tf.keras.layers.Conv2D(c4, kernel_size=1, padding='same', activation='relu')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return tf.concat([p1, p2, p3, p4], axis=-1)  # 在通道维上连结输出


def build_googlenet():
    b1 = tf.keras.models.Sequential()
    b1.add(tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu'))
    b1.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))


    b2 = tf.keras.models.Sequential()
    b2.add(tf.keras.layers.Conv2D(64, kernel_size=1, padding='same', activation='relu'))
    b2.add(tf.keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'))
    b2.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

    b3 = tf.keras.models.Sequential()
    b3.add(Inception(64, (96, 128), (16, 32), 32))
    b3.add(Inception(128, (128, 192), (32, 96), 64))
    b3.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

    b4 = tf.keras.models.Sequential()
    b4.add(Inception(192, (96, 208), (16, 48), 64))
    b4.add(Inception(160, (112, 224), (24, 64), 64))
    b4.add(Inception(128, (128, 256), (24, 64), 64))
    b4.add(Inception(112, (144, 288), (32, 64), 64))
    b4.add(Inception(256, (160, 320), (32, 128), 128))
    b4.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

    b5 = tf.keras.models.Sequential()
    b5.add(Inception(256, (160, 320), (32, 128), 128))
    b5.add(Inception(384, (192, 384), (48, 128), 128))
    b5.add(tf.keras.layers.GlobalAvgPool2D())
    net = tf.keras.models.Sequential([b1, b2, b3, b4, b5, tf.keras.layers.Dense(10,activation='softmax')])
    return net
