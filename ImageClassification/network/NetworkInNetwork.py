import tensorflow as tf


def nin_block(num_channels, kernel_size, strides, padding):
    blk = tf.keras.models.Sequential()
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size,
                                   strides=strides, padding=padding, activation='relu')) 
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=1,activation='relu')) 
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=1,activation='relu'))    
    return blk


def build_networkinnetwork():
    net = tf.keras.models.Sequential()
    net.add(nin_block(96, kernel_size=11, strides=4, padding='valid'))
    net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    net.add(nin_block(256, kernel_size=5, strides=1, padding='same'))
    net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    net.add(nin_block(384, kernel_size=3, strides=1, padding='same'))
    net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    net.add(tf.keras.layers.Dropout(0.5))
    net.add(nin_block(10, kernel_size=3, strides=1, padding='same'))
    net.add(tf.keras.layers.GlobalAveragePooling2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10, activation='softmax'))
    return net