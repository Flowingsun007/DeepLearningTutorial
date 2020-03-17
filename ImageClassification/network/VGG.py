import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        # strides默认 = (1, 1)
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk


def build_vggnet(conv_arch):
    net = tf.keras.models.Sequential()
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs,num_channels))
    net.add(tf.keras.models.Sequential([tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(4096,activation='relu'),
             tf.keras.layers.Dropout(0.5),
             tf.keras.layers.Dense(4096,activation='relu'),
             tf.keras.layers.Dropout(0.5),
             tf.keras.layers.Dense(10,activation='softmax')]))
    return net


def build_vgg(keyword = 'vgg11'):
    if keyword == 'vgg11':
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  #VGG11
    if keyword == 'vgg16':
        conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  #VGG16
    if keyword == 'vgg19':
        conv_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))  #VGG16
    net = build_vggnet(conv_arch)
    return net



