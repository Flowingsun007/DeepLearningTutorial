import tensorflow as tf


class DarkNetConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DarkNetConv2D, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding="same")
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = DarkNetConv2D(filters=filters, kernel_size=(1, 1), strides=1)
        self.conv2 = DarkNetConv2D(filters=filters * 2, kernel_size=(3, 3), strides=1)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = tf.keras.layers.add([x, inputs])
        return x


def make_residual_block(filters, num_blocks):
    x = tf.keras.Sequential()
    x.add(DarkNetConv2D(filters=2 * filters, kernel_size=(3, 3), strides=2))
    for _ in range(num_blocks):
        x.add(ResidualBlock(filters=filters))
    return x


class DarkNet53(tf.keras.Model):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv1 = DarkNetConv2D(filters=32, kernel_size=(3, 3), strides=1)
        self.block1 = make_residual_block(filters=32, num_blocks=1)
        self.block2 = make_residual_block(filters=64, num_blocks=2)
        self.block3 = make_residual_block(filters=128, num_blocks=8)
        self.block4 = make_residual_block(filters=256, num_blocks=8)
        self.block5 = make_residual_block(filters=512, num_blocks=4)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        output_1 = self.block3(x, training=training)
        output_2 = self.block4(output_1, training=training)
        output_3 = self.block5(output_2, training=training)
        # print(output_1.shape, output_2.shape, output_3.shape)
        return output_3, output_2, output_1


class YOLOTail(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(YOLOTail, self).__init__()
        self.conv1 = DarkNetConv2D(filters=in_channels, kernel_size=(1, 1), strides=1)
        self.conv2 = DarkNetConv2D(filters=2 * in_channels, kernel_size=(3, 3), strides=1)
        self.conv3 = DarkNetConv2D(filters=in_channels, kernel_size=(1, 1), strides=1)
        self.conv4 = DarkNetConv2D(filters=2 * in_channels, kernel_size=(3, 3), strides=1)
        self.conv5 = DarkNetConv2D(filters=in_channels, kernel_size=(1, 1), strides=1)

        self.conv6 = DarkNetConv2D(filters=2 * in_channels, kernel_size=(3, 3), strides=1)
        self.normal_conv = tf.keras.layers.Conv2D(filters=out_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        branch = self.conv5(x, training=training)

        stem = self.conv6(branch, training=training)
        stem = self.normal_conv(stem)
        return stem, branch


class YOLOV3(tf.keras.Model):
    def __init__(self, out_channels):
        super(YOLOV3, self).__init__()
        self.darknet = DarkNet53()
        self.tail_1 = YOLOTail(in_channels=512, out_channels=out_channels)
        self.upsampling_1 = self._make_upsampling(num_filter=256)
        self.tail_2 = YOLOTail(in_channels=256, out_channels=out_channels)
        self.upsampling_2 = self._make_upsampling(num_filter=128)
        self.tail_3 = YOLOTail(in_channels=128, out_channels=out_channels)

    def _make_upsampling(self, num_filter):
        layer = tf.keras.Sequential()
        layer.add(DarkNetConv2D(filters=num_filter, kernel_size=(1, 1), strides=1))
        layer.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        return layer

    def call(self, inputs, training=None, mask=None):
        x_1, x_2, x_3 = self.darknet(inputs, training=training)
        stem_1, branch_1 = self.tail_1(x_1, training=training)
        branch_1 = self.upsampling_1(branch_1, training=training)
        x_2 = tf.keras.layers.concatenate([branch_1, x_2])
        stem_2, branch_2 = self.tail_2(x_2, training=training)
        branch_2 = self.upsampling_2(branch_2, training=training)
        x_3 = tf.keras.layers.concatenate([branch_2, x_3])
        stem_3, _ = self.tail_3(x_3, training=training)

        return [stem_1, stem_2, stem_3]

