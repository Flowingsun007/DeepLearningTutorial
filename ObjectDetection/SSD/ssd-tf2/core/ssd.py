import tensorflow as tf
from core.models.resnet import ResNet50
from configuration import NUM_CLASSES, ASPECT_RATIOS


class SSD(tf.keras.Model):
    def __init__(self):
        super(SSD, self).__init__()
        self.num_classes = NUM_CLASSES + 1
        self.anchor_ratios = ASPECT_RATIOS

        self.backbone = ResNet50()
        self.conv1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")
        self.conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")
        self.conv2_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same")
        self.conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")
        self.conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")
        self.conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")
        self.conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        self.predict_1 = self._predict_layer(k=self._get_k(i=0))
        self.predict_2 = self._predict_layer(k=self._get_k(i=1))
        self.predict_3 = self._predict_layer(k=self._get_k(i=2))
        self.predict_4 = self._predict_layer(k=self._get_k(i=3))
        self.predict_5 = self._predict_layer(k=self._get_k(i=4))
        self.predict_6 = self._predict_layer(k=self._get_k(i=5))

    def _predict_layer(self, k):
        filter_num = k * (self.num_classes + 4)
        return tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding="same")

    def _get_k(self, i):
        # k is the number of boxes generated at each position of the feature map.
        return len(self.anchor_ratios[i]) + 1

    def call(self, inputs, training=None, mask=None):
        branch_1, x = self.backbone(inputs, training=training)
        predict_1 = self.predict_1(branch_1)

        x = self.conv1(x)
        branch_2 = x
        predict_2 = self.predict_2(branch_2)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        branch_3 = x
        predict_3 = self.predict_3(branch_3)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        branch_4 = x
        predict_4 = self.predict_4(branch_4)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        branch_5 = x
        predict_5 = self.predict_5(branch_5)

        branch_6 = self.pool(x)
        branch_6 = tf.expand_dims(input=branch_6, axis=1)
        branch_6 = tf.expand_dims(input=branch_6, axis=2)
        predict_6 = self.predict_6(branch_6)

        # predict_i shape : (batch_size, h, w, k * (c+4)), where c is self.num_classes.
        return [predict_1, predict_2, predict_3, predict_4, predict_5, predict_6]


def ssd_prediction(feature_maps, num_classes):
    batch_size = feature_maps[0].shape[0]
    predicted_features_list = []
    for feature in feature_maps:
        predicted_features_list.append(tf.reshape(tensor=feature, shape=(batch_size, -1, num_classes + 4)))
    predicted_features = tf.concat(values=predicted_features_list, axis=1)
    return predicted_features
