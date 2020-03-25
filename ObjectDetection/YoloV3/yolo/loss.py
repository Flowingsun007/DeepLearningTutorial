import tensorflow as tf
from configuration import SCALE_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IGNORE_THRESHOLD
from utils.iou import IOUDifferentXY
from yolo.bounding_box import bounding_box_predict
from yolo.anchor import get_coco_anchors


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.scale_num = len(SCALE_SIZE)

    def call(self, y_true, y_pred):
        loss = self.__calculate_loss(y_true=y_true, y_pred=y_pred)
        return loss

    def __generate_grid_shape(self):
        scale_tensor = tf.convert_to_tensor(SCALE_SIZE, dtype=tf.dtypes.float32)
        grid_shape = tf.stack(values=[scale_tensor, scale_tensor], axis=-1)
        return grid_shape

    def __get_scale_size(self, scale):
        return tf.convert_to_tensor([IMAGE_HEIGHT, IMAGE_WIDTH], dtype=tf.dtypes.float32) / get_coco_anchors(scale_type=scale)

    def __binary_crossentropy_keep_dim(self, y_true, y_pred, from_logits):
        x = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=from_logits)
        x = tf.expand_dims(x, axis=-1)
        return x

    def __calculate_loss(self, y_true, y_pred):
        grid_shapes = self.__generate_grid_shape()
        total_loss = 0
        # batch size
        B = y_pred[0].shape[0]
        B_int = tf.convert_to_tensor(B, dtype=tf.dtypes.int32)    # tf.Tensor(4, shape=(), dtype=int32)
        B_float = tf.convert_to_tensor(B, dtype=tf.dtypes.float32)  # tf.Tensor(4.0, shape=(), dtype=float32)
        for i in range(self.scale_num):
            true_object_mask = y_true[i][..., 4:5]
            true_object_mask_bool = tf.cast(true_object_mask, dtype=tf.dtypes.bool)
            true_class_probs = y_true[i][..., 5:]

            pred_xy, pred_wh, grid, pred_features = bounding_box_predict(feature_map=y_pred[i],
                                                                         scale_type=i,
                                                                         is_training=True)
            pred_box = tf.concat(values=[pred_xy, pred_wh], axis=-1)
            true_xy_offset = y_true[i][..., 0:2] * grid_shapes[i] - grid
            true_wh_offset = tf.math.log(y_true[i][..., 2:4] * self.__get_scale_size(scale=i) + 1e-10)
            true_wh_offset = tf.keras.backend.switch(true_object_mask_bool, true_wh_offset, tf.zeros_like(true_wh_offset))

            box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]

            ignore_mask = tf.TensorArray(dtype=tf.dtypes.float32, size=1, dynamic_size=True)

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(y_true[i][b, ..., 0:4], true_object_mask_bool[b, ..., 0])
                true_box = tf.cast(true_box, dtype=tf.dtypes.float32)
                # expand dim for broadcasting
                box_1 = tf.expand_dims(pred_box[b], axis=-2)
                box_2 = tf.expand_dims(true_box, axis=0)
                iou = IOUDifferentXY(box_1=box_1, box_2=box_2).calculate_iou()
                best_iou = tf.keras.backend.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, tf.cast(best_iou < IGNORE_THRESHOLD, dtype=tf.dtypes.float32))
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, *args: b < B_int, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

            xy_loss = true_object_mask * box_loss_scale * self.__binary_crossentropy_keep_dim(true_xy_offset, pred_features[..., 0:2], from_logits=True)
            wh_loss = true_object_mask * box_loss_scale * 0.5 * tf.math.square(true_wh_offset - pred_features[..., 2:4])
            confidence_loss = true_object_mask * self.__binary_crossentropy_keep_dim(true_object_mask, pred_features[..., 4:5], from_logits=True) + (1 - true_object_mask) * self.__binary_crossentropy_keep_dim(true_object_mask, pred_features[..., 4:5], from_logits=True) * ignore_mask
            class_loss = true_object_mask * self.__binary_crossentropy_keep_dim(true_class_probs, pred_features[..., 5:], from_logits=True)

            average_xy_loss = tf.keras.backend.sum(xy_loss) / B_float
            average_wh_loss = tf.keras.backend.sum(wh_loss) / B_float
            average_confidence_loss = tf.keras.backend.sum(confidence_loss) / B_float
            average_class_loss = tf.keras.backend.sum(class_loss) / B_float
            total_loss += average_xy_loss + average_wh_loss + average_confidence_loss + average_class_loss

        return total_loss
