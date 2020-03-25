import tensorflow as tf
from configuration import COCO_ANCHORS, COCO_ANCHOR_INDEX


def get_coco_anchors(scale_type):
    index_list = COCO_ANCHOR_INDEX[scale_type]
    return tf.convert_to_tensor(COCO_ANCHORS[index_list[0]: index_list[-1] + 1], dtype=tf.dtypes.float32)