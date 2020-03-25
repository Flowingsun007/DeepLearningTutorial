import tensorflow as tf
from yolo.bounding_box import bounding_box_predict
from configuration import CATEGORY_NUM, SCALE_SIZE
from utils.nms import NMS
from utils.resize_image import ResizeWithPad


class Inference():
    def __init__(self, yolo_output, input_image_shape):
        super(Inference, self).__init__()
        self.yolo_output = yolo_output
        self.input_image_h = input_image_shape[0]
        self.input_image_w = input_image_shape[1]

    def __yolo_post_processing(self, feature, scale_type):
        box_xy, box_wh, confidence, class_prob = bounding_box_predict(feature_map=feature,
                                                                      scale_type=scale_type,
                                                                      is_training=False)
        boxes = self.__boxes_to_original_image(box_xy, box_wh)
        boxes = tf.reshape(boxes, shape=(-1, 4))
        box_scores = confidence * class_prob
        box_scores = tf.reshape(box_scores, shape=(-1, CATEGORY_NUM))
        return boxes, box_scores

    def __boxes_to_original_image(self, box_xy, box_wh):
        x = tf.expand_dims(box_xy[..., 0], axis=-1)
        y = tf.expand_dims(box_xy[..., 1], axis=-1)
        w = tf.expand_dims(box_wh[..., 0], axis=-1)
        h = tf.expand_dims(box_wh[..., 1], axis=-1)
        x, y, w, h = ResizeWithPad(h=self.input_image_h, w=self.input_image_w).resized_to_raw(center_x=x, center_y=y, width=w, height=h)
        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2
        boxes = tf.concat(values=[xmin, ymin, xmax, ymax], axis=-1)
        return boxes

    def get_final_boxes(self):
        boxes_list = []
        box_scores_list = []
        for i in range(len(SCALE_SIZE)):
            boxes, box_scores = self.__yolo_post_processing(feature=self.yolo_output[i],
                                                            scale_type=i)
            boxes_list.append(boxes)
            box_scores_list.append(box_scores)
        boxes_array = tf.concat(boxes_list, axis=0)
        box_scores_array = tf.concat(box_scores_list, axis=0)
        return NMS().nms(boxes=boxes_array, box_scores=box_scores_array)

