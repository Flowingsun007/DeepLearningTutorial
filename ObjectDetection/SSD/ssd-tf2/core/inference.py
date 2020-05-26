import tensorflow as tf
import numpy as np
from configuration import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH
from core.anchor import DefaultBoxes
from core.ssd import ssd_prediction
from utils.nms import NMS


class InferenceProcedure(object):
    def __init__(self, model):
        self.model = model
        self.num_classes = NUM_CLASSES + 1
        self.image_size = np.array([IMAGE_HEIGHT, IMAGE_WIDTH], dtype=np.float32)
        self.nms_op = NMS()

    def __get_ssd_prediction(self, image):
        output = self.model(image, training=False)
        pred = ssd_prediction(feature_maps=output, num_classes=self.num_classes)
        return pred, output

    @staticmethod
    def __resize_boxes(boxes, image_height, image_width):
        cx = boxes[..., 0] * image_width
        cy = boxes[..., 1] * image_height
        w = boxes[..., 2] * image_width
        h = boxes[..., 3] * image_height
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        resized_boxes = tf.stack(values=[xmin, ymin, xmax, ymax], axis=-1)
        return resized_boxes

    def __filter_background_boxes(self, ssd_predict_boxes):
        is_object_exist = True
        num_of_total_predict_boxes = ssd_predict_boxes.shape[1]
        # scores = tf.nn.softmax(ssd_predict_boxes[..., :self.num_classes])
        scores = ssd_predict_boxes[..., :self.num_classes]
        classes = tf.math.argmax(input=scores, axis=-1)
        filtered_boxes_list = []
        for i in range(num_of_total_predict_boxes):
            if classes[:, i] != 0:
                filtered_boxes_list.append(ssd_predict_boxes[:, i, :])
        if filtered_boxes_list:
            filtered_boxes = tf.stack(values=filtered_boxes_list, axis=1)
            return is_object_exist, filtered_boxes
        else:
            is_object_exist = False
            return is_object_exist, ssd_predict_boxes

    def __offsets_to_true_coordinates(self, pred_boxes, ssd_output):
        pred_classes = tf.reshape(tensor=pred_boxes[..., :self.num_classes], shape=(-1, self.num_classes))
        pred_coords = tf.reshape(tensor=pred_boxes[..., self.num_classes:], shape=(-1, 4))
        default_boxes = DefaultBoxes(feature_map_list=ssd_output).generate_default_boxes()
        d_cx, d_cy, d_w, d_h = default_boxes[:, 0:1], default_boxes[:, 1:2], default_boxes[:, 2:3], default_boxes[:, 3:4]
        offset_cx, offset_cy, offset_w, offset_h = pred_coords[:, 0:1], pred_coords[:, 1:2], pred_coords[:, 2:3], pred_coords[:, 3:4]
        true_cx = offset_cx * d_w + d_cx
        true_cy = offset_cy * d_h + d_cy
        true_w = tf.math.exp(offset_w) * d_w
        true_h = tf.math.exp(offset_h) * d_h
        true_coords = tf.concat(values=[true_cx, true_cy, true_w, true_h], axis=-1)
        true_classes_and_coords = tf.concat(values=[pred_classes, true_coords], axis=-1)
        true_classes_and_coords = tf.expand_dims(input=true_classes_and_coords, axis=0)
        return true_classes_and_coords

    def get_final_boxes(self, image):
        pred_boxes, ssd_output = self.__get_ssd_prediction(image)
        pred_boxes = self.__offsets_to_true_coordinates(pred_boxes=pred_boxes, ssd_output=ssd_output)
        is_object_exist, filtered_pred_boxes = self.__filter_background_boxes(pred_boxes)
        if is_object_exist:
            # scores = tf.nn.softmax(filtered_pred_boxes[..., :self.num_classes])
            scores = filtered_pred_boxes[..., :self.num_classes]
            pred_boxes_scores = tf.reshape(tensor=scores, shape=(-1, self.num_classes))
            pred_boxes_coord = filtered_pred_boxes[..., self.num_classes:]
            pred_boxes_coord = tf.reshape(tensor=pred_boxes_coord, shape=(-1, 4))
            resized_pred_boxes = self.__resize_boxes(boxes=pred_boxes_coord,
                                                     image_height=image.shape[1],
                                                     image_width=image.shape[2])
            box_tensor, score_tensor, class_tensor = self.nms_op.nms(boxes=resized_pred_boxes,
                                                                     box_scores=pred_boxes_scores)
            return is_object_exist, box_tensor, score_tensor, class_tensor
        else:
            return is_object_exist, tf.zeros(shape=(1, 4)), tf.zeros(shape=(1,)), tf.zeros(shape=(1,))