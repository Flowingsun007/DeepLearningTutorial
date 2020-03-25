import tensorflow as tf
from configuration import IOU_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_BOX_NUM, CATEGORY_NUM


class NMS():
    def __init__(self):
        super(NMS, self).__init__()
        self.max_box_num = MAX_BOX_NUM
        self.num_class = CATEGORY_NUM

    def nms(self, boxes, box_scores):
        mask = box_scores >= CONFIDENCE_THRESHOLD
        box_list = []
        score_list = []
        class_list = []
        for i in range(self.num_class):
            box_of_class = tf.boolean_mask(boxes, mask[:, i])
            score_of_class = tf.boolean_mask(box_scores[:, i], mask[:, i])
            selected_indices = tf.image.non_max_suppression(boxes=box_of_class,
                                                            scores=score_of_class,
                                                            max_output_size=self.max_box_num,
                                                            iou_threshold=IOU_THRESHOLD)
            selected_boxes = tf.gather(box_of_class, selected_indices)
            selected_scores = tf.gather(score_of_class, selected_indices)
            classes = tf.ones_like(selected_scores, dtype=tf.dtypes.int32) * i
            box_list.append(selected_boxes)
            score_list.append(selected_scores)
            class_list.append(classes)
        box_array = tf.concat(values=box_list, axis=0)
        score_array = tf.concat(values=score_list, axis=0)
        class_array = tf.concat(values=class_list, axis=0)

        return box_array, score_array, class_array
