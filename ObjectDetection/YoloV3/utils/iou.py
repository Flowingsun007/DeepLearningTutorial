import numpy as np


# Box_1 and box_2 have different center points, and their last dimension is 4 (x, y, w, h).
class IOUDifferentXY():
    def __init__(self, box_1, box_2):
        super(IOUDifferentXY, self).__init__()
        self.box_1_min, self.box_1_max = self.__get_box_min_and_max(box_1)
        self.box_2_min, self.box_2_max = self.__get_box_min_and_max(box_2)
        self.box_1_area = self.__get_box_area(box_1)
        self.box_2_area = self.__get_box_area(box_2)

    def __get_box_min_and_max(self, box):
        box_xy = box[..., 0:2]
        box_wh = box[..., 2:4]
        box_min = box_xy - box_wh / 2
        box_max = box_xy + box_wh / 2
        return box_min, box_max

    def __get_box_area(self, box):
        return box[..., 2] * box[..., 3]

    def calculate_iou(self):
        intersect_min = np.maximum(self.box_1_min, self.box_2_min)
        intersect_max = np.minimum(self.box_1_max, self.box_2_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = self.box_1_area + self.box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou


# Calculate the IOU between two boxes, both center points are (0, 0).
# The shape of anchors : [1, 9, 2]
# The shape of boxes : [N, 1, 2]
class IOUSameXY():
    def __init__(self, anchors, boxes):
        super(IOUSameXY, self).__init__()
        self.anchor_max = anchors / 2
        self.anchor_min = - self.anchor_max
        self.box_max = boxes / 2
        self.box_min = - self.box_max
        self.anchor_area = anchors[..., 0] * anchors[..., 1]
        self.box_area = boxes[..., 0] * boxes[..., 1]

    def calculate_iou(self):
        intersect_min = np.maximum(self.box_min, self.anchor_min)
        intersect_max = np.minimum(self.box_max, self.anchor_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]    # w * h
        union_area = self.anchor_area + self.box_area - intersect_area
        iou = intersect_area / union_area  # shape : [N, 9]

        return iou


