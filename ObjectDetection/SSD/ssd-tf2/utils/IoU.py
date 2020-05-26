import numpy as np


# The last dimensions of box_1 and box_2 are both 4. (x, y, w, h)
class IOU(object):
    def __init__(self, box_1, box_2):
        self.box_1_min, self.box_1_max = self.__get_box_min_and_max(box_1)
        self.box_2_min, self.box_2_max = self.__get_box_min_and_max(box_2)
        self.box_1_area = self.__get_box_area(box_1)
        self.box_2_area = self.__get_box_area(box_2)

    @staticmethod
    def __get_box_min_and_max(box):
        box_xy = box[..., 0:2]
        box_wh = box[..., 2:4]
        box_min = box_xy - box_wh / 2
        box_max = box_xy + box_wh / 2
        return box_min, box_max

    @staticmethod
    def __get_box_area(box):
        return box[..., 2] * box[..., 3]

    def calculate_iou(self):
        intersect_min = np.maximum(self.box_1_min, self.box_2_min)
        intersect_max = np.minimum(self.box_1_max, self.box_2_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = self.box_1_area + self.box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou