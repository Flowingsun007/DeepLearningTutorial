import numpy as np
import math
from configuration import IMAGE_WIDTH, IMAGE_HEIGHT, ASPECT_RATIOS
from utils.tools import x_y_meshgrid


class FeatureMap(object):
    def __init__(self, feature_map_list):
        self.s_min = 0.2
        self.s_max = 0.9
        self.num_feature_maps = 6
        self.feature_maps = feature_map_list

    def get_downsampling_ratio(self, index):
        ratio_h = IMAGE_HEIGHT / self.get_height(index)
        ratio_w = IMAGE_WIDTH / self.get_width(index)
        if ratio_h != ratio_w:
            raise ValueError("The ratio_h must be equal to the ratio_w!")
        return ratio_h

    def get_num_feature_maps(self):
        return self.num_feature_maps

    def get_height(self, index):
        return self.feature_maps[index].shape[1]

    def get_width(self, index):
        return self.feature_maps[index].shape[2]

    def get_box_scale(self, index):
        scale = self.s_min + (self.s_max - self.s_min) * index / (self.num_feature_maps - 1)
        return scale


class DefaultBoxes(object):
    def __init__(self, feature_map_list):
        self.image_width = IMAGE_WIDTH
        self.image_height = IMAGE_HEIGHT
        self.aspect_ratios = ASPECT_RATIOS
        self.feature_map = FeatureMap(feature_map_list)
        self.num_feature_maps = self.feature_map.get_num_feature_maps()
        self.offset = 0.5

    def __generate_default_boxes_for_one_feature_map(self, feature_map_index):
        feature_map_size = [self.feature_map.get_height(feature_map_index), self.feature_map.get_width(feature_map_index)]
        s_k = self.feature_map.get_box_scale(feature_map_index) * self.feature_map.get_downsampling_ratio(feature_map_index)
        s_k1 = self.feature_map.get_box_scale((feature_map_index + 1) % self.num_feature_maps) * self.feature_map.get_downsampling_ratio((feature_map_index + 1) % self.num_feature_maps)
        ar = self.aspect_ratios[feature_map_index]
        # The coordinates of the center point of the default box: (center_x, center_y)
        center_x, center_y = x_y_meshgrid(x_row=feature_map_size[1], y_col=feature_map_size[0])
        center_x = (center_x + self.offset) / feature_map_size[1]
        center_y = (center_y + self.offset) / feature_map_size[0]
        w = []
        h = []
        for i in range(len(ar)):
            # for the aspect ratio of 1
            if ar[i] == 1.0:
                s_k_ar1 = math.sqrt(s_k * s_k1)
                w.append(s_k_ar1)
                h.append(s_k_ar1)
            w.append(s_k * math.sqrt(ar[i]))
            h.append(s_k / math.sqrt(ar[i]))
        cx = np.array(center_x, dtype=np.float32).reshape((self.feature_map.get_height(feature_map_index), self.feature_map.get_width(feature_map_index)))
        cy = np.array(center_y, dtype=np.float32).reshape((self.feature_map.get_height(feature_map_index), self.feature_map.get_width(feature_map_index)))
        w = np.array(w, dtype=np.float32) / self.image_width
        h = np.array(h, dtype=np.float32) / self.image_height
        return cx, cy, w, h

    @staticmethod
    def __get_default_boxes_for_single_feature(xy, wh):
        xywh_list = []
        for i in range(xy.shape[0]):
            for j in range(xy.shape[1]):
                for k in range(wh.shape[0]):
                    xywh = np.concatenate((xy[i, j], wh[k]), axis=0)
                    xywh_list.append(xywh)
        default_boxes = np.stack(xywh_list, axis=0)
        return default_boxes

    def generate_default_boxes(self):
        feature_map_boxes = []
        for i in range(self.num_feature_maps):
            cx, cy, w, h = self.__generate_default_boxes_for_one_feature_map(feature_map_index=i)
            # cx, cy: numpy ndarray, shape: (feature_map_height, feature_map_width)
            # w, h: numpy ndarray, shape: (N, ), where N is the number of boxes for this feature map.
            center_xy = np.stack((cx, cy), axis=-1)  # shape: (feature_map_height, feature_map_width, 2)
            wh = np.stack((w, h), axis=-1)  # shape: (N, 2)
            default_boxes = self.__get_default_boxes_for_single_feature(center_xy, wh) # shape: (N * feature_map_height * feature_map_width, 4)
            (center_xy, wh) # shape: (N * feature_map_height * feature_map_width, 4)
            feature_map_boxes.append(default_boxes)
        return np.concatenate(feature_map_boxes, axis=0)