import numpy as np
from configuration import CATEGORY_NUM, SCALE_SIZE, \
    COCO_ANCHORS, ANCHOR_NUM_EACH_SCALE, COCO_ANCHOR_INDEX
from utils import iou


class GenerateLabel():
    def __init__(self, true_boxes, input_shape):
        super(GenerateLabel, self).__init__()
        self.true_boxes = np.array(true_boxes, dtype=np.float32)
        self.input_shape = np.array(input_shape, dtype=np.int32)
        self.anchors = np.array(COCO_ANCHORS, dtype=np.float32)
        self.batch_size = self.true_boxes.shape[0]

    def generate_label(self):
        center_xy = (self.true_boxes[..., 0:2] + self.true_boxes[..., 2:4]) // 2  # shape : [B, N, 2]
        box_wh = self.true_boxes[..., 2:4] - self.true_boxes[..., 0:2]     # shape : [B, N, 2]
        self.true_boxes[..., 0:2] = center_xy / self.input_shape   # Normalization
        self.true_boxes[..., 2:4] = box_wh / self.input_shape   # Normalization
        true_label_1 = np.zeros((self.batch_size, SCALE_SIZE[0], SCALE_SIZE[0], ANCHOR_NUM_EACH_SCALE, CATEGORY_NUM + 5))
        true_label_2 = np.zeros((self.batch_size, SCALE_SIZE[1], SCALE_SIZE[1], ANCHOR_NUM_EACH_SCALE, CATEGORY_NUM + 5))
        true_label_3 = np.zeros((self.batch_size, SCALE_SIZE[2], SCALE_SIZE[2], ANCHOR_NUM_EACH_SCALE, CATEGORY_NUM + 5))
        # true_label : list of 3 arrays of type numpy.ndarray(all elements are 0), which shapes are:
        # (self.batch_size, 13, 13, 3, 5 + C)
        # (self.batch_size, 26, 26, 3, 5 + C)
        # (self.batch_size, 52, 52, 3, 5 + C)
        true_label = [true_label_1, true_label_2, true_label_3]
        # shape : (9, 2) --> (1, 9, 2)
        anchors = np.expand_dims(self.anchors, axis=0)
        # valid_mask filters out the valid boxes.
        valid_mask = box_wh[..., 0] > 0

        for b in range(self.batch_size):
            wh = box_wh[b, valid_mask[b]]
            if len(wh) == 0:
                # For pictures without boxes, iou is not calculated.
                continue
            # shape of wh : [N, 1, 2], N is the actual number of boxes per picture
            wh = np.expand_dims(wh, axis=1)
            # Calculate the iou between the box and the anchor, both center points are (0, 0).
            iou_value = iou.IOUSameXY(anchors=anchors, boxes=wh).calculate_iou()
            # shape of best_anchor : [N]
            best_anchor = np.argmax(iou_value, axis=-1)
            for i, n in enumerate(best_anchor):
                for s in range(ANCHOR_NUM_EACH_SCALE):
                    if n in COCO_ANCHOR_INDEX[s]:
                        x = np.floor(self.true_boxes[b, i, 0] * SCALE_SIZE[s]).astype('int32')
                        y = np.floor(self.true_boxes[b, i, 1] * SCALE_SIZE[s]).astype('int32')
                        anchor_id = COCO_ANCHOR_INDEX[s].index(n)
                        class_id = self.true_boxes[b, i, 4].astype('int32')
                        true_label[s][b, y, x, anchor_id, 0:4] = self.true_boxes[b, i, 0:4]
                        true_label[s][b, y, x, anchor_id, 4] = 1
                        true_label[s][b, y, x, anchor_id, 5 + class_id - 1] = 1

        return true_label
