from __future__ import absolute_import, division, print_function, unicode_literals
import os
import random
import numpy as np
import cv2
import tensorflow as tf
import core.utils as utils
from core.config import cfg



class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE                # 输入图像尺寸
        self.strides = np.array(cfg.YOLO.STRIDES)                    # FPN采样尺寸：[8, 16, 32]
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)  
        self.num_classes = len(self.classes)                         # 图像类别总数
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS)) # 9种尺寸的先验框anchor box
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE            # 每个网格中anchor的个数，值 = 3
        self.max_bbox_per_scale = 150                                # 每个采样率下许存在的目标框最大数量

        self.annotations = self.load_annotations(dataset_type)             # 真实标记(box)
        self.num_samples = len(self.annotations)                           # 标记总数
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size)) # 每轮迭代总步数  np.ceil返回上进位整数上（53.1 >>> 54）
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        """next产生一批(三种采样率)图像和标签数据"""
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)    # 输入图片尺寸
            self.train_output_sizes = self.train_input_size // self.strides  # 输出图片尺寸 = 输入//下采样缩放倍数

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)
            # 这里s,m,l分别对应三种下采样缩放率8,16,32
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    # 根据给定的真实标记bbox来解析出三种采样率下对应的label和box
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                # 随机打乱annotation
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        """随机水平平移"""
        if random.random() < 0.2:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        """随机剪裁"""
        if random.random() < 0.2:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        """随机旋转"""
        if random.random() < 0.2:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
        """根据path解析图片，并返回所有的bboxs标记目标框
        一个标记框坐标如： 58,107,291,465,2   x,y,h,w,class_id
        分别表示框中心坐标(x,y)，box框的width,height,目标框所属类别索引(如VOC数据集class_id索引为0~19共20类
        """
        line = annotation.split()
        # image_path在每一行开头的位置
        image_path = line[0] 
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
        # 是否采用图片数据增强
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))  # 随机水平移动
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))             # 随机剪裁
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))        # 随机旋转

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):
        """计算两个box之间的iou"""
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):
        """根据给定的真实标记bbox来解析出三种采样率下对应的label和box
           即用先验的anchor box来铆定对应的真实box
        """
        # label[i]的shape——(train_output_sizes × train_output_sizes × anchor_per_scale × (5 + num_classes))
        # 5 + num_classes：x,y,w,h,置信度 + num_classes:分类概率矩阵
        labels = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]

        # bboxes_xywh[i]的shape——max_bbox_per_scale × 4
        # max_bbox_per_scale即该采样率下最多可以包含的真实box数量150；4即x,y,h,w
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)] # [(150,4),(150,4),(150,4)]
        # 三种采样率下bbox的数量，每种采样率下最多包含max_bbox_per_scale个box
        bbox_count = np.zeros((3,))
        # 遍历真实标记的boxes,找到对应box在相应网格中的label（即充当label的anchor box）
        for bbox in bboxes:
            # 获取x_min, y_min, x_max, y_max
            bbox_coor = bbox[:4]
            # 类别id
            bbox_class_ind = bbox[4]
            # 将物体类别转化为one_hot编码
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            # 平滑处理
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            # 计算(x,y,w,h)——(x,y) = ((x_max, y_max) + (x_min, y_min)) * 0.5 ; (w,h) = (x_max, y_max) - (x_min, y_min)
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # 按8,16,32缩放后的(x,y,w,h) shape = (3, 4)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            # 这里exist_positive表示标记的box所落在的网格中，存在和其iou值大于0.3的anchor box,即用此anchor来表示标记box
            exist_positive = False
            for i in range(3):
                # 根据缩放后的bbox_xywh_scaled在3个缩放率下计算iou从而找到用来对应真实box的anchor box
                anchors_xywh = np.zeros((self.anchor_per_scale, 4)) # shape 3×4，存放该下采样缩放率下的三个anchor box
                # 定位anchor的x,y坐标位置(+0.5是神马意思？？？！！！)
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                # 定位anchor的w和h（从文件中读取的anchor的宽和高乘以对应缩放率进行还原）
                anchors_xywh[:, 2:4] = self.anchors[i]
                # 计算标记box和对应网格内三个anchor box的交并比
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                # 找到iou > 0.3的anchor box
                iou_mask = iou_scale > 0.3
                # 处理iou大于0.3的anchor box
                if np.any(iou_mask): # 只要有一个满足，即为true
                    # 构造box对应的label
                    # 向下取整，获取anchor所在格子的在该缩放率下的坐标索引xind和yind
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    #print(']]]]]]]]]]]]]]]]]]]]]]]]xind, yind, best_anchor',xind, yind, iou_mask)
                    labels[i][yind, xind, iou_mask, :] = 0
                    # 填充真实x,y,w,h
                    labels[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    #print('labels[i][yind, xind, iou_mask, 0:4]', labels[i][yind, xind, iou_mask, 0:4])
                    # 填充该label置信度为1
                    labels[i][yind, xind, iou_mask, 4:5] = 1.0
                    # 填充分类概率矩阵为smooth_onehot
                    labels[i][yind, xind, iou_mask, 5:] = smooth_onehot
                    # bbox_ind即真实框在150个bbox下的索引
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    # 给第bbox_ind的bbox赋值x,y,w,h
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    # 该缩放率下真实框的个数+1
                    bbox_count[i] += 1

                    exist_positive = True

            # 三个采样率下的各3个anchor box都没有找到对应的iou>0.3的正例box,则取iou最大的那个来代表
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                labels[best_detect][yind, xind, best_anchor, :] = 0
                labels[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                labels[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                labels[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1


        label_sbbox, label_mbbox, label_lbbox = labels
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        # 返回三个缩放尺度下的label，和true box数据对
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs