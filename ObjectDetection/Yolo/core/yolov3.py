#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:47:10
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg


NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
ANCHORS         = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES         = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output


def darknet53(input_data):
    """YOLOV3网络的分类网络"""
    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)
    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)
    # print('output_1.shape,output_2.shape,output_3.shape >>>>>>>>>>> ', route_1.shape,route_2.shape,input_data.shape) # (None, 52, 52, 256) (None, 26, 26, 512) (None, 13, 13, 1024)
    return route_1, route_2, input_data


def YOLOv3(input_layer, class_num=NUM_CLASS):
    """YOLOV3网络主体"""
    route_1, route_2, conv = darknet53(input_layer)
    # print('route_1, route_2, conv >>>> shape :', route_1.shape, route_2.shape, conv.shape) # (None, 52, 52, 256) (None, 26, 26, 512) (None, 13, 13, 1024)
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))

    conv_branch_1 = convolutional(conv, (3, 3, 512, 1024))
    branch_1 = convolutional(conv_branch_1, (1, 1, 1024, 3*(class_num + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1,  512,  256))
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    conv_branch_2 = convolutional(conv, (3, 3, 256, 512))
    branch_2 = convolutional(conv_branch_2, (1, 1, 512, 3*(class_num + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    conv_master = convolutional(conv, (3, 3, 128, 256))
    master = convolutional(conv_master, (1, 1, 256, 3*(class_num +5)), activate=False, bn=False)
    # print('master.shape, branch_2.shape, branch_1.shape', master.shape, branch_2.shape, branch_1.shape)  # (None, 52, 52, 75) (None, 26, 26, 75) (None, 13, 13, 75)
    return [master, branch_2, branch_1]


def upsample(input_layer):
    """下采样，缩放特征"""
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


def build_yolov3():
    """构建包含三种采样率decode输出的网络
    conv_tensors = YOLOv3(input_tensor)中
    conv_tensors为YOLOv3的网络输出列表，其内容为3个tensor,分别表示8,16,32倍采样率下的输出。
    [<tf.Tensor 'conv2d_74/Identity:0' shape=(None, 52, 52, 75) dtype=float32>, 
    <tf.Tensor 'conv2d_66/Identity:0' shape=(None, 26, 26, 75) dtype=float32>, 
    <tf.Tensor 'conv2d_58/Identity:0' shape=(None, 13, 13, 75) dtype=float32>]

    输入图像为416×416时，8,16,32倍下采样后的尺寸分别为52,26,13，即为输出tensor的中间维度
    最后一个维度 = 3 × (5 + num_class) 这里类别数num_class在VOC上是20，在COCO上是80，故3×(5+20) = 75
    3标记了3种尺寸的先验框的；5则 = x,y,w,h,边框prob，num_class长度则是判定的所有类别的概率向量。
    """
    input_tensor = tf.keras.layers.Input([416, 416, 3])
    output_tensor = YOLOv3(input_tensor)
    model = tf.keras.Model(input_tensor, output_tensor)
    return model


def build_for_test():
    """构建测试和验证的yolo模型"""
    inputs = tf.keras.layers.Input([416, 416, 3])
    feature_maps = YOLOv3(inputs)
    outputs = []
    for i, feature_map in enumerate(feature_maps):
        bbox_tensor = decode(feature_map, i)
        outputs.append(bbox_tensor)
    model = tf.keras.Model(inputs, outputs)
    return model


def decode(conv_output, i=0):
    """
    decode()的作用是给网络输出的目标先验框解编码，生成其原始尺寸图片上预测box的x,y,w,h以及类别置信度和num_classes个分类的概率矩阵
    input:
        conv_output yolo网络在指定下采样倍数下的输出(包含预测框的tensor)
        i           通过i来指定具体的下采样倍数，如SSTRIDES[0]表示8倍下采样；STRIDES[1]是16倍下采样
    return:
        tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
        batch_size和输入保持一致，其大小即预测框的个数
        output_size
    
    """

    conv_shape       = tf.shape(conv_output) # 将conv_output的维度数字转化为维度矩阵输出
    batch_size       = conv_shape[0] # 预测出box框的数量 
    output_size      = conv_shape[1] # 特征图的尺寸：52、26或13

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]   # 预测box和先验box中心坐标的偏移量
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]   # 预测box和先验box长宽偏移量
    conv_raw_conf = conv_output[:, :, :, :, 4:5]   # 预测box类别置信度(分类准确度)
    conv_raw_prob = conv_output[:, :, :, :, 5: ]   # 预测的类别概率(类别矩阵)

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    # 先验框坐标x,y
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)   
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    # 预测框在416×416图片上的实际坐标x,y,w,h
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]  
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    # 预测类别的置信度0~1
    pred_conf = tf.sigmoid(conv_raw_conf)
    #print('decode() pred_conf >>>>>>>>>>>>>>>>>>>>>>>>>',pred_conf)
    # 预测的类别向量(num_class类)
    pred_prob = tf.sigmoid(conv_raw_prob)
    # 拼接所有预测结果tensor
    abc = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    return abc


def bbox_iou(boxes1, boxes2):
    """
    采用IoU方式计算目标框的损失
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    """
    采用GIoU方式计算目标框的损失
    """
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):
    """ 计算yolo中的损失，总损失由三个部分的损失相加而成：giou_loss +　conf_loss　+ prob_loss
    giou_loss为预测框和真实box的iou损失；
    conf_loss为置信度损失(目标框中物体分类概率0~1和真实类别间的损失)
    prob_loss为类别概率损失(分类损失矩阵，当完全命中目标类记为1，其他类别值记为0时，总损失为0）
    输入：
    pred是网络预测的boxes;conv是网络总输出;label真实标记box在指定缩放率下标定真实box的anchor box；bboxes是真实标记box
    输出：
    注意：后两者采用sigmoid交叉熵损失——sigmoid_cross_entropy_with_logits()
         相比较softmax，其可以为一个目标打上多个不互斥的标签，如，一个女人，可以打上girl,person这两个标签
    """
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size  # stride表示下采样缩放倍数，分别 = 8,16,32
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]  # 网络输出的分类置信度prob
    conv_raw_prob = conv[:, :, :, :, 5:]   # 网络输出的分类概率矩阵

    pred_xywh     = pred[:, :, :, :, 0:4]  # (x,y,w,h)
    pred_conf     = pred[:, :, :, :, 4:5]  # 预测的分类置信度

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    # print('x,y,w,h >> anchor_xywh;pred_xywh;truth_xywh', label_xywh[0,0,0,0,0:4], pred_xywh[0,0,0,0,0:4], bboxes[0,0,0:4])

    # 计算giou损失(预测box之间和anchor box之间)
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)
    # 计算conf loss
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )
    # 计算prob loss
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
    # print('giou_loss, conf_loss, prob_loss >>>>>>>>>>>>>>>> ',giou_loss, conf_loss, prob_loss)
    return giou_loss, conf_loss, prob_loss

