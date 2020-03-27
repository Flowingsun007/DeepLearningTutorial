import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg


NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
OUT_CHANNELS    = 3*(NUM_CLASS + 5)
ANCHORS         = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES         = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH



class DarkNetConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DarkNetConv2D, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding="same")
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x

        
class YOLOTail(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(YOLOTail, self).__init__()
        self.conv1 = DarkNetConv2D(filters=in_channels, kernel_size=(1, 1), strides=1)
        self.conv2 = DarkNetConv2D(filters=2 * in_channels, kernel_size=(3, 3), strides=1)
        self.conv3 = DarkNetConv2D(filters=in_channels, kernel_size=(1, 1), strides=1)
        self.conv4 = DarkNetConv2D(filters=2 * in_channels, kernel_size=(3, 3), strides=1)
        self.conv5 = DarkNetConv2D(filters=in_channels, kernel_size=(1, 1), strides=1)

        self.conv6 = DarkNetConv2D(filters=2 * in_channels, kernel_size=(3, 3), strides=1)
        self.normal_conv = tf.keras.layers.Conv2D(filters=out_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        branch = self.conv5(x, training=training)

        stem = self.conv6(branch, training=training)
        stem = self.normal_conv(stem)
        return stem, branch


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = DarkNetConv2D(filters=filters, kernel_size=(1, 1), strides=1)
        self.conv2 = DarkNetConv2D(filters=filters * 2, kernel_size=(3, 3), strides=1)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        # x = tf.keras.layers.concatenate([x, inputs])
        x = tf.keras.layers.add([x, inputs])
        return x


def make_residual_block(filters, num_blocks,use_head=True):
    x = tf.keras.Sequential()
    if(use_head):
        x.add(DarkNetConv2D(filters=2 * filters, kernel_size=(3, 3), strides=2))
    for _ in range(num_blocks):
        x.add(ResidualBlock(filters=filters))
    return x


class DarkNet53(tf.keras.Model):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv1 = DarkNetConv2D(filters=32, kernel_size=(3, 3), strides=1)
        self.conv2 = DarkNetConv2D(filters=64, kernel_size=(3, 3), strides=1)
        self.block1 = make_residual_block(filters=32, num_blocks=1, use_head=False)
        self.block2 = make_residual_block(filters=64, num_blocks=2)
        self.block3 = make_residual_block(filters=128, num_blocks=8)
        self.block4 = make_residual_block(filters=256, num_blocks=8)
        self.block5 = make_residual_block(filters=512, num_blocks=4)


    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        output_1 = self.block3(x, training=training)
        output_2 = self.block4(output_1, training=training)
        output_3 = self.block5(output_2, training=training)
        # print('output_1.shape,output_2.shape,output_3.shape >>>>>>>>>>> ', output_1.shape,output_2.shape,output_3.shape)
        return output_1, output_2, output_3


class YOLOV3(tf.keras.Model):
    def __init__(self):
        super(YOLOV3, self).__init__()
        self.darknet = DarkNet53()
        self.tail_1 = YOLOTail(in_channels=512, out_channels=OUT_CHANNELS)
        self.upsampling_1 = self._make_upsampling(num_filter=256)
        self.tail_2 = YOLOTail(in_channels=256, out_channels=OUT_CHANNELS)
        self.upsampling_2 = self._make_upsampling(num_filter=128)
        # self.tail_3 = YOLOTail(in_channels=128, out_channels=OUT_CHANNELS)

        self.conv1 = DarkNetConv2D(filters=128, kernel_size=(1, 1), strides=1)
        self.conv2 = DarkNetConv2D(filters=2 * 128, kernel_size=(3, 3), strides=1)
        self.conv3 = DarkNetConv2D(filters=128, kernel_size=(1, 1), strides=1)
        self.conv4 = DarkNetConv2D(filters=2 * 128, kernel_size=(3, 3), strides=1)
        self.conv5 = DarkNetConv2D(filters=128, kernel_size=(1, 1), strides=1)

        self.conv6 = DarkNetConv2D(filters=2 * 128, kernel_size=(3, 3), strides=1)
        self.normal_conv = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")


    def _make_upsampling(self, num_filter):
        layer = tf.keras.Sequential()
        layer.add(DarkNetConv2D(filters=num_filter, kernel_size=(1, 1), strides=1))
        layer.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        return layer


    def call(self, inputs, training=None, mask=None):
        x_1, x_2, conv = self.darknet(inputs, training=training)
        branch_1, conv = self.tail_1(conv, training=training)
        conv = self.upsampling_1(conv, training=training)
        x_2 = tf.keras.layers.concatenate([conv, x_2])
        branch_2, conv = self.tail_2(x_2, training=training)
        conv = self.upsampling_2(conv, training=training)
        conv = tf.keras.layers.concatenate([conv, x_1])



        conv = self.conv1(conv, training=training)
        conv = self.conv2(conv, training=training)
        conv = self.conv3(conv, training=training)
        conv = self.conv4(conv, training=training)
        conv = self.conv5(conv, training=training)

        conv = self.conv6(conv, training=training)
        master = self.normal_conv(conv)

        # master, _ = self.tail_3(conv, training=training)
        # print('master.shape, branch_2.shape, branch_1.shape', master.shape, branch_2.shape, branch_1.shape)
        return [master, branch_2, branch_1]


def build_yolov3():
    """
    conv_tensors = YOLOv3(input_tensor)中
    conv_tensors为YOLOv3的网络输出列表，其内容为3个tensor,分别表示8,16,32倍采样率下的输出。
    [<tf.Tensor 'conv2d_74/Identity:0' shape=(None, 52, 52, 75) dtype=float32>, 
    <tf.Tensor 'conv2d_66/Identity:0' shape=(None, 26, 26, 75) dtype=float32>, 
    <tf.Tensor 'conv2d_58/Identity:0' shape=(None, 13, 13, 75) dtype=float32>]

    输入图像为416×416时，8,16,32倍下采样后的尺寸分别为52,26,13，即为输出tensor的中间维度
    最后一个维度 = 3 × (5 + num_class) 这里类别数num_class在VOC上是20，在COCO上是80，故3×(5+20) = 75
    3标记了3种尺寸的先验框的；5则 = x,y,w,h,边框prob，num_class长度则是判定的所有类别的概率向量。
    """
    net = YOLOV3()
    input_tensor = tf.keras.layers.Input([416, 416, 3])
    output_tensor = net(input_tensor)
    model = tf.keras.Model(input_tensor, output_tensor)
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

    print('label_xywh x,y,w,h >>> pred_xywh  x,y,w,h', label_xywh[0,0,0,0,0:4], pred_xywh[0,0,0,0,0:4])

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
    print('giou_loss, conf_loss, prob_loss >>>>>>>>>>>>>>>> ',giou_loss, conf_loss, prob_loss)
    return giou_loss, conf_loss, prob_loss