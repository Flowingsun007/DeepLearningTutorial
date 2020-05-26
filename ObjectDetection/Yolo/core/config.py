from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
# 注意：yolov3.weights是在coco数据集上训练出来的，需要和coco.names配套使用；训练和测试VOC训练的模型，则需要voc.names
__C.YOLO.CLASSES              = "./data/classes/voc.names"            # voc.names  coco.names
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"   # 先验框的尺寸，每种采样率下各3个尺寸，共计3×3 = 9种尺寸比例(此基准尺寸需要乘以8,16,32进行还原)
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/voc_train.txt"
__C.TRAIN.BATCH_SIZE          = 8
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 0
__C.TRAIN.EPOCHS              = 60



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/voc_test.txt"
__C.TEST.BATCH_SIZE           = 1
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False   # 图片旋转平移，详见dataset.py
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.3
__C.TEST.IOU_THRESHOLD        = 0.45


