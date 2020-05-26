# training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 20
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
CHANNELS = 3

load_weights_before_training = False
load_weights_from_epoch = 0
save_frequency = 5

test_picture_path = "dataset/kite.jpg"

test_images_during_training = False
training_results_save_dir = "dataset/test_pictures/"
test_images_dir_list = ["", ""]

# When the iou value of the anchor and the real box is less than the IoU_threshold,
# the anchor is divided into negative classes, otherwise positive.
IOU_THRESHOLD = 0.6

# generate anchor
ASPECT_RATIOS = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]

# focal loss
alpha = 0.25
gamma = 2.0

reg_loss_weight = 0.5

# dataset
PASCAL_VOC_DIR = "./dataset/VOCdevkit/VOC2012/"
# The 20 object classes of PASCAL VOC
OBJECT_CLASSES = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
                  "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
                  "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
                  "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
                  "pottedplant": 18, "sofa": 19, "tvmonitor": 20}

TXT_DIR = "dataset/voc.txt"


MAX_BOXES_PER_IMAGE = 20

# nms
NMS_IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
MAX_BOX_NUM = 50


# directory of saving model
save_model_dir = "weight/"

