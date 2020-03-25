# training
EPOCHS = 60
BATCH_SIZE = 8
load_weights_before_training = False
load_weights_from_epoch = 10

# input image
IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
CHANNELS = 3

# Dataset
CATEGORY_NUM = 20
ANCHOR_NUM_EACH_SCALE = 3
COCO_ANCHORS = [[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]]
COCO_ANCHOR_INDEX = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
SCALE_SIZE = [13, 26, 52] # 对应下采样缩放倍数32,16,8

use_dataset = "pascal_voc"      # "custom", "pascal_voc", "coco"

PASCAL_VOC_DIR = "./dataset/VOCdevkit/VOC2012/"
PASCAL_VOC_ANNOTATION = PASCAL_VOC_DIR + "Annotations"
PASCAL_VOC_IMAGE = PASCAL_VOC_DIR + "JPEGImages"
# The 20 object classes of PASCAL VOC
PASCAL_VOC_CLASSES = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
                      "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
                      "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
                      "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
                      "pottedplant": 18, "sofa": 19, "tvmonitor": 20}

COCO_DIR = "./dataset/COCO/2017/"
COCO_CLASSES = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
                "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
                "fire hydrant": 11, "stop sign": 12, "parking meter": 13, "bench": 14,
                "bird": 15, "cat": 16, "dog": 17, "horse": 18, "sheep": 19, "cow": 20,
                "elephant": 21, "bear": 22, "zebra": 23, "giraffe": 24, "backpack": 25,
                "umbrella": 26, "handbag": 27, "tie": 28, "suitcase": 29, "frisbee": 30,
                "skis": 31, "snowboard": 32, "sports ball": 33, "kite": 34, "baseball bat": 35,
                "baseball glove": 36, "skateboard": 37, "surfboard": 38, "tennis racket": 39,
                "bottle": 40, "wine glass": 41, "cup": 42, "fork": 43, "knife": 44, "spoon": 45,
                "bowl": 46, "banana": 47, "apple": 48, "sandwich": 49, "orange": 50, "broccoli": 51,
                "carrot": 52, "hot dog": 53, "pizza": 54, "donut": 55, "cake": 56, "chair": 57,
                "couch": 58, "potted plant": 59, "bed": 60, "dining table": 61, "toilet": 62,
                "tv": 63, "laptop": 64, "mouse": 65, "remote": 66, "keyboard": 67, "cell phone": 68,
                "microwave": 69, "oven": 70, "toaster": 71, "sink": 72, "refrigerator": 73,
                "book": 74, "clock": 75, "vase": 76, "scissors": 77, "teddy bear": 78,
                "hair drier": 79, "toothbrush": 80}



TXT_DIR = "./data_process/data.txt"

custom_dataset_dir = ""
custom_dataset_classes = {}



# loss
IGNORE_THRESHOLD = 0.5


# NMS
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.5
MAX_BOX_NUM = 50

MAX_TRUE_BOX_NUM_PER_IMG = 20


# save model
save_model_dir = "saved_model/"
save_frequency = 5

# tensorflow lite model
TFLite_model_dir = "yolov3_model.tflite"

test_images_during_training = True
training_results_save_dir = "./test_results_during_training/"
test_images = ["", ""]

test_picture_dir = "./test_data/1.jpg"
test_video_dir = "./test_data/test_video.mp4"
temp_frame_dir = "./test_data/temp.jpg"