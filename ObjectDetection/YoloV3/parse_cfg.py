from configuration import PASCAL_VOC_DIR, PASCAL_VOC_CLASSES, \
    custom_dataset_classes, custom_dataset_dir, use_dataset, COCO_CLASSES, COCO_DIR


class ParseCfg():

    def get_images_dir(self):
        if use_dataset == "custom":
            return custom_dataset_dir
        elif use_dataset == "pascal_voc":
            return PASCAL_VOC_DIR + "JPEGImages"
        elif use_dataset == "coco":
            return COCO_DIR + "train2017"

    def get_classes(self):
        if use_dataset == "custom":
            return custom_dataset_classes
        elif use_dataset == "pascal_voc":
            return PASCAL_VOC_CLASSES
        elif use_dataset == "coco":
            return COCO_CLASSES


