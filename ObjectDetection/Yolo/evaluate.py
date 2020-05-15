import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from core import utils, yolov3
from core.config import cfg

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def evaluate(model_path):
    INPUT_SIZE = 416
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

    predicted_dir_path = './data/mAP/predicted'
    ground_truth_dir_path = './data/mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model
    model = yolov3.build_for_test()
    model.load_weights(model_path)
    # 加载利用darknet训练的权重文件
    # utils.load_weights(model, "./weight/yolov3-voc_10000.weights")
    print(model.summary())

    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            # print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            image_size = image.shape[:2]
            image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            # Predict
            pred_bbox = model.predict(image_data)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
            bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

            if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                image = utils.draw_bbox(image, bboxes)
                cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image)

            print('bboxes length >>>>>>>>>>>>>>>>> ', bboxes.__len__())
            with open(predict_result_path, 'w') as f:
                # bbox：xmin,ymin,xmax,ymax,score(分类置信度),class_ind(分类index)
                for bbox in bboxes:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())


if __name__ == '__main__':
    model_path = './weight/60_epoch_yolov3_weights'
    evaluate(model_path)
