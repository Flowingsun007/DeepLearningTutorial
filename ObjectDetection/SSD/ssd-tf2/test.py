import tensorflow as tf
import cv2

from configuration import OBJECT_CLASSES, save_model_dir, test_picture_path
from core.inference import InferenceProcedure
from core.ssd import SSD
from utils.tools import preprocess_image


def find_class_name(class_id):
    for k, v in OBJECT_CLASSES .items():
        if v == class_id:
            return k


# shape of boxes : (N, 4)  (xmin, ymin, xmax, ymax)
# shape of scores : (N,)
# shape of classes : (N,)
def draw_boxes_on_image(image, boxes, scores, classes):
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = str(find_class_name(classes[i])) + ": " + str(scores[i].numpy())
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(255, 0, 0), thickness=2)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 255), thickness=2)
    return image


def test_single_picture(picture_dir, model):
    image_tensor = preprocess_image(picture_dir)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    procedure = InferenceProcedure(model=model)
    is_object_exist, boxes, scores, classes = procedure.get_final_boxes(image=image_tensor)
    if is_object_exist:
        image_with_boxes = draw_boxes_on_image(cv2.imread(picture_dir), boxes, scores, classes)
    else:
        print("No objects were detected.")
        image_with_boxes = cv2.imread(picture_dir)
    return image_with_boxes


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    ssd_model = SSD()
    ssd_model.load_weights(filepath=save_model_dir+'epoch-0')

    image = test_single_picture(picture_dir=test_picture_path, model=ssd_model)
    cv2.imwrite('dataset/result.png', image)
    # cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
    # cv2.imshow("detect result", image)
    # cv2.waitKey(0)
