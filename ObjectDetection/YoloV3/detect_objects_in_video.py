import tensorflow as tf
import cv2
from configuration import test_video_dir, temp_frame_dir, CATEGORY_NUM, save_model_dir
from test_on_single_image import single_image_inference
from yolo.yolo_v3 import YOLOV3


def frame_detection(frame, model):
    cv2.imwrite(filename=temp_frame_dir, img=frame)
    frame = single_image_inference(image_dir=temp_frame_dir, model=model)
    return frame


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # load model
    yolo_v3 = YOLOV3(out_channels=3 * (CATEGORY_NUM + 5))
    yolo_v3.load_weights(filepath=save_model_dir+"saved_model")

    capture = cv2.VideoCapture(test_video_dir)
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = capture.read()
        if ret:
            new_frame = frame_detection(frame, yolo_v3)
            cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("detect result", new_frame)
            cv2.waitKey(int(1000 / fps))
        else:
            break
    capture.release()
    cv2.destroyAllWindows()