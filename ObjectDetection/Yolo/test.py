import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core import yolov3
from core.yolov3 import YOLOv3, decode
import time
from PIL import Image

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def test_image(image_path, model_path):
    input_size = 416
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    model = yolov3.build_for_test()
    # 加载tf model:model.load_weights(model_path);加载darknet model: utils.load_weights(model, model_path)
    utils.load_weights(model, model_path)
    model.summary()
    start_time = time.time()
    pred_bbox = model.predict(image_data)
    print('pred_bbox>>>>>>>>>>>>>>>>>', pred_bbox)
    end_time = time.time()
    print("time: %.2f ms" %(1000*(end_time-start_time)))

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    # 将416×416下的bbox坐标转换为原图上的坐标并删除部分无效box
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    # 构建原图和bbox画出坐标框
    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    image.show()


def test_video(video_path, model_path):
    input_size      = 416

    model = yolov3.build_for_test()
    # model.load_weights(model_path)
    utils.load_weights(model, model_path)
    model.summary()
    vid = cv2.VideoCapture(video_path)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        prev_time = time.time()
        pred_bbox = model.predict_on_batch(image_data)
        curr_time = time.time()
        exec_time = curr_time - prev_time

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__=='__main__':

    model_path = "./weight/yolov3.weights"
    # model_path = "./weight/60_epoch_yolov3_weights"

    # 测试图片
    test_image("./resource/kite.jpg", model_path)

    # 测试视频
    # test_video("./resource/road.mp4", model_path)


