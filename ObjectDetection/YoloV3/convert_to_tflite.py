import tensorflow as tf

from configuration import CATEGORY_NUM, save_model_dir, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, TFLite_model_dir
from yolo.yolo_v3 import YOLOV3

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices(device_type="GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    # load model
    yolo_v3 = YOLOV3(out_channels=3 * (CATEGORY_NUM + 5))
    yolo_v3.load_weights(filepath=save_model_dir+"saved_model")
    yolo_v3._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))

    converter = tf.lite.TFLiteConverter.from_keras_model(yolo_v3)
    tflite_model = converter.convert()
    open(TFLite_model_dir, "wb").write(tflite_model)