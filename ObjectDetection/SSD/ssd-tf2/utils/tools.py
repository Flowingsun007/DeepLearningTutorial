import tensorflow as tf
import numpy as np

from configuration import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


def x_y_meshgrid(x_row, y_col):
    x = np.arange(0, x_row)
    y = np.arange(0, y_col)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    return X, Y


def preprocess_image(img_path):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.io.decode_image(contents=img_raw, channels=CHANNELS, dtype=tf.dtypes.float32)
    # resize
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return img_tensor


def str_to_int(x):
    return int(float(x))


# If you resize the input image, the coordinates of boxes should also be resized.
def resize_box(h, w, xmin, ymin, xmax, ymax):
    resize_ratio = [IMAGE_HEIGHT / h, IMAGE_WIDTH / w]
    xmin = int(resize_ratio[1] * xmin)
    xmax = int(resize_ratio[1] * xmax)
    ymin = int(resize_ratio[0] * ymin)
    ymax = int(resize_ratio[0] * ymax)
    return xmin, ymin, xmax, ymax