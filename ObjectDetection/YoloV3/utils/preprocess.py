import tensorflow as tf
from configuration import IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS
from parse_cfg import ParseCfg
import os


def resize_image_with_pad(image):
    image_tensor = tf.image.resize_with_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = image_tensor / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor


def process_single_image(image_filename):
    img_raw = tf.io.read_file(image_filename)
    image = tf.io.decode_jpeg(img_raw, channels=CHANNELS)
    image = resize_image_with_pad(image=image)
    image = tf.dtypes.cast(image, dtype=tf.dtypes.float32)
    image = image / 255.0
    return image


def process_image_filenames(filenames):
    image_list = []
    for filename in filenames:
        image_path = os.path.join(ParseCfg().get_images_dir(), filename)
        image_tensor = process_single_image(image_path)
        image_list.append(image_tensor)
    return tf.concat(values=image_list, axis=0)

