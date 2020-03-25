import tensorflow as tf
from data_process.read_txt import ReadTxt
from configuration import BATCH_SIZE, TXT_DIR
import numpy as np


def get_length_of_dataset(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count


def generate_dataset():
    txt_dataset = tf.data.TextLineDataset(filenames=TXT_DIR)

    train_count = get_length_of_dataset(txt_dataset)
    train_dataset = txt_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, train_count


# Return :
# image_name_list : list, length is N (N is the batch size.)
# boxes_array : numpy.ndarrray, shape is (N, MAX_TRUE_BOX_NUM_PER_IMG, 5)
def parse_dataset_batch(dataset):
    image_name_list = []
    boxes_list = []
    len_of_batch = dataset.shape[0]
    for i in range(len_of_batch):
        image_name, boxes = ReadTxt(line_bytes=dataset[i].numpy()).parse_line()
        image_name_list.append(image_name)
        boxes_list.append(boxes)
    boxes_array = np.array(boxes_list)
    return image_name_list, boxes_array
