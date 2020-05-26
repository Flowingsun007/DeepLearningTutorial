import cv2
from test import test_single_picture
from configuration import training_results_save_dir


def visualize_training_results(pictures, model, epoch):
    # pictures : List of image directories.
    index = 0
    for picture in pictures:
        index += 1
        result = test_single_picture(picture_dir=picture, model=model)
        cv2.imwrite(filename=training_results_save_dir + "epoch-{}-picture-{}.jpg".format(epoch, index), img=result)

