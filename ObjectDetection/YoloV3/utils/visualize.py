import cv2
from test_on_single_image import single_image_inference
from configuration import training_results_save_dir


def visualize_training_results(pictures, model, epoch):
    # pictures : List of image directories.
    index = 0
    for picture in pictures:
        index += 1
        result = single_image_inference(image_dir=picture, model=model)
        cv2.imwrite(filename=training_results_save_dir + "epoch-{}-picture-{}.jpg".format(epoch, index), img=result)

