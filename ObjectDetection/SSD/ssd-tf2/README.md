# TensorFlow2.0_SSD
A tensorflow_2.0 implementation of SSD (Single Shot MultiBox Detector) .


## Requirements:
+ Python >= 3.6
+ TensorFlow == 2.1.0
+ numpy == 1.17.0
+ opencv-python == 4.1.0.25

## Usage
### Train on PASCAL VOC 2012
1. Download the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
2. Unzip the file and place it in the 'dataset' folder, make sure the directory is like this : 
```
|——dataset
    |——VOCdevkit
        |——VOC2012
            |——Annotations
            |——ImageSets
            |——JPEGImages
            |——SegmentationClass
            |——SegmentationObject
```
3. Run **write_voc_to_txt.py** to generate **voc.txt**.
4. Run **train.py** to start training, before that, you can change the value of the parameters in **configuration.py**.

### Test on single picture
1. Change the *test_picture_dir* in **configuration.py**.
2. Run **test.py** to test on single picture.


## References
+ The paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
+ [focal_loss implemented in TensorFlow_Addons](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/focal_loss.py)

