import os
import json
import tensorflow as tf
import numpy as np
from pycocotools.cocoeval import COCOeval

from detection.datasets import coco, data_generator
from detection.models.detectors import faster_rcnn

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


# build dataset
img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)
val_dataset = coco.CocoDataSet('./COCO2017/', 'val',
                               flip_ratio=0,
                               pad_mode='fixed',
                               mean=img_mean,
                               std=img_std,
                               scale=(800, 1344))
print('len(val_dataset) >>>>>>>>>>>>>> ', len(val_dataset))

# load faster-rcnn model
model = faster_rcnn.FasterRCNN(num_classes=len(val_dataset.get_categories()))

img, img_meta, bboxes, labels = val_dataset[0]
batch_imgs = tf.Variable(np.expand_dims(img, 0))
batch_metas = tf.Variable(np.expand_dims(img_meta, 0))

_ = model((batch_imgs, batch_metas), training=False)
model.load_weights('weights/faster_rcnn.h5', by_name=True)



# test on the validation dataset
batch_size = 1
dataset_results = []
imgIds = []
for idx in range(len(val_dataset)):
    if idx % 10 == 0:
        print(idx)
    
    img, img_meta, _, _ = val_dataset[idx]

    # generate proposals
    proposals = model.simple_test_rpn(img, img_meta)
    # detect on pictures with proposal
    res = model.simple_test_bboxes(img, img_meta, proposals)
    
    image_id = val_dataset.img_ids[idx]
    imgIds.append(image_id)
    
    for pos in range(res['class_ids'].shape[0]):
        results = dict()
        results['score'] = float(res['scores'][pos])
        results['category_id'] = val_dataset.label2cat[int(res['class_ids'][pos])]
        y1, x1, y2, x2 = [float(num) for num in list(res['rois'][pos])]
        results['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        results['image_id'] = image_id
        dataset_results.append(results)

# write result to json 
with open('coco_val2017_detection_result.json', 'w') as f:
    f.write(json.dumps(dataset_results))
coco_dt = val_dataset.coco.loadRes('coco_val2017_detection_result.json')

# evaluate mAP
cocoEval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()