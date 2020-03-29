import os
import tensorflow as tf
import numpy as np
import visualize

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# load dataset
from detection.datasets import coco, data_generator
img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)


val_dataset = coco.CocoDataSet('./COCO2017/', 'val',
                                 flip_ratio=0,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1024))

# display a sample
img, img_meta, bboxes, labels = val_dataset[0]
rgb_img = np.round(img + img_mean)
visualize.display_instances(rgb_img, bboxes, labels, val_dataset.get_categories())


# load model
from detection.models.detectors import faster_rcnn
model = faster_rcnn.FasterRCNN(
    num_classes=len(val_dataset.get_categories()))

batch_imgs = tf.Variable(np.expand_dims(img, 0))
batch_metas = tf.Variable(np.expand_dims(img_meta, 0))
batch_bboxes = tf.Variable(np.expand_dims(bboxes, 0))
batch_labels = tf.Variable(np.expand_dims(labels, 0))

_ = model((batch_imgs, batch_metas), training=False)
model.load_weights('weights/faster_rcnn.h5')

# Stage 1: Region Proposal Network
# 1.a RPN Targets
anchors, valid_flags = model.rpn_head.generator.generate_pyramid_anchors(batch_metas)

rpn_labels, rpn_label_weights, rpn_delta_targets, rpn_delta_weights = \
    model.rpn_head.anchor_target.build_targets(anchors, valid_flags, batch_bboxes, batch_labels)

positive_anchors = tf.gather(anchors, tf.where(tf.equal(rpn_labels, 1))[:, 1])
negative_anchors = tf.gather(anchors, tf.where(tf.equal(rpn_labels, 0))[:, 1])
neutral_anchors = tf.gather(anchors, tf.where(tf.equal(rpn_labels, -1))[:, 1])
positive_target_deltas = tf.gather_nd(rpn_delta_targets, tf.where(tf.equal(rpn_labels, 1)))


from detection.core.bbox import transforms
    
refined_anchors = transforms.delta2bbox(
    positive_anchors, positive_target_deltas, (0., 0., 0., 0.), (0.1, 0.1, 0.2, 0.2))


print('rpn_labels:\t\t', rpn_labels[0].shape.as_list())
print('rpn_delta_targets:\t', rpn_delta_targets[0].shape.as_list())
print('positive_anchors:\t', positive_anchors.shape.as_list())
print('negative_anchors:\t', negative_anchors.shape.as_list())
print('neutral_anchors:\t', neutral_anchors.shape.as_list())
print('refined_anchors:\t', refined_anchors.shape.as_list())

visualize.draw_boxes(rgb_img, 
                     boxes=positive_anchors.numpy(), 
                     refined_boxes=refined_anchors.numpy())

# 1.b RPN Predictions
training = False
C2, C3, C4, C5 = model.backbone(batch_imgs, 
                                training=training)

P2, P3, P4, P5, P6 = model.neck([C2, C3, C4, C5], 
                                training=training)

rpn_feature_maps = [P2, P3, P4, P5, P6]
rcnn_feature_maps = [P2, P3, P4, P5]

rpn_class_logits, rpn_probs, rpn_deltas = model.rpn_head(
    rpn_feature_maps, training=training)

rpn_probs_tmp = rpn_probs[0, :, 1]
# Show top anchors by score (before refinement)
limit = 100
ix = tf.nn.top_k(rpn_probs_tmp, k=limit).indices[::-1]
visualize.draw_boxes(rgb_img, boxes=tf.gather(anchors, ix).numpy())


# Stage 2: Proposal Classification
proposals = model.rpn_head.get_proposals(
    rpn_probs, rpn_deltas, batch_metas)
rois = proposals

pooled_regions = model.roi_align(
    (rois, rcnn_feature_maps, batch_metas), training=training)

rcnn_class_logits, rcnn_probs, rcnn_deltas = \
    model.bbox_head(pooled_regions, training=training)

detections_list = model.bbox_head.get_bboxes(
    rcnn_probs, rcnn_deltas, rois, batch_metas)

tmp = detections_list[0][:, :4]

visualize.draw_boxes(rgb_img, boxes=tmp.numpy())


# Stage 3: Run model directly
detections_list = model((batch_imgs, batch_metas), training=training)
tmp = detections_list[0][:, :4]
visualize.draw_boxes(rgb_img, boxes=tmp.numpy())


# Stage 4: Test (Detection)
from detection.datasets.utils import get_original_image
ori_img = get_original_image(img, img_meta, img_mean)
proposals = model.simple_test_rpn(img, img_meta)
res = model.simple_test_bboxes(img, img_meta, proposals)
visualize.display_instances(ori_img, res['rois'], res['class_ids'], 
                            val_dataset.get_categories(), scores=res['scores'])