import tensorflow as tf

from detection.core.bbox import geometry, transforms
from detection.utils.misc import trim_zeros

class AnchorTarget(object):
    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        '''Compute regression and classification targets for anchors.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RPN.
            target_stds: [4]. Bounding box refinement standard deviation for RPN.
            num_rpn_deltas: int. Maximal number of Anchors per image to feed to rpn heads.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

    def build_targets(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image 
                coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.

        Returns
        ---
            rpn_labels: [batch_size, num_anchors] 
                Matches between anchors and GT boxes. 1 - positive samples; 0 - negative samples; -1 - neglect
            rpn_label_weights: [batch_size, num_anchors] 
            rpn_delta_targets: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))] 
                Anchor bbox deltas.
            rpn_delta_weights: [batch_size, num_anchors, 4]
        '''
        rpn_labels = []
        rpn_label_weights = []
        rpn_delta_targets = []
        rpn_delta_weights = []
        
        num_imgs = gt_class_ids.shape[0]
        for i in range(num_imgs):
            labels, label_weights, delta_targets, delta_weights = self._build_single_target(
                anchors, valid_flags[i], gt_boxes[i], gt_class_ids[i])
            rpn_labels.append(labels)
            rpn_label_weights.append(label_weights)
            rpn_delta_targets.append(delta_targets)
            rpn_delta_weights.append(delta_weights)
        
        rpn_labels = tf.stack(rpn_labels)
        rpn_label_weights = tf.stack(rpn_label_weights)
        rpn_delta_targets = tf.stack(rpn_delta_targets)
        rpn_delta_weights = tf.stack(rpn_delta_weights)
        
        return rpn_labels, rpn_label_weights, rpn_delta_targets, rpn_delta_weights

    def _build_single_target(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''Compute targets per instance.
        
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)]
            valid_flags: [num_anchors]
            gt_class_ids: [num_gt_boxes]
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
        
        Returns
        ---
            labels: [num_anchors]
            label_weights: [num_anchors]
            delta_targets: [num_anchors, (dy, dx, log(dh), log(dw))] 
            delta_weights: [num_anchors, 4]
        '''
        gt_boxes, _ = trim_zeros(gt_boxes)
        
        labels = -tf.ones(anchors.shape[0], dtype=tf.int32)
        
        # Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = geometry.compute_overlaps(anchors, gt_boxes)

        # Match anchors to GT Boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).
        
        
        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them.
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        anchor_iou_max = tf.reduce_max(overlaps, axis=[1])
        
        labels = tf.where(anchor_iou_max < self.neg_iou_thr, 
                          tf.zeros(anchors.shape[0], dtype=tf.int32), labels)

        # Filter invalid anchors
        labels = tf.where(tf.equal(valid_flags, 1), 
                          labels, -tf.ones(anchors.shape[0], dtype=tf.int32))

        # 2. Set anchors with high overlap as positive.
        labels = tf.where(anchor_iou_max >= self.pos_iou_thr, 
                          tf.ones(anchors.shape[0], dtype=tf.int32), labels)

        # 3. Set an anchor for each GT box (regardless of IoU value).        
        gt_iou_argmax = tf.argmax(overlaps, axis=0)
        labels = tf.tensor_scatter_nd_update(labels, 
                                             tf.reshape(gt_iou_argmax, (-1, 1)), 
                                             tf.ones(gt_iou_argmax.shape, dtype=tf.int32))
        
        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = tf.where(tf.equal(labels, 1))
        extra = ids.shape.as_list()[0] - int(self.num_rpn_deltas * self.positive_fraction)
        if extra > 0:
            # Reset the extra ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            labels = tf.tensor_scatter_nd_update(labels, 
                                                 ids, 
                                                 -tf.ones(ids.shape[0], dtype=tf.int32))
        # Same for negative proposals
        ids = tf.where(tf.equal(labels, 0))
        extra = ids.shape.as_list()[0] - (self.num_rpn_deltas -
            tf.reduce_sum(tf.cast(tf.equal(labels, 1), tf.int32)))
        if extra > 0:
            # Rest the extra ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            labels = tf.tensor_scatter_nd_update(labels, 
                                                 ids, 
                                                 -tf.ones(ids.shape[0], dtype=tf.int32))
        
        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.

        # Compute box deltas
        gt = tf.gather(gt_boxes, anchor_iou_argmax)
        delta_targets = transforms.bbox2delta(anchors, gt, self.target_means, self.target_stds)

        # Compute weights
        label_weights = tf.zeros((anchors.shape[0],), dtype=tf.float32)
        delta_weights = tf.zeros((anchors.shape[0],), dtype=tf.float32)
        num_bfg = tf.where(tf.greater_equal(labels, 0)).shape[0]
        if num_bfg > 0:
            label_weights = tf.where(labels >= 0, 
                                     tf.ones(label_weights.shape, dtype=tf.float32) / num_bfg, 
                                     label_weights)

            delta_weights = tf.where(labels > 0, 
                                     tf.ones(delta_weights.shape, dtype=tf.float32) / num_bfg, 
                                     delta_weights)
        delta_weights = tf.tile(tf.reshape(delta_weights, (-1, 1)), [1, 4])
        
        return labels, label_weights, delta_targets, delta_weights