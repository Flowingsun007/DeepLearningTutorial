import tensorflow as tf
layers = tf.keras.layers
losses = tf.keras.losses

class SmoothL1Loss(layers.Layer):
    def __init__(self, rho=1):
        super(SmoothL1Loss, self).__init__()
        self._rho = rho
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.abs(y_true - y_pred)
        loss = tf.where(loss > self._rho, loss - 0.5 * self._rho, 
                        (0.5 / self._rho) * tf.square(loss))

        if sample_weight is not None:
            loss = tf.multiply(loss, sample_weight)
           
        return loss


class RPNClassLoss(layers.Layer):
    def __init__(self):
        super(RPNClassLoss, self).__init__()
        self.sparse_categorical_crossentropy = \
            losses.SparseCategoricalCrossentropy(from_logits=True,
                                                 reduction=losses.Reduction.NONE)

    def __call__(self, rpn_labels, rpn_class_logits, rpn_label_weights):       
        # Filtering if label == -1
        indices = tf.where(tf.not_equal(rpn_labels, -1))
        rpn_labels = tf.gather_nd(rpn_labels, indices)
        rpn_label_weights = tf.gather_nd(rpn_label_weights, indices)
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        
        # Calculate loss
        loss = self.sparse_categorical_crossentropy(y_true=rpn_labels,
                                                    y_pred=rpn_class_logits,
                                                    sample_weight=rpn_label_weights)
        loss = tf.reduce_sum(loss)
        return loss
    

class RPNBBoxLoss(layers.Layer):
    def __init__(self):
        super(RPNBBoxLoss, self).__init__()
        self.smooth_l1_loss = SmoothL1Loss()
        
    def __call__(self, rpn_delta_targets, rpn_deltas, rpn_delta_weights):
        loss = self.smooth_l1_loss(y_true=rpn_delta_targets, 
                                   y_pred=rpn_deltas, 
                                   sample_weight=rpn_delta_weights)
        loss = tf.reduce_sum(loss)
        return loss



class RCNNClassLoss(layers.Layer):
    def __init__(self):
        super(RCNNClassLoss, self).__init__()
        self.sparse_categorical_crossentropy = \
            losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                 reduction=losses.Reduction.NONE)

    def __call__(self, rcnn_labels, rcnn_class_logits, rcnn_label_weights):
        # Filtering if label == -1
        indices = tf.where(tf.not_equal(rcnn_labels, -1))
        rcnn_labels = tf.gather_nd(rcnn_labels, indices)
        rcnn_label_weights = tf.gather_nd(rcnn_label_weights, indices)
        rcnn_class_logits = tf.gather_nd(rcnn_class_logits, indices)

        # Calculate loss
        loss = self.sparse_categorical_crossentropy(y_true=rcnn_labels,
                                                    y_pred=rcnn_class_logits,
                                                    sample_weight=rcnn_label_weights)
        loss = tf.reduce_sum(loss)
        return loss
    

class RCNNBBoxLoss(layers.Layer):
    def __init__(self):
        super(RCNNBBoxLoss, self).__init__()
        self.smooth_l1_loss = SmoothL1Loss()
        
    def __call__(self, rcnn_delta_targets, rcnn_deltas, rcnn_delta_weights):
        loss = self.smooth_l1_loss(y_true=rcnn_delta_targets, 
                                   y_pred=rcnn_deltas, 
                                   sample_weight=rcnn_delta_weights)
        loss = tf.reduce_sum(loss)
        return loss
