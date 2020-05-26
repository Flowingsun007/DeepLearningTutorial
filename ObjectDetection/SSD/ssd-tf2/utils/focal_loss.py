import tensorflow as tf


def sigmoid_focal_loss(y_true, y_pred, alpha, gamma):
    ce = tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred, from_logits=True)
    pred_prob = tf.math.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = tf.math.pow((1.0 - p_t), gamma)
    return tf.math.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)