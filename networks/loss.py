import tensorflow as tf


def depth_loss(y_true, y_pred, maxDepthVal=1000.0/10.0):
    # Point-wise depth
    l_depth = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), axis=-1)

    # Structure similarity (SSIM)
    l_ssim = tf.clip_by_value((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    return (1.0 * tf.reduce_mean(l_ssim)) + (1.0 * tf.reduce_mean(l_edges)) + (0.1 * tf.reduce_mean(l_depth))