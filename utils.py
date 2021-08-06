import matplotlib.pyplot as plt
import tensorflow as tf


def visualize(tensor):
    plasma = plt.get_cmap('plasma')
    img = tf.reshape(tensor, [3, 240, 320])
    img = tf.clip_by_value(depth_normalize(img), 10, 1000) / 1000
    img = plasma(img)[:, :, :, :3]
    return img


def print_model(model):
    model.build(input_shape=(None, 480, 640, 3))
    model.summary(line_length=110)


def depth_normalize(depth, max_depth=1000):
    return max_depth / depth