import csv
import random
import numpy as np
import tensorflow as tf


class NYUDataLoader():
    def __init__(self, batch_size, csv_path, FLAG):
        self.batch_size = batch_size
        self.rgbs = [i[0] for i in self.get_data_path(csv_path)]
        self.depths = [i[1] for i in self.get_data_path(csv_path)]
        self.shape_rgb = (480, 640, 3)
        self.shape_depth = (240, 320, 1)
        self.min_depth = 10
        self.max_depth = 10000
        self.FLAG = FLAG

    def get_dataset_size(self):
        return len(self.rgbs)

    def get_data_path(self, csv_path, DEBUG=False):
        with open(csv_path, 'r') as csv_file:
            csv_data = csv.reader(csv_file)
            nyu_list = []
            for row in csv_data:
                nyu_list.append(row)

        # Test on a smaller dataset
        if DEBUG:
            nyu_list = nyu_list[:10]

        return nyu_list

    def augment(self, img):        
        # Gamma augmentation
        gamma = tf.random.uniform([], 0.9, 1.1)
        aug = img ** gamma

        # Brightness augmentation
        brightness = tf.random.uniform([], 0.75, 1.25)
        aug = aug * brightness

        # Color augmentation
        colors = tf.random.uniform([3], 0.9, 1.1)
        white = tf.ones([tf.shape(img)[0], tf.shape(img)[1]])
        color_img = tf.stack([white * colors[i] for i in range(3)], axis=2)
        aug = aug * color_img
           
        aug = tf.clip_by_value(aug, 0, 1)
        
        return aug

    def preprocess_dataset(self, rgb_path, depth_path):
        rgb_data = tf.io.read_file(rgb_path)
        depth_data = tf.io.read_file(depth_path)

        rgb = tf.image.decode_png(rgb_data)
        rgb = tf.image.convert_image_dtype(rgb, dtype=tf.float32)

        # Preprocess data
        if self.FLAG == 'train':
            depth = tf.image.decode_png(depth_data)
            depth = tf.image.resize(depth, [self.shape_depth[0], self.shape_depth[1]])
            depth = tf.image.convert_image_dtype(depth / 255, dtype=tf.float32)
            depth = tf.clip_by_value(depth * self.max_depth, self.min_depth, self.max_depth)
            depth = self.max_depth / depth

            # Data Augmentation
            # Random flipping
            flip = tf.random.uniform([], 0, 1)
            rgb = tf.cond(flip > 0.5, lambda: tf.image.flip_left_right(rgb), lambda: rgb)
            depth = tf.cond(flip > 0.5, lambda: tf.image.flip_left_right(depth), lambda: depth)

            # Random gamma, brightness, color augmentation
            augs = tf.random.uniform([], 0, 1)
            rgb = tf.cond(augs > 0.5, lambda: self.augment(rgb), lambda: rgb)
        else:
            depth = tf.image.decode_png(depth_data, dtype=tf.uint16)
            depth = tf.image.resize(depth, [self.shape_depth[0], self.shape_depth[1]])
            depth = tf.cast(depth, tf.float32)
            depth = depth / self.min_depth
            depth = self.max_depth / depth
        
        return rgb, depth

    def get_batched_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.rgbs, self.depths))
        self.dataset = self.dataset.shuffle(buffer_size=len(self.rgbs), reshuffle_each_iteration=True)
        # self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(map_func=self.preprocess_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size=self.batch_size)

        return self.dataset
    
