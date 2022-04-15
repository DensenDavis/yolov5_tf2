import math
import tensorflow as tf
from data import data_preprocessing as preprocessing
from data import augment
from data import data_utils
from config import Configuration
autotune = tf.data.experimental.AUTOTUNE
cfg = Configuration()


class Dataset():
    def __init__(self):
        self.train_label_files = sorted(tf.io.gfile.glob(cfg.train_labels))
        self.train_image_files = sorted(tf.io.gfile.glob(cfg.train_images))
        self.num_train_imgs = len(self.train_image_files)
        self.num_train_batches = math.ceil(self.num_train_imgs/cfg.train_batch_size)
        self.val_label_files = sorted(tf.io.gfile.glob(cfg.val_labels))
        self.val_image_files = sorted(tf.io.gfile.glob(cfg.val_images))
        self.num_val_images = len(self.val_image_files)
        self.train_ds = self.get_train_data()
        self.val_ds = self.get_val_data()

    def decode_io_stream(self, image, label):
        # label : string Tensor
        image = preprocessing.decode_images(image)
        labels = preprocessing.decode_labels(label)
        return image, labels

    def get_train_data(self):
        images = tf.data.Dataset.from_tensor_slices(self.train_image_files)
        labels = tf.data.Dataset.from_tensor_slices(self.train_label_files)
        images = images.map(tf.io.read_file,autotune)
        labels = labels.map(tf.io.read_file,autotune)
        images = images.cache()
        labels = labels.cache()
        ds = tf.data.Dataset.zip((images, labels))
        ds = ds.shuffle(buffer_size=200, reshuffle_each_iteration=True)
        ds = ds.map(self.decode_io_stream,autotune)
        ds = ds.map(preprocessing.resize_and_pad,autotune)
        ds = ds.map(augment.apply_augmentations, autotune)
        ds = ds.padded_batch(cfg.train_batch_size,padded_shapes=([None,None,3],[1000,5]), drop_remainder=True)
        ds = ds.map(preprocessing.transform_labels, autotune)
        ds = ds.prefetch(buffer_size=autotune)
        return ds

    def get_val_data(self):
        images = tf.data.Dataset.from_tensor_slices(self.val_image_files)
        labels = tf.data.Dataset.from_tensor_slices(self.val_label_files)
        images = images.map(tf.io.read_file,autotune)
        labels = labels.map(tf.io.read_file,autotune)
        images = images.cache()
        labels = labels.cache()
        ds = tf.data.Dataset.zip((images, labels))
        ds = ds.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        ds = ds.map(self.decode_io_stream,autotune)
        ds = ds.map(preprocessing.resize_and_pad,autotune)
        ds = ds.padded_batch(cfg.val_batch_size,padded_shapes=([None,None,3],[1000,5]), drop_remainder=True)
        ds = ds.map(preprocessing.transform_labels, autotune)
        ds = ds.prefetch(buffer_size=autotune)
        return ds

