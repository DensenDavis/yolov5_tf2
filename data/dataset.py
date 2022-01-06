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
        # self.val_label_files = sorted(tf.io.gfile.glob(cfg.val_labels))
        # self.val_image_files = sorted(tf.io.gfile.glob(cfg.val_images))
        # self.num_val_images = len(self.val_input_files)
        self.train_ds = self.get_train_data()
        # self.val_ds = self.get_val_data()

    def decode_io_stream(self, image, label):
        image = preprocessing.decode_image(image)
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
        ds = ds.map(preprocessing.crop_or_pad_to_fixed_size,autotune)
        ds = ds.map(augment.apply_augmentations, autotune)
        # ds = ds.map(data_utils.zero_pad_bboxes, autotune)
        ds = ds.batch(cfg.train_batch_size, drop_remainder=False)
        ds = ds.map(preprocessing.transform_labels, autotune)
        ds = ds.prefetch(buffer_size=autotune)
        return ds

    # def get_val_data(self):
    #     input_ds = tf.data.Dataset.from_tensor_slices(self.val_input_files)
    #     gt_ds = tf.data.Dataset.from_tensor_slices(self.val_gt_files)
    #     ds = tf.data.Dataset.zip((input_ds, gt_ds))
    #     # Shuffle val dataset only if you want different sample images to be printed at each display frequency
    #     ds = ds.shuffle(buffer_size=20, reshuffle_each_iteration=True)
    #     ds = ds.map(self.read_files, num_parallel_calls=autotune)
    #     # ds = ds.cache()
    #     ds = ds.map(self.create_pair, num_parallel_calls=autotune)
    #     if cfg.val_augmentation:
    #         # If number of validation images is small, augmenting validation set is recommended
    #         ds = ds.map(self.val_augmentation, num_parallel_calls=autotune)
    #         # ds = ds.unbatch()
    #         # ds = ds.shuffle(buffer_size=20, reshuffle_each_iteration=True)
    #         # ds = ds.batch(cfg.val_batch_size)
    #     ds = ds.map(self.split_val_pair, num_parallel_calls=autotune)
    #     ds = ds.batch(cfg.val_batch_size, drop_remainder=False)
    #     ds = ds.prefetch(buffer_size=autotune)
    #     return ds
