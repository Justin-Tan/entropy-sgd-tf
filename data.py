#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import directories

def random_rotation(img, max_rotation=0.1, crop=True):
    with tf.name_scope('RandomRotation'):
        rotation = tf.random_uniform([], -max_rotation, max_rotation)
        rotated_image = tf.contrib.image.rotate(img, rotation, interpolation='BILINEAR')
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l * tf.cos(rotation) - old_s * tf.sin(rotation)) / tf.cos(2*rotation)
            new_s = (old_s - tf.sin(rotation) * new_l) / tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.ceil((h-new_h)/2), tf.int32), tf.cast(tf.ceil((w-new_w)/2), tf.int32)
            rotated_image_crop = rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :]

            # If crop removes the entire image, keep the original image
            rotated_image = tf.cond(tf.equal(tf.size(rotated_image_crop), 0),
                                    true_fn=lambda: img,
                                    false_fn=lambda: rotated_image_crop)

        return rotated_image

def random_padding(image, max_pad_w=5, max_pad_h=10):
    w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
    h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
    paddings = [h_pad, w_pad, [0, 0]]

    return tf.pad(image, paddings, mode='REFLECT', name='random_padding')

def padding_inputs_width(image, target_shape, increment):

    target_ratio = target_shape[1]/target_shape[0]
    # Compute ratio to keep the same ratio in new image and get the size of padding
    # necessary to have the final desired shape
    shape = tf.shape(image)
    ratio = tf.divide(shape[1], shape[0], name='ratio')

    new_h = target_shape[0]
    new_w = tf.cast(tf.round((ratio * new_h) / increment) * increment, tf.int32)
    f1 = lambda: (new_w, ratio)
    f2 = lambda: (new_h, tf.constant(1.0, dtype=tf.float64))
    new_w, ratio = tf.case({tf.greater(new_w, 0): f1,
                            tf.less_equal(new_w, 0): f2},
                            default=f1, exclusive=True)
    target_w = target_shape[1]

    # Definitions for cases
    def pad_fn():
        with tf.name_scope('mirror_padding'):
            pad = tf.subtract(target_w, new_w)

            img_resized = tf.image.resize_images(image, [new_h, new_w])

            # Padding to have the desired width
            paddings = [[0, 0], [0, pad], [0, 0]]
            pad_image = tf.pad(img_resized, paddings, mode='SYMMETRIC', name=None)

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def replicate_fn():
        with tf.name_scope('replication_padding'):
            img_resized = tf.image.resize_images(image, [new_h, new_w])

            # If one symmetry is not enough to have a full width
            # Count number of replications needed
            n_replication = tf.cast(tf.ceil(target_shape[1]/new_w), tf.int32)
            img_replicated = tf.tile(img_resized, tf.stack([1, n_replication, 1]))
            pad_image = tf.image.crop_to_bounding_box(image=img_replicated, offset_height=0, offset_width=0,
                                                      target_height=target_shape[0], target_width=target_shape[1])

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def simple_resize():
        with tf.name_scope('simple_resize'):
            img_resized = tf.image.resize_images(image, target_shape)

            img_resized.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return img_resized, target_shape

    # 3 cases
    pad_image, (new_h, new_w) = tf.case(
        {  # case 1 : new_w >= target_w
            tf.logical_and(tf.greater_equal(ratio, target_ratio),
                           tf.greater_equal(new_w, target_w)): simple_resize,
            # case 2 : new_w >= target_w/2 & new_w < target_w & ratio < target_ratio
            tf.logical_and(tf.less(ratio, target_ratio),
                           tf.logical_and(tf.greater_equal(new_w, tf.cast(tf.divide(target_w, 2), tf.int32)),
                                          tf.less(new_w, target_w))): pad_fn,
            # case 3 : new_w < target_w/2 & new_w < target_w & ratio < target_ratio
            tf.logical_and(tf.less(ratio, target_ratio),
                           tf.logical_and(tf.less(new_w, target_w),
                                          tf.less(new_w, tf.cast(tf.divide(target_w, 2), tf.int32)))): replicate_fn
        },
        default=simple_resize, exclusive=True)

    return pad_image, new_w  # new_w = image width used for computing sequence lengths


class Data(object):

    @staticmethod
    def preprocess_inference(image_path, resize=(64,128), padding=False):
        # Preprocess individual images during inference
        image_path = tf.squeeze(image_path)
        image = tf.image.decode_png(tf.read_file(image_path), channels=1)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.per_image_standardization(image)
        if padding:
            image, width = padding_inputs_width(image, resize, increment=2*2)
        else:
            image = tf.image.resize_images(image, size=resize)
            width = tf.shape(image)[1]
            image = tf.reshape(image, [1,64,128,1])

        return image, width

    @staticmethod
    def load_dataframe(filename):
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)

        return df['path'].values, df['label'].values

    @staticmethod
    def load_dataset(path_placeholder, labels_placeholder, batch_size,
            resize=(64,128), test=False, augment=False, padding=False):

        def _augment(image):
            # On-the-fly data augmentation
            image = random_padding(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, 0.5, 1.5)
            image = random_rotation(image, 0.05, crop=True) # radians

            return image

        def _preprocess(image_path, label):
            # Apply to each element of the input dataset
            image = tf.image.decode_png(tf.read_file(image_path), channels=1)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            if augment:
                image = _augment(image)

            image = tf.image.per_image_standardization(image)
            if padding:
                image, width = padding_inputs_width(image, resize, increment=2*2)
            else:
                image = tf.image.resize_images(image, size=resize)
                width = tf.shape(image)[1]

            return image, label, width

        dataset = tf.contrib.data.Dataset.from_tensor_slices((path_placeholder,
                    labels_placeholder))
        dataset = dataset.map(_preprocess)
        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(batch_size)

        if test:
            dataset = dataset.repeat()

        return dataset
