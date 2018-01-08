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

class Data(object):

    @staticmethod
    def preprocess_inference(image_path, resize=(32,32)):
        # Preprocess individual images during inference
        image_path = tf.squeeze(image_path)
        image = tf.image.decode_png(tf.read_file(image_path))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize_images(image, size=resize)

        return image

    @staticmethod
    def load_dataset(filenames, batch_size, resize=(32,32), test=False,
                     augment=False):
        # Consume TFRecord image data

        def _augment(image):
            # On-the-fly data augmentation
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, 0.5, 1.5)
            image = tf.image.random_flip_left_right(image)
            image = random_rotation(image, 0.05, crop=True) # radians

            return image

        def _parser(record):
            keys_to_features = {
                "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature((), tf.int64)
            }
            parsed = tf.parse_single_example(record, keys_to_features)

            # image = tf.decode_raw(parsed['image/encoded'], tf.uint8)
            image = tf.image.decode_image(parsed["image/encoded"])
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image.set_shape([32,32,3])

            if augment:
                image = _augment(image)

            image = tf.image.per_image_standardization(image)
            image = tf.image.resize_images(image, size=resize)
            label = tf.cast(parsed["image/class/label"], tf.int32)

            return image, label

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parser)
        dataset = dataset.shuffle(buffer_size=2048)
        dataset = dataset.batch(batch_size)

        if test:
            dataset = dataset.repeat()

        return dataset
