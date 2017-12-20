#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import glob, time, os
from network import Network
from data import Data

class Model():
    def __init__(self, config, directories, single_infer=False):
        # Build the computational graph
        self.global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        beta = config.learning_rate

        train_dataset = Data.load_dataset(directories.train,
                                          config.batch_size,
                                          augment=False)
        test_dataset = Data.load_dataset(directories.test,
                                         config.batch_size,
                                         augment=False,
                                         test=True)
        val_dataset = Data.load_dataset(directories.test, config.batch_size)

        self.iterator = tf.contrib.data.Iterator.from_string_handle(self.handle,
                                                                    train_dataset.output_types,
                                                                    train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()
        self.val_iterator = val_dataset.make_initializable_iterator()

        self.example, self.labels = self.iterator.get_next()

        if single_infer:
            self.path = tf.placeholder(paths.dtype)
            self.example = Data.preprocess_inference(self.path)

        self.logits = Network.cnn(self.example, config, self.training_phase)
        self.pred = tf.argmax(self.logits, 1)
        self.softmax = tf.nn.softmax(self.logits)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
            labels=self.labels)
        self.cost = tf.reduce_mean(self.cross_entropy)

        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.opt_op = tf.train.AdamOptimizer(beta).minimize(self.cost, global_step=self.global_step)

        self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
        maintain_averages_op = self.ema.apply(tf.trainable_variables())

        with tf.control_dependencies(update_ops+[self.opt_op]):
            self.train_op = tf.group(maintain_averages_op)

        self.str_accuracy, self.update_accuracy = tf.metrics.accuracy(self.labels, self.pred)
        correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', beta)
        tf.summary.scalar('cost', self.cost)
        tf.summary.image('images', self.example, max_outputs=8)
        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'train_{}'.format(time.strftime('%d-%m_%I:%M'))), graph = tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'test_{}'.format(time.strftime('%d-%m_%I:%M'))))
