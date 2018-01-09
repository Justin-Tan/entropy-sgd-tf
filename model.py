#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import glob, time, os

from network import Network
from data import Data
from optimizer import EntropySGD
from sgld import local_entropy_sgld
from graphdef import ResNet

class Model():
    def __init__(self, config, directories, name='', optimizer='entropy-sgd'):
        # Build the computational graph

        print('Using optimizer: {}'.format(optimizer))
        self.global_step = tf.Variable(0, trainable=False)
        self.sgld_global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)

        train_dataset = Data.load_dataset(directories.train,
                                          config.batch_size,
                                          augment=True)
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

        graph = ResNet(config, self.training_phase)
        self.logits = Network.wrn(self.example, config, self.training_phase)
        # self.logits = graph.wrn(self.example)

        self.pred = tf.argmax(self.logits, 1)
        self.softmax = tf.nn.softmax(self.logits)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
            labels=self.labels)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.cost += graph.weight_decay(var_label='kernel')

        # decay by e every 10 epochs
        # learning_rate = tf.train.natural_exp_decay(config.learning_rate,
        #    self.global_step, decay_steps=1, decay_rate=config.lr_decay_rate)

        if optimizer=='entropy-sgd':
            epoch_bounds = [4, 8, 12, 16, 20, 24]
            lr_values = [8e-1, 1.6e-1, 3.2e-2, 6.4e-3, 1.28e-3, 2.56e-4, 5.12e-5]
        else:
            epoch_bounds = [60, 120, 160, 200, 220, 240]
            lr_values = [1e-1, 2e-2, 4e-3, 8e-4, 1.6e-4, 3.2e-5, 6.4e-6]

        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries=[s*config.steps_per_epoch for s in
            epoch_bounds], values=lr_values)

        # Exponential scoping
        gamma = tf.train.exponential_decay(config.g0, self.global_step,
            decay_steps=1, decay_rate=(1+config.g1))

        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            if optimizer=='entropy-sgd':
                opt = EntropySGD(self.iterator, self.training_phase, self.sgld_global_step,
                        config={'lr':learning_rate, 'gamma':gamma, 'lr_prime':0.1, 'momentum':0.9})
                self.sgld_op = opt.sgld_opt.minimize(self.cost, global_step=self.sgld_global_step)
                self.opt_op = opt.minimize(self.cost, global_step=self.global_step)
            elif optimizer=='adam':
                self.opt_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, global_step=self.global_step)
            elif optimizer=='momentum':
                self.opt_op = tf.train.MomentumOptimizer(learning_rate, config.momentum,
                    use_nesterov=True).minimize(self.cost, global_step=self.global_step)
            elif optimizer=='sgd':
                self.opt_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost,
                    global_step=self.global_step)

        self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
        maintain_averages_op = self.ema.apply(tf.trainable_variables())

        with tf.control_dependencies(update_ops+[self.opt_op]):
            self.train_op = tf.group(maintain_averages_op)

        self.str_accuracy, self.update_accuracy = tf.metrics.accuracy(self.labels, self.pred)
        correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('gamma', gamma)
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('global_step', self.global_step)
        tf.summary.scalar('sgld_global_step', self.sgld_global_step)
        tf.summary.image('images', self.example, max_outputs=8)
        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_train_{}'.format(name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_test_{}'.format(name, time.strftime('%d-%m_%I:%M'))))
