#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import glob, time, os
from network import Network
from data import Data
from config import Alphabet, directories
from decoding import get_words_from_chars

class Model():
    def __init__(self, config, directories, paths, labels, single_infer=False):
        # Build the computational graph
        self.global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        beta = config.learning_rate

        self.path_placeholder = tf.placeholder(paths.dtype, paths.shape)
        self.labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

        self.test_path_placeholder = tf.placeholder(paths.dtype)
        self.test_labels_placeholder = tf.placeholder(labels.dtype)

        self.val_path_placeholder = tf.placeholder(paths.dtype)
        self.val_labels_placeholder = tf.placeholder(labels.dtype)
        self.rnn_keep_prob = tf.placeholder(tf.float32)

        train_dataset = Data.load_dataset(self.path_placeholder,
                                          self.labels_placeholder,
                                          config.batch_size,
                                          augment=True,
                                          padding=True)
        test_dataset = Data.load_dataset(self.test_path_placeholder,
                                         self.test_labels_placeholder,
                                         config.batch_size,
                                         padding=True,
                                         test=True)
        val_dataset = Data.load_dataset(self.val_path_placeholder,
                                        self.val_labels_placeholder,
                                        config.batch_size)


        self.iterator = tf.contrib.data.Iterator.from_string_handle(self.handle,
                                                                    train_dataset.output_types,
                                                                    train_dataset.output_shapes)
        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()
        self.val_iterator = val_dataset.make_initializable_iterator()

        self.example, self.labels, self.width = self.iterator.get_next()
        if single_infer:
            self.path, self.labels = tf.placeholder(paths.dtype), tf.placeholder(labels.dtype)
            self.example, self.width = Data.preprocess_inference(self.path, padding=True)
            self.width = tf.expand_dims(self.width, axis=0)

        self.logits = Network.small_conv_rnn_ctc(self.example, config, self.training_phase, self.rnn_keep_prob)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        n_pools = config.n_pools # 2x2 pooling in layers 1,2,3
        self.seq_lengths = tf.divide(self.width, n_pools)
        keys = [c for c in Alphabet.alphabet]
        values = Alphabet.codes

        # Convert string label to code label
        with tf.name_scope('str2code_conversion'):
            table_str2int = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
            splited = tf.string_split(self.labels, delimiter='')
            codes = table_str2int.lookup(splited.values)
            self.sparse_code_target = tf.SparseTensor(splited.indices, codes, splited.dense_shape)

        seq_lengths_labels = tf.bincount(tf.cast(self.sparse_code_target.indices[:, 0], tf.int32),
                                         minlength=tf.shape(self.logits)[1])

        self.ctc_loss = tf.nn.ctc_loss(labels=self.sparse_code_target,
                                       inputs=self.logits,
                                       sequence_length=tf.cast(self.seq_lengths, tf.int32),
                                       ignore_longer_outputs_than_inputs=False,
                                       time_major=True)

        self.ctc_cost = tf.reduce_mean(self.ctc_loss)


        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.opt_op = tf.train.AdamOptimizer(beta).minimize(self.ctc_cost, global_step=self.global_step)

        self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
        maintain_averages_op = self.ema.apply(tf.trainable_variables())

        with tf.control_dependencies(update_ops+[self.opt_op]):
            self.train_op = tf.group(maintain_averages_op)

        with tf.name_scope('code2str_conversion'):
            decode_keys = tf.cast(Alphabet.codes, tf.int64)
            decode_values = [c for c in Alphabet.alphabet]
            table_int2str = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(decode_keys, decode_values), '?')

            #self.sparse_code_pred, self.log_prob = tf.nn.ctc_beam_search_decoder(
            #    self.logits,
            #    sequence_length=tf.cast(self.seq_lengths, tf.int32),
            #    merge_repeated=True,
            #    beam_width=100)
            self.sparse_code_pred, self.log_prob = tf.nn.ctc_greedy_decoder(
                self.logits,
                sequence_length = tf.cast(self.seq_lengths, tf.int32)
            )
            self.sparse_code_pred = self.sparse_code_pred[0]

        # self.decoded, self.log_prob = Model.ctc_decoder(inputs=self.logits,
        #                                                 seq_lengths=self.length,
        #                                                 greedy=True)

        seq_lengths_pred = tf.bincount(tf.cast(self.sparse_code_pred.indices[:, 0], tf.int32),
                                       minlength=tf.shape(self.logits)[1])
        pred_chars = table_int2str.lookup(self.sparse_code_pred)
        self.pred_string = get_words_from_chars(pred_chars.values, sequence_lengths=seq_lengths_pred)

        self.label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(self.sparse_code_pred, tf.int32),
                                                           self.sparse_code_target))

        target_chars = table_int2str.lookup(tf.cast(self.sparse_code_target, tf.int64))
        target_string = get_words_from_chars(target_chars.values, sequence_lengths=seq_lengths_labels)

        self.str_accuracy, self.update_accuracy = tf.metrics.accuracy(target_string, self.pred_string)
        correct_prediction = tf.equal(target_string, self.pred_string)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if single_infer:
            with tf.name_scope('confidence_estimate'):
                splited = tf.string_split(self.pred_string, delimiter='')
                codes = table_str2int.lookup(splited.values)
                sparse_code_pred = tf.SparseTensor(splited.indices, codes, splited.dense_shape)

            
            self.nll = tf.cond(tf.shape(splited)[-1]>0, 
                lambda: tf.nn.ctc_loss(labels=sparse_code_pred,
                                       inputs=self.logits,
                                       sequence_length=tf.cast(self.seq_lengths, tf.int32),
                                       ignore_longer_outputs_than_inputs=False,
                                       time_major=True), 
                lambda: tf.constant(100.0))
            self.seq_prob = tf.exp(-self.nll)

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', beta)
        tf.summary.scalar('ctc_cost', self.ctc_cost)
        tf.summary.image('images', self.example, max_outputs=8)
        tf.summary.text('labels', target_string)
        tf.summary.text('preds', self.pred_string)
        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'train_{}'.format(time.strftime('%d-%m_%I:%M'))), graph = tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'test_{}'.format(time.strftime('%d-%m_%I:%M'))))
