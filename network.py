import tensorflow as tf
import numpy as np
import glob, time, os

class Network(object):

    @staticmethod
    def cnn(x, config, training, reuse=False, actv=tf.nn.relu):
        init = tf.contrib.layers.xavier_initializer()
        kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': True}
        with tf.variable_scope('conv_rnn', reuse=reuse):
            x = tf.reshape(x, shape=[-1, config.im_x, config.im_y, 1])

            # Convolutional blocks -------------------------------------------->
            # max pool 2x2
            with tf.variable_scope('conv0', reuse=reuse):
                conv = tf.layers.conv2d(x, filters=64, kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='same')
                bn = tf.layers.batch_normalization(pool, **kwargs)
                hidden0 = tf.layers.dropout(bn, rate=1-config.conv_keep_prob, training=training)

            # max pool 2x2
            with tf.variable_scope('conv1', reuse=reuse):
                conv = tf.layers.conv2d(hidden0, filters=128, kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='same')
                bn = tf.layers.batch_normalization(pool, **kwargs)
                hidden1 = tf.layers.dropout(bn, rate=1-config.conv_keep_prob, training=training)

            # batch norm
            with tf.variable_scope('conv2', reuse=reuse):
                conv = tf.layers.conv2d(hidden1, filters=256, kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='same')
                bn = tf.layers.batch_normalization(pool, **kwargs)
                hidden2 = tf.layers.dropout(bn, rate=1-config.conv_keep_prob, training=training)

            # maxpool 2x2
            with tf.variable_scope('conv3', reuse=reuse):
                conv = tf.layers.conv2d(hidden2, filters=256, kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='same')
                bn = tf.layers.batch_normalization(pool, **kwargs)
                hidden3 = tf.layers.dropout(bn, rate=1-config.conv_keep_prob, training=training)

            # batch norm
            with tf.variable_scope('conv4', reuse=reuse):
                conv = tf.layers.conv2d(hidden3, filters=512, kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='same')
                bn = tf.layers.batch_normalization(pool, **kwargs)
                hidden4 = tf.layers.dropout(bn, rate=1-config.conv_keep_prob, training=training)

            # max pool 2x2
            with tf.variable_scope('conv5', reuse=reuse):
                conv = tf.layers.conv2d(hidden4, filters=512, kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='same')
                bn = tf.layers.batch_normalization(pool, **kwargs)
                hidden5 = tf.layers.dropout(bn, rate=1-config.conv_keep_prob, training=training)

            # batch norm
            with tf.variable_scope('conv6', reuse=reuse):
                conv = tf.layers.conv2d(hidden3, filters=512, kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='same')
                bn = tf.layers.batch_normalization(pool, **kwargs)
                hidden6 = tf.layers.dropout(bn, rate=1-config.conv_keep_prob, training=training)

            with tf.variable_scope('fc1', reuse=resue):
                cnn_out = tf.layers.dense(hidden6, units=config.n_classes, kernel_initializer=init)

        return cnn_out
