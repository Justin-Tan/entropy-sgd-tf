""" Network wiring """

import tensorflow as tf
import numpy as np
import glob, time, os

class Network(object):

    @staticmethod
    def wrn(x, config, training, reuse=False, actv=tf.nn.relu):
        # Implements W-28-10 wide residual network
        # See Arxiv 1605.07146
        network_width = 10 # k
        block_multiplicity = 4 # n

        filters = [16, 16, 32, 64]
        init = tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}

        def residual_block(x, n_filters, actv, keep_prob, training, project_shortcut=False, first_block=False):
            init = tf.contrib.layers.xavier_initializer()
            kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}

            if project_shortcut:
                strides = [2,2] if not first_block else [1,1]
                identity_map = tf.layers.conv2d(x, filters=n_filters, kernel_size=[1,1],
                                   strides=strides, kernel_initializer=init, padding='same')
                # identity_map = tf.layers.batch_normalization(identity_map, **kwargs)
            else:
                strides = [1,1]
                identity_map = x

            bn = tf.layers.batch_normalization(x, **kwargs)
            conv = tf.layers.conv2d(bn, filters=n_filters, kernel_size=[3,3], activation=actv,
                       strides=strides, kernel_initializer=init, padding='same')

            bn = tf.layers.batch_normalization(conv, **kwargs)
            do = tf.layers.dropout(bn, rate=1-keep_prob, training=training)

            conv = tf.layers.conv2d(do, filters=n_filters, kernel_size=[3,3], activation=actv,
                       kernel_initializer=init, padding='same')
            out = tf.add(conv, identity_map)

            return out

        def residual_block_2(x, n_filters, actv, keep_prob, training, project_shortcut=False, first_block=False):
            init = tf.contrib.layers.xavier_initializer()
            kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
            prev_filters = x.get_shape().as_list()[-1]
            if project_shortcut:
                strides = [2,2] if not first_block else [1,1]
                # identity_map = tf.layers.conv2d(x, filters=n_filters, kernel_size=[1,1],
                #                   strides=strides, kernel_initializer=init, padding='same')
                identity_map = tf.layers.average_pooling2d(x, strides, strides, 'valid')
                identity_map = tf.pad(identity_map, 
                    tf.constant([[0,0],[0,0],[0,0],[(n_filters-prev_filters)//2, (n_filters-prev_filters)//2]]))
                # identity_map = tf.layers.batch_normalization(identity_map, **kwargs)
            else:
                strides = [1,1]
                identity_map = x

            x = tf.layers.batch_normalization(x, **kwargs)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=n_filters, kernel_size=[3,3], strides=strides,
                    kernel_initializer=init, padding='same')

            x = tf.layers.batch_normalization(x, **kwargs)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate=1-keep_prob, training=training)

            x = tf.layers.conv2d(x, filters=n_filters, kernel_size=[3,3],
                       kernel_initializer=init, padding='same')
            out = tf.add(x, identity_map)

            return out

        with tf.variable_scope('wrn_conv', reuse=reuse):
            # Initial convolution --------------------------------------------->
            with tf.variable_scope('conv0', reuse=reuse):
                conv = tf.layers.conv2d(x, filters[0], kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
            # Residual group 1 ------------------------------------------------>
            rb = conv
            f1 = filters[1]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group1/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f1, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training, first_block=True)
            # Residual group 2 ------------------------------------------------>
            f2 = filters[2]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group2/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f2, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training)
            # Residual group 3 ------------------------------------------------>
            f3 = filters[3]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group3/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f3, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training)
            # Avg pooling + output -------------------------------------------->
            with tf.variable_scope('output', reuse=reuse):
                bn = tf.nn.relu(tf.layers.batch_normalization(rb, **kwargs))
                avp = tf.layers.average_pooling2d(bn, pool_size=[8,8], strides=[1,1], padding='valid')
                flatten = tf.contrib.layers.flatten(avp)
                out = tf.layers.dense(flatten, units=config.n_classes, kernel_initializer=init)

            return out

    @staticmethod
    def cnn_base(x, config, training, reuse=False, actv=tf.nn.relu):
        init = tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        with tf.variable_scope('conv', reuse=reuse):
            # x = tf.reshape(x, shape=[-1, config.im_x, config.im_y, 1])
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

            with tf.variable_scope('fc1', reuse=reuse):
                flatten = tf.contrib.layers.flatten(hidden6)
                hidden7 = tf.layers.dense(flatten, units=512, kernel_initializer=init, activation=actv)

            with tf.variable_scope('output', reuse=reuse):
                cnn_out = tf.layers.dense(hidden7, units=config.n_classes, kernel_initializer=init)

        return cnn_out

    @staticmethod
    def cnn_elu(x, config, training, reuse=False, actv=tf.nn.elu):
        # CIFAR architecture used in arXiv 1511.07289
        init = tf.contrib.layers.xavier_initializer()
        filters = [384, 384, 384, 640, 640, 640, 768, 768, 768, 768, 896, 896,
            896, 1024, 1024, 1024, 1152, 1152, 128]
        dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0]
        with tf.variable_scope('conv', reuse=reuse):
            # x = tf.reshape(x, shape=[-1, config.im_x, config.im_y, 1])
            # Convolutional blocks -------------------------------------------->
            with tf.variable_scope('conv0', reuse=reuse):
                conv = tf.layers.conv2d(x, filters=filters[0], kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden0 = tf.layers.dropout(conv, rate=dropout[0], training=training)

            with tf.variable_scope('conv1', reuse=reuse):
                conv = tf.layers.conv2d(hidden0, filters=filters[1], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[2], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[3], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[4], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden1 = tf.layers.dropout(pool, rate=dropout[1], training=training)

            with tf.variable_scope('conv2', reuse=reuse):
                conv = tf.layers.conv2d(hidden1, filters=filters[5], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[6], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[7], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[8], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden2 = tf.layers.dropout(pool, rate=dropout[2], training=training)

            with tf.variable_scope('conv3', reuse=reuse):
                conv = tf.layers.conv2d(hidden2, filters=filters[9], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[10], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[11], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden3 = tf.layers.dropout(pool, rate=dropout[3], training=training)

            with tf.variable_scope('conv4', reuse=reuse):
                conv = tf.layers.conv2d(hidden3, filters=filters[12], kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[13], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[14], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden4 = tf.layers.dropout(pool, rate=dropout[4], training=training)

            with tf.variable_scope('conv5', reuse=reuse):
                conv = tf.layers.conv2d(hidden4, filters=filters[15], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[16], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden5 = tf.layers.dropout(pool, rate=dropout[5], training=training)

            with tf.variable_scope('conv6', reuse=reuse):
                conv = tf.layers.conv2d(hidden5, filters=filters[17], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[18], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden6 = tf.layers.dropout(conv, rate=dropout[6], training=training)

            with tf.variable_scope('fc1', reuse=reuse):
                flatten = tf.contrib.layers.flatten(hidden6)
                fc = tf.layers.dense(flatten, units=128, kernel_initializer=init, activation=actv)
                fc = tf.layers.dropout(fc, rate=dropout[6], training=training)

            with tf.variable_scope('output', reuse=reuse):
                cnn_out = tf.layers.dense(fc, units=config.n_classes, kernel_initializer=init)

        return cnn_out

    @staticmethod
    def cnn_elu_small(x, config, training, reuse=False, actv=tf.nn.elu):
        # CIFAR architecture used in arXiv 1511.07289
        init = tf.contrib.layers.xavier_initializer()
        filters = [192, 192, 220, 240, 240, 256, 256, 280, 280, 512, 512, 128]
        dropout = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        with tf.variable_scope('conv', reuse=reuse):
            # x = tf.reshape(x, shape=[-1, config.im_x, config.im_y, 1])
            # Convolutional blocks -------------------------------------------->
            with tf.variable_scope('conv0', reuse=reuse):
                conv = tf.layers.conv2d(x, filters=filters[0], kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                # pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden0 = tf.layers.dropout(conv, rate=dropout[0], training=training)

            with tf.variable_scope('conv1', reuse=reuse):
                conv = tf.layers.conv2d(hidden0, filters=filters[1], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[2], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[3], kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden1 = tf.layers.dropout(pool, rate=dropout[1], training=training)

            with tf.variable_scope('conv2', reuse=reuse):
                conv = tf.layers.conv2d(hidden1, filters=filters[4], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[5], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden2 = tf.layers.dropout(pool, rate=dropout[2], training=training)

            with tf.variable_scope('conv3', reuse=reuse):
                conv = tf.layers.conv2d(hidden2, filters=filters[6], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[7], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden3 = tf.layers.dropout(pool, rate=dropout[3], training=training)

            with tf.variable_scope('conv4', reuse=reuse):
                conv = tf.layers.conv2d(hidden3, filters=filters[8], kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[9], kernel_size=[2,2], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden4 = tf.layers.dropout(pool, rate=dropout[4], training=training)

            with tf.variable_scope('conv5', reuse=reuse):
                conv = tf.layers.conv2d(hidden4, filters=filters[10], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                conv = tf.layers.conv2d(conv, filters=filters[11], kernel_size=[1,1], activation=actv,
                                        kernel_initializer=init, padding='same')
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='valid')
                hidden5 = tf.layers.dropout(conv, rate=dropout[5], training=training)

            with tf.variable_scope('fc1', reuse=reuse):
                flatten = tf.contrib.layers.flatten(hidden5)
                fc = tf.layers.dense(flatten, units=128, kernel_initializer=init, activation=actv)
                fc = tf.layers.dropout(fc, rate=dropout[6], training=training)

            with tf.variable_scope('output', reuse=reuse):
                cnn_out = tf.layers.dense(fc, units=config.n_classes, kernel_initializer=init)

        return cnn_out

    @staticmethod
    def encoder(x, config, training, reuse=False, actv=tf.nn.relu):
        init =  tf.keras.initializers.he_normal()

