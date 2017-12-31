#!/usr/bin/env python3

class config_test(object):
    mode = 'alpha'
    n_layers = 5
    num_epochs = 512
    batch_size = 256
    ema_decay = 0.999
    learning_rate = 8e-5
    n_classes = 10
    conv_keep_prob = 1.0
    dense_keep_prob = 1.0
    im_x = 32
    im_y = 32
    g0 = 1e-2
    g1 = 1e-3

class config_train(object):
    mode = 'beta'
    n_layers = 5
    num_epochs = 512
    batch_size = 256
    ema_decay = 0.999
    learning_rate = 8e-5
    n_classes = 10
    conv_keep_prob = 0.75
    dense_keep_prob = 0.75
    im_x = 32
    im_y = 32
    g0 = 1e-2
    g1 = 1e-3

class directories(object):
    train = 'tfrecords/cifar10/cifar10_train.tfrecord'
    test = 'tfrecords/cifar10/cifar10_test.tfrecord'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
