#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from diagnostics import Diagnostics
from data import Data
from model import Model
from config import config_train, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def train(config, architecture, restore=False, restore_path=None):
    print('Architecture: {}'.format(architecture))
    start_time = time.time()
    global_step, n_checkpoints, v_acc_best = 0, 0, 0.
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    # Build graph
    cnn = Model(config_train, directories)
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(cnn.train_iterator.string_handle())
        test_handle = sess.run(cnn.test_iterator.string_handle())

        if restore and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(restore_path))
                new_saver.restore(sess, restore_path)
                print('{} restored.'.format(restore_path))

        sess.run(cnn.test_iterator.initializer)

        for epoch in range(config.num_epochs):
            sess.run(cnn.train_iterator.initializer)
            # Run diagnostics
            v_acc_best = Diagnostics.run_diagnostics(cnn, config_train, directories, sess, saver, train_handle, test_handle, start_time, v_acc_best, epoch)
            while True:
                try:
                    sess.run([cnn.train_op, cnn.update_accuracy], feed_dict={cnn.training_phase: True,
                        cnn.handle: train_handle})

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        'cnn_{}_last.ckpt'.format(config.mode)), global_step=epoch)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()


        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               'cnn_{}_end.ckpt'.format(config.mode)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore", help="path to model to be restored")
    args = parser.parse_args()
    config=config_train

    architecture = 'Layers: {} | Conv dropout: {} | Base LR: {} | Epochs: {}'.format(
                    config.n_layers,
                    config.conv_keep_prob,
                    config.learning_rate,
                    config.num_epochs
    )
    # Launch training
    train(config_train, architecture, restore=args.restore_last, restore_path=args.restore)

if __name__ == '__main__':
    main()
