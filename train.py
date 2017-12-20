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

def train(config, architecture, paths, labels, paths_test, labels_test, restore=False, restore_path=None): 
    print('Architecture: {}'.format(architecture))
    start_time = time.time()
    global_step, n_checkpoints, v_acc_best = 0, 0, 0.
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    # Build graph
    cnn = Model(config_train, directories, paths, labels)
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
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

        sess.run(cnn.test_iterator.initializer, feed_dict={
            cnn.test_path_placeholder:paths_test,
            cnn.test_labels_placeholder:labels_test
        })

        for epoch in range(config.num_epochs):
            sess.run(cnn.train_iterator.initializer, feed_dict={
                cnn.path_placeholder:paths,
                cnn.labels_placeholder:labels
            })
            
            while True:
                try:
                    sess.run([cnn.train_op, cnn.update_accuracy], feed_dict={cnn.training_phase: True,
                        cnn.rnn_keep_prob: config.rnn_keep_prob, cnn.handle: train_handle})
                    global_step+=1

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break
                
                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        'crnn_{}_last.ckpt'.format(config.mode)), global_step=epoch)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()

                if global_step%100==0:
                    v_acc_best = Diagnostics.run_diagnostics(cnn, config_train, directories, sess, saver, train_handle, test_handle, global_step, start_time, v_acc_best, epoch)


        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               'crnn_{}_end.ckpt'.format(config.mode)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore", help="path to model to be restored")
    args = parser.parse_args()
    config=config_train

    # Load training, test data
    paths, labels = Data.load_dataframe(directories.train)
    test_paths, test_labels = Data.load_dataframe(directories.test)

    architecture = 'Layers: {} | Conv dropout: {} | RNN dropout: {} | Base LR: {} | Epochs: {}'.format(
                    config.n_layers,
                    config.conv_keep_prob,
                    config.rnn_keep_prob,
                    config.learning_rate,
                    config.num_epochs
    )
    # Launch training
    train(config_train, architecture, paths=paths, labels=labels, paths_test = test_paths, labels_test = test_labels,
            restore=args.restore_last, restore_path=args.restore)

if __name__ == '__main__':
    main()
