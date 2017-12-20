#!/usr/bin/python3
import tensorflow as tf
import time, os
import argparse

# User-defined
from network import Network
from diagnostics import Diagnostics
from data import Data
from model import Model
from config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def evaluate(config, directories, ckpt):
    pin_cpu = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':0})
    start = time.time()
    # Build graph
    cnn = Model(config, directories)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = cnn.ema.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session(config=pin_cpu) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('{} restored.'.format(ckpt.model_checkpoint_path))

        val_handle = sess.run(cnn.val_iterator.string_handle())
        sess.run(cnn.val_iterator.initializer)
        eval_dict = {cnn.training_phase: False, cnn.handle: val_handle}

        while True:
            try:
                _ = sess.run([cnn.update_accuracy, cnn.merge_op], feed_dict=eval_dict)
                v_acc = sess.run(cnn.str_accuracy, feed_dict=eval_dict)

            except tf.errors.OutOfRangeError:
                break

        print("Validation accuracy: {:.3f}".format(v_acc))
        print("Inference complete. Duration: %g s" %(time.time()-start))

        return v_acc


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="path to test dataset in tfrecords format")
    args = parser.parse_args()

    # Load training, test data
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    # Evaluate
    val_accuracy = evaluate(config_test, directories, ckpt)

if __name__ == '__main__':
    main()
