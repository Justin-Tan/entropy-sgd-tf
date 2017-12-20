#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# User-defined
from network import Network
from diagnostics import Diagnostics
from data import Data
from model import Model
from config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def infer(config, directories, ckpt, path, label):
    pin_cpu = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':0})
    start = time.time()

    #Build graph
    cnn = Model(config, directories, path, label, single_infer=True)
    
    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = cnn.ema.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session(config=pin_cpu) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('{} restored.'.format(ckpt.model_checkpoint_path))

        eval_dict = {cnn.training_phase: False, cnn.path: path, cnn.labels: label}

        seq_prob, predicted_string = sess.run([cnn.seq_prob, cnn.pred_string], feed_dict=eval_dict) 
        print('Predicted string: {} | Confidence: {}'.format(predicted_string, seq_prob))
        if seq_prob < 0.001:
            print('Unable to extract digit string from image!')
        return np.asscalar(predicted_string)

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", help="path to image")
    args = parser.parse_args()

    # Load most recent checkpoint - TODO enable checkpoint selection
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    # Perform inference
    sample_path = pd.Series([args.image_path]).values
    sample_label = pd.Series(['111']).values

    predicted_string = infer(config_test, directories, ckpt, path=sample_path, label=sample_label)
   
    # Temporary 
    x = mpimg.imread(args.image_path)
    plt.imshow(x, cmap='gray')
    plt.title('Prediction: {}'.format(str(predicted_string)))
    plt.show()

if __name__ == '__main__':
    main()
