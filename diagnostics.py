# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
import os, time
from tensorflow.python.client import device_lib
from datasets import dl_cifar10, dlcifar100

class Diagnostics(object, dataset_name):
    
    @staticmethod
    def setup_ckpts():
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
            os.mkdir('checkpoints/best')

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, start_time, v_acc_best, epoch, name):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        try:
            t_acc, t_loss, t_summary = sess.run([model.accuracy, model.cost, model.merge_op], feed_dict=feed_dict_train)
            model.train_writer.add_summary(t_summary)
        except tf.errors.OutOfRangeError:
            t_loss, t_acc = float('nan'), float('nan')

        v_acc, v_loss, v_summary = sess.run([model.accuracy, model.cost, model.merge_op], feed_dict=feed_dict_test)
        model.test_writer.add_summary(v_summary)

        if v_acc > v_acc_best:
            v_acc_best = v_acc
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, 'crnn_{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        if epoch % 10 == 0 and epoch>10:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, 'crnn_{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))

        print('Epoch {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Train Loss: {:.3f} | Test Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, t_loss, v_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return v_acc_best
