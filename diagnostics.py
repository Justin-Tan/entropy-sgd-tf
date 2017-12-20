# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
import os, time
from tensorflow.python.client import device_lib

class Diagnostics(object):

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, global_step, start_time, v_acc_best, epoch):
        t0 = time.time()

        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle, model.rnn_keep_prob: 1.0}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle, model.rnn_keep_prob: 1.0}
		
        try:
            t_acc, t_loss, t_ler, t_summary = sess.run([model.accuracy, model.ctc_cost, model.label_error_rate, model.merge_op],
                                                 feed_dict = feed_dict_train)
            model.train_writer.add_summary(t_summary, global_step)
        except tf.errors.OutOfRangeError:
            t_loss, t_acc = 'n/a', 'n/a'

        v_acc, v_loss, v_ler, v_summary = sess.run([model.accuracy, model.ctc_cost, model.label_error_rate, model.merge_op],
                                            feed_dict = feed_dict_test)
        model.test_writer.add_summary(v_summary, global_step)

        if v_acc > v_acc_best:
            v_acc_best = v_acc
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, 'crnn_{}_epoch{}.ckpt'.format(config.mode, epoch)),
                            global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        if epoch % 10 == 0 and epoch>10:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, 'crnn_{}_epoch{}.ckpt'.format(config.mode, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))
        
        if isinstance(t_loss, str):
            print('Epoch {} | Training Acc: {} | Test Acc: {:.3f} | Train Loss: {} | Test Loss: {:.3f} | LER: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, t_loss, v_loss, v_ler, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))
        else:
            print('Epoch {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Train Loss: {:.3f} | Test Loss: {:.3f} | LER: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, t_loss, v_loss, v_ler, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return v_acc_best