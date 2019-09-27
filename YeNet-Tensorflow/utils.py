import tensorflow as tf
import numpy as np
import sys
import time
from glob import glob
from functools import partial
import os
from os.path import expanduser
home = expanduser("~")
user = home.split('/')[-1]
sys.path.append(home + '/tflib/')
from queues import *
from generator import *

def optimistic_restore(session, save_file, \
                       graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])    
    restore_vars = []    
    for var_name, saved_var_name in var_names:            
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

class average_summary(object):
    def __init__(self, variable, name, num_iterations):
        self.sum_variable = tf.get_variable(name, shape=[], \
                                initializer=tf.constant_initializer(0.), \
                                dtype='float32', \
                                trainable=False, \
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])
        with tf.control_dependencies([variable]):
            self.increment_op = tf.assign_add(self.sum_variable, variable)
        self.mean_variable = self.sum_variable / float(num_iterations)
        self.summary = tf.summary.scalar(name, self.mean_variable)
        with tf.control_dependencies([self.summary]):
            self.reset_variable_op = tf.assign(self.sum_variable, 0)

    def add_summary(self, sess, writer, step):
        s, _ = sess.run([self.summary, self.reset_variable_op])
        writer.add_summary(s, step)
    
class Model(object):
    def __init__(self, is_training=None, data_format='NCHW'):
        self.data_format = data_format
        if is_training is None:
            self.is_training = tf.get_variable('is_training', dtype=tf.bool, \
                                    initializer=tf.constant_initializer(True), \
                                    trainable=False)
        else:
            self.is_training = is_training

    def _build_model(self, inputs):
        raise NotImplementedError('Here is your model definition')

    def _build_losses(self, labels):
        self.labels = tf.cast(labels, tf.int64)
        with tf.variable_scope('loss'):
            oh = tf.one_hot(self.labels, 2)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
                                  labels=oh, logits=self.outputs))
        with tf.variable_scope('accuracy'):
            am = tf.argmax(self.outputs, 1)
            equal = tf.equal(am, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
        return self.loss, self.accuracy

def train(model_class, train_gen, valid_gen, train_batch_size, \
          valid_batch_size, valid_ds_size, optimizer, \
          train_interval, valid_interval, max_iter, \
          save_interval, log_path, num_runner_threads=1, \
          load_path=None):
    tf.reset_default_graph()
    train_runner = GeneratorRunner(train_gen, train_batch_size * 10)
    valid_runner = GeneratorRunner(valid_gen, valid_batch_size * 10)
    is_training = tf.get_variable('is_training', dtype=tf.bool, \
                                  initializer=True, trainable=False)
    if train_batch_size == valid_batch_size:
        batch_size = train_batch_size
        disable_training_op = tf.assign(is_training, False)
        enable_training_op = tf.assign(is_training, True)
    else:
        batch_size = tf.get_variable('batch_size', dtype=tf.int32, \
                                     initializer=train_batch_size, \
                                     trainable=False, \
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])
        disable_training_op = tf.group(tf.assign(is_training, False), \
                                tf.assign(batch_size, valid_batch_size))
        enable_training_op = tf.group(tf.assign(is_training, True), \
                                tf.assign(batch_size, train_batch_size))
    img_batch, label_batch = queueSelection([valid_runner, train_runner], \
                                            tf.cast(is_training, tf.int32), \
                                            batch_size)
    model = model_class(is_training, 'NCHW')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    regularization_losses = tf.get_collection(
                          tf.GraphKeys.REGULARIZATION_LOSSES)
    regularized_loss = tf.add_n([loss] + regularization_losses)
    train_loss_s = average_summary(loss, 'train_loss', train_interval)
    train_accuracy_s = average_summary(accuracy, 'train_accuracy', \
                                       train_interval)
    valid_loss_s = average_summary(loss, 'valid_loss', \
                                   float(valid_ds_size) / float(valid_batch_size))
    valid_accuracy_s = average_summary(accuracy, 'valid_accuracy', \
                                       float(valid_ds_size) / float(valid_batch_size))
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
    minimize_op = optimizer.minimize(regularized_loss, global_step)
    train_op = tf.group(minimize_op, train_loss_s.increment_op, \
                        train_accuracy_s.increment_op)
    increment_valid = tf.group(valid_loss_s.increment_op, \
                               valid_accuracy_s.increment_op)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        if load_path is not None:
            loader = tf.train.Saver(reshape=True)
            loader.restore(sess, load_path)
        train_runner.start_threads(sess, num_runner_threads)
        valid_runner.start_threads(sess, 1)
        writer = tf.summary.FileWriter(log_path + '/LogFile/', \
                                       sess.graph)
        start = sess.run(global_step)
        sess.run(disable_training_op)
        sess.run([valid_loss_s.reset_variable_op, \
                  valid_accuracy_s.reset_variable_op, \
                  train_loss_s.reset_variable_op, \
                  train_accuracy_s.reset_variable_op])
        _time = time.time()
        for j in range(0, valid_ds_size, valid_batch_size):
            sess.run([increment_valid])
        _acc_val = sess.run(valid_accuracy_s.mean_variable)
        print "validation:", _acc_val, " | ", \
                "duration:", time.time() - _time, \
                "seconds long"
        valid_accuracy_s.add_summary(sess, writer, start)
        valid_loss_s.add_summary(sess, writer, start)
        sess.run(enable_training_op)
        print valid_interval
        for i in xrange(start+1, max_iter+1):
            sess.run(train_op)
            if i % train_interval == 0:
                train_loss_s.add_summary(sess, writer, i)
                train_accuracy_s.add_summary(sess, writer, i)
            if i % valid_interval == 0:
                sess.run(disable_training_op)
                for j in range(0, valid_ds_size, valid_batch_size):
                    sess.run([increment_valid])
                valid_loss_s.add_summary(sess, writer, i)
                valid_accuracy_s.add_summary(sess, writer, i)
                sess.run(enable_training_op)
            if i % save_interval == 0:
                saver.save(sess, log_path + '/Model_' + str(i) + '.ckpt')

def test_dataset(model_class, gen, batch_size, ds_size, load_path):
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NCHW')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss',  \
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy',  \
                                   float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op, \
                            accuracy_summary.increment_op)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        for j in range(0, ds_size, batch_size):
            sess.run(increment_op)
        mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable ,\
                                             accuracy_summary.mean_variable])
    print "Accuracy:", mean_accuracy, " | Loss:", mean_loss

def find_best(model_class, valid_gen, test_gen, valid_batch_size, \
              test_batch_size, valid_ds_size, test_ds_size, load_paths):
    tf.reset_default_graph()
    valid_runner = GeneratorRunner(valid_gen, valid_batch_size * 30)
    img_batch, label_batch = valid_runner.get_batched_inputs(valid_batch_size)
    model = model_class(False, 'NCHW')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss',  \
                                          float(valid_ds_size) \
                                          / float(valid_batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy',  \
                                          float(valid_ds_size) \
                                          / float(valid_batch_size))
    increment_op = tf.group(loss_summary.increment_op, \
                            accuracy_summary.increment_op)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    accuracy_arr = []
    loss_arr = []
    print "validation"
    for load_path in load_paths:
        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, load_path)
            valid_runner.start_threads(sess, 1)
            _time = time.time()
            for j in range(0, valid_ds_size, valid_batch_size):
                sess.run(increment_op)
            mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable ,\
                                            accuracy_summary.mean_variable])
            accuracy_arr.append(mean_accuracy)
            loss_arr.append(mean_loss)
            print load_path
            print "Accuracy:", accuracy_arr[-1], "| Loss:", loss_arr[-1], \
                    "in", time.time() - _time, "seconds."
    argmax = np.argmax(accuracy_arr)
    print "best savestate:", load_paths[argmax], "with", \
            accuracy_arr[argmax], "accuracy and", loss_arr[argmax], \
            "loss on validation"
    print "test:"
    test_dataset(model_class, test_gen, test_batch_size, test_ds_size, \
                 load_paths[argmax])
    return argmax, accuracy_arr, loss_arr


def extract_stats_outputs(model_class, gen, batch_size, ds_size, load_path):
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NCHW')
    model._build_model(img_batch)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    stats_outputs_arr = np.empty([ds_size, \
        model.stats_outputs.get_shape().as_list()[1]])
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        for j in range(0, ds_size, batch_size):
            stats_outputs_arr[j:j+batch_size] = sess.run(model.stats_outputs)
    return stats_outputs_arr

def stats_outputs_all_datasets(model_class, ds_head_dir, payload, \
                               algorithm, load_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/')
    payload_str = ''.join(str(payload).strip('.'))
    train_ds_size = len(glob(ds_head_dir + '/train/cover/*'))
    valid_ds_size = len(glob(ds_head_dir + '/valid/cover/*'))
    test_ds_size = len(glob(ds_head_dir + '/test/cover/*'))
    train_gen = partial(gen_all_flip_and_rot, ds_head_dir + \
                        '/train/cover/', ds_head_dir + '/train/' + \
                        algorithm + '/payload' + payload_str + '/stego/')
    valid_gen = partial(gen_valid, ds_head_dir + '/valid/cover/', \
                        ds_head_dir + '/valid/' + algorithm + \
                        '/payload' + payload_str + '/stego/')
    test_gen = partial(gen_valid, ds_head_dir + '/test/cover/', \
                      ds_head_dir + '/test/' + algorithm + \
                      '/payload' + payload_str + '/stego/')
    print "train..."
    stats_outputs = extract_stats_outputs(model_class, train_gen, 16, \
                                                train_ds_size * 2 * 4 * 2, \
                                                load_path)
    stats_shape = stats_outputs.shape
    stats_outputs = stats_outputs.reshape(train_ds_size, 2, 4, \
                                          2, stats_shape[-1])
    stats_outputs = np.transpose(stats_outputs, axes=[0,3,2,1,4])
    np.save(save_dir + '/train.npy', stats_outputs)
    print "validation..."
    stats_outputs = extract_stats_outputs(model_class, valid_gen, 16, \
                                          valid_ds_size * 2, load_path)
    np.save(save_dir + '/valid.npy', stats_outputs)
    print "test..."
    stats_outputs = extract_stats_outputs(model_class, test_gen, 16, \
                                          test_ds_size * 2, load_path)
    np.save(save_dir + '/test.npy', stats_outputs)
