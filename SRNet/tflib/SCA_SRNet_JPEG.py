import tensorflow as tf
from functools import partial
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import functools
from queues import *
from generator import *       
from utils_multistep_lr import *

class SCA_SRNet(Model):
    def _build_model(self, input_batch):
        inputs_image, inputs_Beta = tf.split(input_batch, num_or_size_splits=2, axis=3)
        if self.data_format == 'NCHW':
            reduction_axis = [2,3]
            _inputs_image = tf.cast(tf.transpose(inputs_image, [0, 3, 1, 2]), tf.float32)
            _inputs_Beta = tf.cast(tf.transpose(inputs_Beta, [0, 3, 1, 2]), tf.float32)
        else:
            reduction_axis = [1,2]
            _inputs_image = tf.cast(inputs_image, tf.float32)
            _inputs_Beta = tf.cast(inputs_Beta, tf.float32)
        with arg_scope([layers.conv2d], num_outputs=16,
                       kernel_size=3, stride=1, padding='SAME',
                       data_format=self.data_format,
                       activation_fn=None,
                       weights_initializer=layers.variance_scaling_initializer(),
                       weights_regularizer=layers.l2_regularizer(2e-4),
                       biases_initializer=tf.constant_initializer(0.2),
                       biases_regularizer=None),\
            arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True, 
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format),\
            arg_scope([layers.avg_pool2d],
                       kernel_size=[3,3], stride=[2,2], padding='SAME',
                       data_format=self.data_format):
            with tf.variable_scope('Layer1'): # 256*256
                W = tf.get_variable('W', shape=[3,3,1,64],\
                            initializer=layers.variance_scaling_initializer(), \
                            dtype=tf.float32, \
                            regularizer=layers.l2_regularizer(5e-4))
                b = tf.get_variable('b', shape=[64], dtype=tf.float32, \
                            initializer=tf.constant_initializer(0.2))
                conv = tf.nn.bias_add( \
                        tf.nn.conv2d(tf.cast(_inputs_image, tf.float32), \
                        W, [1,1,1,1], 'SAME', \
                        data_format=self.data_format), b, \
                        data_format=self.data_format, name='Layer1')
                actv=tf.nn.relu(conv)
                prob_map = tf.sqrt(tf.nn.conv2d(tf.cast(_inputs_Beta, tf.float32), \
                        tf.abs(W), [1,1,1,1], 'SAME', \
                        data_format=self.data_format))
                out_L1=tf.add_n([actv,prob_map])
            with tf.variable_scope('Layer2'): # 256*256
                conv=layers.conv2d(out_L1)
                actv=tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer3'): # 256*256
                conv1=layers.conv2d(actv)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn2=layers.batch_norm(conv2)
                res= tf.add(actv, bn2)
            with tf.variable_scope('Layer4'): # 256*256
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn2=layers.batch_norm(conv2)
                res= tf.add(res, bn2)
            with tf.variable_scope('Layer5'): # 256*256
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer6'): # 256*256
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer7'): # 256*256
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer8'): # 256*256
                convs = layers.conv2d(res, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer9'):  # 128*128
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=64)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=64)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer10'): # 64*64
                convs = layers.conv2d(res, num_outputs=128, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=128)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=128)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer11'): # 32*32
                convs = layers.conv2d(res, num_outputs=256, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=256)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=256)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer12'): # 16*16
                conv1=layers.conv2d(res, num_outputs=512)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=512)
                bn=layers.batch_norm(conv2)
                avgp = tf.reduce_mean(bn, reduction_axis,  keep_dims=True )
        ip=layers.fully_connected(layers.flatten(avgp), num_outputs=2,
                    activation_fn=None, normalizer_fn=None,
                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                    biases_initializer=tf.constant_initializer(0.), scope='ip')
        self.outputs = ip
        return self.outputs