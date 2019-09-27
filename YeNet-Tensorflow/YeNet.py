import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import add_arg_scope, arg_scope, arg_scoped_arguments
import layers as my_layers
from utils import *

SRM_Kernels = np.load('SRM_Kernels.npy')

class YeNet(Model):
    def __init__(self, is_training=None, data_format='NCHW', \
                 with_bn=False, tlu_threshold=3):
        super(YeNet, self).__init__(is_training=is_training, \
                                    data_format=data_format)
        self.with_bn = with_bn
        self.tlu_threshold = tlu_threshold

    def _build_model(self, inputs):
        self.inputs = inputs
        if self.data_format == 'NCHW':
            channel_axis = 1
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:
            channel_axis = 3
            _inputs = tf.cast(inputs, tf.float32)
        self.L = []
        with arg_scope([layers.avg_pool2d], \
                padding='VALID', data_format=self.data_format):
            with tf.variable_scope('SRM_preprocess'):
                W_SRM = tf.get_variable('W', initializer=SRM_Kernels, \
                            dtype=tf.float32, \
                            regularizer=None)
                b = tf.get_variable('b', shape=[30], dtype=tf.float32, \
                            initializer=tf.constant_initializer(0.))
                self.L.append(tf.nn.bias_add( \
                        tf.nn.conv2d(_inputs, \
                        W_SRM, [1,1,1,1], 'VALID', \
                        data_format=self.data_format), b, \
                        data_format=self.data_format, name='Layer1'))
                self.L.append(tf.clip_by_value(self.L[-1], \
                              -self.tlu_threshold, self.tlu_threshold, \
                              name='TLU'))
            with tf.variable_scope('ConvNetwork'):
                with arg_scope([my_layers.conv2d], num_outputs=30, \
                        kernel_size=3, stride=1, padding='VALID', \
                        data_format=self.data_format, \
                        activation_fn=tf.nn.relu, \
                        weights_initializer=layers.xavier_initializer_conv2d(), \
                        weights_regularizer=layers.l2_regularizer(5e-4), \
                        biases_initializer=tf.constant_initializer(0.2), \
                        biases_regularizer=None), arg_scope([layers.batch_norm], \
                        decay=0.9, center=True, scale=True, \
                        updates_collections=None, is_training=self.is_training, \
                        fused=True, data_format=self.data_format):
                    if self.with_bn:
                        self.L.append(layers.batch_norm(self.L[-1], \
                                      scope='Norm1'))
                    self.L.append(my_layers.conv2d(self.L[-1], \
                                  scope='Layer2'))
                    if self.with_bn:
                        self.L.append(layers.batch_norm(self.L[-1], \
                                      scope='Norm2'))
                    self.L.append(my_layers.conv2d(self.L[-1], \
                                  scope='Layer3'))
                    if self.with_bn:
                        self.L.append(layers.batch_norm(self.L[-1], \
                                      scope='Norm3'))
                    self.L.append(my_layers.conv2d(self.L[-1], \
                                  scope='Layer4'))
                    if self.with_bn:
                        self.L.append(layers.batch_norm(self.L[-1], \
                                      scope='Norm4'))
                    self.L.append(layers.avg_pool2d(self.L[-1], \
                                  kernel_size=[2,2], scope='Stride1'))
                    with arg_scope([my_layers.conv2d], kernel_size=5, \
                                   num_outputs=32):
                        self.L.append(my_layers.conv2d(self.L[-1], \
                                      scope='Layer5'))
                        if self.with_bn:
                            self.L.append(layers.batch_norm(self.L[-1], \
                                          scope='Norm5'))
                        self.L.append(layers.avg_pool2d(self.L[-1], \
                                      kernel_size=[3,3], \
                                      scope='Stride2'))
                        self.L.append(my_layers.conv2d(self.L[-1], \
                                      scope='Layer6'))
                        if self.with_bn:
                            self.L.append(layers.batch_norm(self.L[-1], \
                                          scope='Norm6'))
                        self.L.append(layers.avg_pool2d(self.L[-1], \
                                      kernel_size=[3,3], \
                                      scope='Stride3'))
                        self.L.append(my_layers.conv2d(self.L[-1], \
                                      scope='Layer7'))
                        if self.with_bn:
                            self.L.append(layers.batch_norm(self.L[-1], \
                                          scope='Norm7'))
                    self.L.append(layers.avg_pool2d(self.L[-1], \
                                  kernel_size=[3,3], \
                                  scope='Stride4'))
                    self.L.append(my_layers.conv2d(self.L[-1], \
                                  num_outputs=16, \
                                  scope='Layer8'))
                    if self.with_bn:
                        self.L.append(layers.batch_norm(self.L[-1], \
                                      scope='Norm8'))
                    self.L.append(my_layers.conv2d(self.L[-1], \
                                  num_outputs=16, stride=3, \
                                  scope='Layer9'))
                    if self.with_bn:
                        self.L.append(layers.batch_norm(self.L[-1], \
                                      scope='Norm9'))
                self.L.append(layers.flatten(self.L[-1]))
                self.L.append(layers.fully_connected(self.L[-1], num_outputs=2, \
                        activation_fn=None, normalizer_fn=None, \
                        weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), \
                        biases_initializer=tf.constant_initializer(0.), scope='ip'))
        self.outputs = self.L[-1]
        return self.outputs
    
