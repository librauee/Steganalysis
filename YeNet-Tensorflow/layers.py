import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import add_arg_scope

@add_arg_scope
def double_conv2d(ref_half, real_half,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  padding='SAME',
                  data_format=None,
                  rate=1,
                  activation_fn=tf.nn.relu,
                  normalizer_fn=None,
                  normalize_after_activation=True,
                  normalizer_params=None,
                  weights_initializer=layers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=tf.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None):
    with tf.variable_scope(scope, 'Conv', reuse=reuse):
        if data_format == 'NHWC':
            num_inputs = real_half.get_shape().as_list()[3]
            height = real_half.get_shape().as_list()[1]
            width = real_half.get_shape().as_list()[2]
            if isinstance(stride, int):
                strides = [1, stride, stride, 1]
            elif isinstance(stride, list) or isinstance(stride, tuple):
                if len(stride) == 1:
                    strides = [1] + stride * 2 + [1]
                else:
                    strides = [1, stride[0], stride[1], 1]
            else:
                raise TypeError('stride is not an int, list or' \
                                + 'a tuple, is %s' % type(stride))
        else:
            num_inputs = real_half.get_shape().as_list()[1]
            height = real_half.get_shape().as_list()[2]
            width = real_half.get_shape().as_list()[3]
            if isinstance(stride, int):
                strides = [1, 1, stride, stride]
            elif isinstance(stride, list) or isinstance(stride, tuple):
                if len(stride) == 1:
                    strides = [1, 1] + stride * 2
                else:
                    strides = [1, 1, stride[0], stride[1]]
            else:
                raise TypeError('stride is not an int, list or' \
                                + 'a tuple, is %s' % type(stride))
        if isinstance(kernel_size, int):
            kernel_height = kernel_size
            kernel_width = kernel_size
        elif isinstance(kernel_size, list) \
                or isinstance(kernel_size, tuple):
            kernel_height = kernel_size[0]
            kernel_width = kernel_size[1]
        else:
            raise ValueError('kernel_size is not an int, list or' \
                             + 'a tuple, is %s' % type(kernel_size))
        weights = tf.get_variable('weights', [kernel_height, \
                                  kernel_width, num_inputs, num_outputs], \
                                  'float32', weights_initializer, \
                                  weights_regularizer, trainable, \
                                  variables_collections)
        ref_outputs = tf.nn.conv2d(ref_half, weights, strides, padding, \
                                  data_format=data_format)
        real_outputs = tf.nn.conv2d(real_half, weights, strides, padding, \
                                    data_format=data_format)
        if biases_initializer is not None:
            biases = tf.get_variable('biases', [num_outputs], 'float32', \
                                     biases_initializer, \
                                     biases_regularizer, \
                                     trainable, variables_collections)
            ref_outputs = tf.nn.bias_add(ref_outputs, biases, data_format)
            real_outputs = tf.nn.bias_add(real_outputs, biases, data_format)
        if normalizer_fn is not None \
                and not normalize_after_activation:
            normalizer_params = normalizer_params or {}
            ref_outputs, real_outputs = normalizer_fn(ref_outputs, \
                                                      real_outputs,  \
                                                      **normalizer_params)
        if activation_fn is not None:
            ref_outputs = activation_fn(ref_outputs)
            real_outputs = activation_fn(real_outputs)
        if normalizer_fn is not None and normalize_after_activation:
            normalizer_params = normalizer_params or {}
            ref_outputs, real_outputs = normalizer_fn(ref_outputs, \
                                                      real_outputs,\
                                                      **normalizer_params)
    return ref_outputs, real_outputs

@add_arg_scope
def conv2d(inputs,
           num_outputs,
           kernel_size,
           stride=1,
           padding='SAME',
           data_format=None,
           rate=1,
           activation_fn=tf.nn.relu,
           normalizer_fn=None,
           normalize_after_activation=True,
           normalizer_params=None,
           weights_initializer=layers.xavier_initializer(),
           weights_regularizer=None,
           biases_initializer=tf.zeros_initializer(),
           biases_regularizer=None,
           reuse=None,
           variables_collections=None,
           outputs_collections=None,
           trainable=True,
           scope=None):
    with tf.variable_scope(scope, 'Conv', reuse=reuse):
        if data_format == 'NHWC':
            num_inputs = inputs.get_shape().as_list()[3]
            height = inputs.get_shape().as_list()[1]
            width = inputs.get_shape().as_list()[2]
            if isinstance(stride, int):
                strides = [1, stride, stride, 1]
            elif isinstance(stride, list) or isinstance(stride, tuple):
                if len(stride) == 1:
                    strides = [1] + stride * 2 + [1]
                else:
                    strides = [1, stride[0], stride[1], 1]
            else:
                raise TypeError('stride is not an int, list or' \
                                + 'a tuple, is %s' % type(stride))
        else:
            num_inputs = inputs.get_shape().as_list()[1]
            height = inputs.get_shape().as_list()[2]
            width = inputs.get_shape().as_list()[3]
            if isinstance(stride, int):
                strides = [1, 1, stride, stride]
            elif isinstance(stride, list) or isinstance(stride, tuple):
                if len(stride) == 1:
                    strides = [1, 1] + stride * 2
                else:
                    strides = [1, 1, stride[0], stride[1]]
            else:
                raise TypeError('stride is not an int, list or' \
                                + 'a tuple, is %s' % type(stride))
        if isinstance(kernel_size, int):
            kernel_height = kernel_size
            kernel_width = kernel_size
        elif isinstance(kernel_size, list) \
                or isinstance(kernel_size, tuple):
            kernel_height = kernel_size[0]
            kernel_width = kernel_size[1]
        else:
            raise ValueError('kernel_size is not an int, list or' \
                             + 'a tuple, is %s' % type(kernel_size))
        weights = tf.get_variable('weights', [kernel_height, \
                                  kernel_width, num_inputs, num_outputs], \
                                  'float32', weights_initializer, \
                                  weights_regularizer, trainable, \
                                  variables_collections)
        outputs = tf.nn.conv2d(inputs, weights, strides, padding, \
                            data_format=data_format)
        if biases_initializer is not None:
            biases = tf.get_variable('biases', [num_outputs], 'float32', \
                                     biases_initializer, \
                                     biases_regularizer, \
                                     trainable, variables_collections)
            outputs = tf.nn.bias_add(outputs, biases, data_format)
        if normalizer_fn is not None \
                and not normalize_after_activation:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if normalizer_fn is not None and normalize_after_activation:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
    return outputs

class Vbn_double(object):
    def __init__(self, x, epsilon=1e-5, scope=None):
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                if data_format == 'NCHW':
                    x = tf.reshape(x, [shape[0], shape[1], 0, 0])
                else:
                    x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(scope):
            self.epsilon = epsilon
            self.scope = scope
            self.mean, self.var = tf.nn.moments(x, [0,2,3], \
                                                keep_dims=True)
            self.inv_std = tf.rsqrt(self.var + epsilon)
            self.batch_size = int(x.get_shape()[0])
            out = self._normalize(x, self.mean, self.inv_std)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                if self.data_format == 'NCHW':
                    x = tf.reshape(x, [shape[0], shape[1], 0, 0])
                else:
                    x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
        with tf.variable_scope(self.scope, reuse=True):
            out = self._normalize(x, self.mean, self.inv_std)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
        return out

    def _normalize(self, x, mean, inv_std):
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        gamma = tf.get_variable("gamma", [1,shape[1],1,1],
                        initializer=tf.constant_initializer(1.))
        beta = tf.get_variable("beta", [1,shape[1],1,1],
                        initializer=tf.constant_initializer(0.))
        coeff = gamma * inv_std
        return (x * coeff) + (beta - mean * coeff)

@add_arg_scope
def vbn_double(ref_half, real_half, center=True, scale=True, epsilon=1e-5, \
               data_format='NCHW', instance_norm=True, scope=None, \
               reuse=None):
    assert isinstance(epsilon, float)
    shape = real_half.get_shape().as_list()
    batch_size = int(real_half.get_shape()[0])
    with tf.variable_scope(scope, 'VBN', reuse=reuse):
        if data_format == 'NCHW':
            if scale:
                gamma = tf.get_variable("gamma", [1,shape[1],1,1],
                            initializer=tf.constant_initializer(1.))
            if center:
                beta = tf.get_variable("beta", [1,shape[1],1,1],
                            initializer=tf.constant_initializer(0.))
            ref_mean, ref_var = tf.nn.moments(ref_half, [0,2,3], \
                                              keep_dims=True)
        else:
            if scale:
                gamma = tf.get_variable("gamma", [1,1,1,shape[-1]],
                            initializer=tf.constant_initializer(1.))
            if center:
                beta = tf.get_variable("beta", [1,1,1,shape[-1]],
                            initializer=tf.constant_initializer(0.))
            ref_mean, ref_var = tf.nn.moments(ref_half, [0,1,2], \
                                              keep_dims=True)
        def _normalize(x, mean, var):
            inv_std = tf.rsqrt(var + epsilon)
            if scale:
                coeff = inv_std * gamma
            else:
                coeff = inv_std
            if center:
                return (x * coeff) + (beta - mean * coeff)
            else:
                return (x - mean) * coeff
        if instance_norm:
            if data_format == 'NCHW':
                real_mean, real_var = tf.nn.moments(real_half, [2,3], \
                                                  keep_dims=True)
            else:
                real_mean, real_var = tf.nn.moments(real_half, [1,2], \
                                                  keep_dims=True)
            real_coeff = 1. / (batch_size + 1.)
            ref_coeff = 1. - real_coeff
            new_mean = real_coeff * real_mean + ref_coeff * ref_mean
            new_var = real_coeff * real_var + ref_coeff * ref_var
            ref_output = _normalize(ref_half, ref_mean, ref_var)
            real_output = _normalize(real_half, new_mean, new_var)
        else:
            ref_output = _normalize(ref_half, ref_mean, ref_var)
            real_output = _normalize(real_half, ref_mean, ref_var)
        return ref_output, real_output


@add_arg_scope
def vbn_single(x, center=True, scale=True, \
               epsilon=1e-5, data_format='NCHW', \
               instance_norm=True, scope=None, \
               reuse=None):
    assert isinstance(epsilon, float)
    shape = x.get_shape().as_list()
    if shape[0] is None:
        half_size = x.shape[0] // 2
    else:
        half_size = shape[0] // 2
    needs_reshape = len(shape) != 4
    if needs_reshape:
        orig_shape = shape
        if len(shape) == 2:
            if data_format == 'NCHW':
                x = tf.reshape(x, [shape[0], shape[1], 0, 0])
            else:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
        elif len(shape) == 1:
            x = tf.reshape(x, [shape[0], 1, 1, 1])
        else:
            assert False, shape
        shape = x.get_shape().as_list()
    batch_size = int(x.get_shape()[0])
    with tf.variable_scope(scope, 'VBN', reuse=reuse):
        ref_half = tf.slice(x, [0,0,0,0], [half_size, shape[1], \
                            shape[2], shape[3]])
        if data_format == 'NCHW':
            if scale:
                gamma = tf.get_variable("gamma", [1,shape[1],1,1],
                            initializer=tf.constant_initializer(1.))
            if center:
                beta = tf.get_variable("beta", [1,shape[1],1,1],
                            initializer=tf.constant_initializer(0.))
            ref_mean, ref_var = tf.nn.moments(ref_half, [0,2,3], \
                                              keep_dims=True)
        else:
            if scale:
                gamma = tf.get_variable("gamma", [1,1,1,shape[-1]],
                            initializer=tf.constant_initializer(1.))
            if center:
                beta = tf.get_variable("beta", [1,1,1,shape[-1]],
                            initializer=tf.constant_initializer(0.))
            ref_mean, ref_var = tf.nn.moments(ref_half, [0,1,2], \
                                              keep_dims=True)
        def _normalize(x, mean, var):
            inv_std = tf.rsqrt(var + epsilon)
            if scale:
                coeff = inv_std * gamma
            else:
                coeff = inv_std
            if center:
                return (x * coeff) + (beta - mean * coeff)
            else:
                return (x - mean) * coeff
        if instance_norm:
            real_half = tf.slice(x, [half_size,0,0,0], \
                                 [half_size, shape[1], shape[2], shape[3]])
            if data_format == 'NCHW':
                real_mean, real_var = tf.nn.moments(real_half, [2,3], \
                                                  keep_dims=True)
            else:
                real_mean, real_var = tf.nn.moments(real_half, [1,2], \
                                                  keep_dims=True)
            real_coeff = 1. / (batch_size + 1.)
            ref_coeff = 1. - real_coeff
            new_mean = real_coeff * real_mean + ref_coeff * ref_mean
            new_var = real_coeff * real_var + ref_coeff * ref_var
            ref_output = _normalize(ref_half, ref_mean, ref_var)
            real_output = _normalize(real_half, new_mean, new_var)
            return tf.concat([ref_output, real_output], axis=0)
        else:
            return _normalize(x, ref_mean, ref_var)

