import tensorflow as tf
import numpy as np


def get_deconv_kernel(name, input_depth, output_depth):
    # kernel shape (size, size, output dim, input dim)
    return tf.get_variable(name=name, shape=[3, 3, output_depth, input_depth],
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))


def get_deconv_bias(name, output_depth):
    return tf.get_variable(name=name, shape=[output_depth], initializer=tf.zeros_initializer())


def deconv_layer(name, input, output_depth):
    input_shape = input.get_shape().as_list()
    output_shape = [input_shape[0], input_shape[1]+2, input_shape[2]+2, output_depth]
    input_depth = input_shape[3]

    with tf.variable_scope(name):
        kernel = get_deconv_kernel('kernel', input_depth, output_depth)
        bias = get_deconv_bias('bias', output_depth)

    deconv = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(deconv, bias))
