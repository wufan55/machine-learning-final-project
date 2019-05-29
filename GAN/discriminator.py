import tensorflow as tf
import numpy as np

# 常量定义
BATCH_SIZE = 10


def get_conv_kernel(name, input_depth, kernel_size, output_depth):
    return tf.get_variable(name=name, shape=[kernel_size, kernel_size, input_depth, output_depth],
                           initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))


def get_conv_bias(name, output_depth):
    return tf.get_variable(name=name, shape=[output_depth], initializer=tf.zeros_initializer())


# 卷积层
# 使用relu激活函数
def conv_layer(name, input, kernel_size, output_depth):
    input_shape = input.get_shape().as_list()
    input_depth = input_shape[3]

    with tf.variable_scope(name):
        kernel = get_conv_kernel('kernel', input_depth, kernel_size, output_depth)
        bias = get_conv_bias('bias', output_depth)

    conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(conv, bias))


# 池化层
def pool_layer(name, input, kernel_size):
    with tf.name_scope(name):
        return tf.nn.avg_pool(input, ksize=[1, kernel_size, kernel_size, 1], padding='VALID',
                              strides=[1, kernel_size, kernel_size, 1], name='avg_pool')


# 全连接层
# def full_connect_layer(name, input):


# 定义网络结构
# shape (28, 28, 1)
x_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 28, 28, 1], name='x_input')

# shape (24, 24, 2)
conv_layer_1_output = conv_layer(name='conv_layer_1', input=x_input, kernel_size=5, output_depth=2)

# shape (12, 12, 2)
pool_layer_1_output = pool_layer(name='pool_layer_1', input=conv_layer_1_output, kernel_size=2)

# shape (8, 8, 4)
conv_layer_2_output = conv_layer(name='conv_layer_2', input=pool_layer_1_output, kernel_size=5, output_depth=4)

# shape (4, 4, 4)
pool_layer_2_output = pool_layer(name='pool_layer_2', input=conv_layer_2_output, kernel_size=2)

# 变量初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
