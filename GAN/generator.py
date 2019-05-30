import tensorflow as tf

# 常量定义
BATCH_SIZE = 10


def tensorboard_write():
    write = tf.summary.FileWriter('tensorboard/', tf.get_default_graph())
    write.close()


def get_deconv_kernel(name, input_depth, kernel_size, output_depth):
    # kernel shape (size, size, output dim, input dim)
    return tf.get_variable(name=name, shape=[kernel_size, kernel_size, output_depth, input_depth],
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))


def get_deconv_bias(name, output_depth):
    return tf.get_variable(name=name, shape=[output_depth], initializer=tf.zeros_initializer())


# 使用relu激活函数
def deconv_layer(name, input, kernel_size, output_depth):
    input_shape = input.get_shape().as_list()
    output_shape = [input_shape[0], input_shape[1]+kernel_size-1, input_shape[2]+kernel_size-1, output_depth]
    input_depth = input_shape[3]

    with tf.variable_scope(name):
        kernel = get_deconv_kernel('kernel', input_depth, kernel_size, output_depth)
        bias = get_deconv_bias('bias', output_depth)

    deconv = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(deconv, bias))


# 定义网络结构
# shape (7, 7, 4)
with tf.name_scope('layer_1'):
    x_input = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 7, 7, 4), name='x_input')

# shape (14, 14, 2)
with tf.name_scope('layer_2'):
    deconv_layer_1_output = deconv_layer(name='deconv_layer_1', input=x_input, kernel_size=8, output_depth=2)

# shape (28, 28, 1)
with tf.name_scope('layer_3'):
    deconv_layer_2_output = deconv_layer(name='deconv_layer_2', input=deconv_layer_1_output, kernel_size=15, output_depth=1)

# shape (28, 28, 1)
y_output = deconv_layer_2_output

# 变量初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init, tensorboard_write())
