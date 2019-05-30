import tensorflow as tf

# 常量定义
BATCH_SIZE = 10
LEARNING_RATE = 0.001


def tensorboard_write():
    write = tf.summary.FileWriter('tensorboard/', tf.get_default_graph())
    write.close()


def generator(x_input):
    def get_deconv_kernel(name, input_depth, kernel_size, output_depth):
        # kernel shape (size, size, output dim, input dim)
        return tf.get_variable(name=name, shape=[kernel_size, kernel_size, output_depth, input_depth],
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))

    def get_deconv_bias(name, output_depth):
        return tf.get_variable(name=name, shape=[output_depth], initializer=tf.zeros_initializer())

    # 使用relu激活函数
    def deconv_layer(name, input, kernel_size, output_depth):
        input_shape = input.get_shape().as_list()
        output_shape = [input_shape[0], input_shape[1] + kernel_size - 1, input_shape[2] + kernel_size - 1,
                        output_depth]
        input_depth = input_shape[3]

        with tf.variable_scope(name):
            kernel = get_deconv_kernel('kernel', input_depth, kernel_size, output_depth)
            bias = get_deconv_bias('bias', output_depth)

        deconv = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=[1, 1, 1, 1], padding='VALID')
        return tf.nn.relu(tf.nn.bias_add(deconv, bias))

    # 定义generator结构
    deconv_layer_1_output = deconv_layer(name='deconv_layer_1', input=x_input, kernel_size=8, output_depth=2)

    deconv_layer_2_output = deconv_layer(name='deconv_layer_2', input=deconv_layer_1_output, kernel_size=15, output_depth=1)

    return deconv_layer_2_output


def discriminator(x_input):
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
    # 使用relu激活函数
    def full_connect_layer(name, input, nodes):
        input_shape = input.get_shape().as_list()

        with tf.variable_scope(name):
            weight = tf.get_variable('weight', shape=[input_shape[1], nodes],
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
            bias = tf.get_variable('bias', shape=[nodes], initializer=tf.zeros_initializer())

        fc = tf.matmul(input, weight)
        return tf.nn.relu(tf.nn.bias_add(fc, bias))

    conv_layer_1_output = conv_layer(name='conv_layer_1', input=x_input, kernel_size=5, output_depth=2)

    pool_layer_1_output = pool_layer(name='pool_layer_1', input=conv_layer_1_output, kernel_size=2)

    conv_layer_2_output = conv_layer(name='conv_layer_2', input=pool_layer_1_output, kernel_size=5, output_depth=4)

    pool_layer_2_output = pool_layer(name='pool_layer_2', input=conv_layer_2_output, kernel_size=2)

    fc_input = tf.reshape(pool_layer_2_output, [BATCH_SIZE, 64])

    fc_layer_1_output = full_connect_layer(name='fc_layer_1', input=fc_input, nodes=10)

    fc_layer_2_output = full_connect_layer(name='fc_layer_2', input=fc_layer_1_output, nodes=1)

    return fc_layer_2_output


with tf.name_scope('input'):
    g_input = tf.placeholder(dtype=tf.float32)
    d_input = tf.placeholder(dtype=tf.float32)

with tf.name_scope('output'):
    g_output = generator(g_input)
    d_output_real = discriminator(d_input)
    d_output_fake = discriminator(g_output)

with tf.name_scope('g_loss'):
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_fake, labels=tf.ones_like(d_output_fake)))

with tf.name_scope('d_loss'):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_real, labels=tf.ones_like(d_output_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_fake, labels=tf.zeros_like(d_output_fake)))
    d_loss = d_loss_fake + d_loss_real

with tf.name_scope('train'):
    g_train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(g_loss)
    d_train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(d_loss)

with tf.name_scope('init'):
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
