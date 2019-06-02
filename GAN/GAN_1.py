import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('dataset/', one_hot=True)

# 常量定义
BATCH_SIZE = 128
TRAIN_STEP = 5001
G_LEARNING_RATE = 0.002
D_LEARNING_RATE = 0.005


def get_discriminator_batch(batch_size):
    x_batch, _ = mnist.train.next_batch(batch_size)
    return np.reshape(x_batch, [batch_size, 28, 28, 1])


def get_generator_batch(batch_size):
    return np.random.normal(0, 1, size=[batch_size, 100]).astype(np.float32)


def generator(x_input):
    with tf.variable_scope('generator'):
        def full_connect_layer(name, input, nodes):
            input_shape = input.get_shape().as_list()

            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                weight = tf.get_variable('weight', shape=[input_shape[1], nodes],
                                         initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
                bias = tf.get_variable('bias', shape=[nodes], initializer=tf.zeros_initializer())

            fc = tf.matmul(input, weight)
            return tf.nn.bias_add(fc, bias)

        g_fc_layer_1_output = tf.nn.relu(full_connect_layer('generator_fc_layer_1', x_input, 256))

        g_fc_layer_2_output = tf.nn.relu(full_connect_layer('generator_fc_layer_2', g_fc_layer_1_output, 512))

        g_fc_layer_3_output = tf.nn.tanh(full_connect_layer('generator_fc_layer_3', g_fc_layer_2_output, 784))

    return tf.reshape(g_fc_layer_3_output, [x_input.get_shape().as_list()[0], 28, 28, 1])


def discriminator(x_input):
    with tf.variable_scope('discriminator'):
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

            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
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

            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                weight = tf.get_variable('weight', shape=[input_shape[1], nodes],
                                         initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
                bias = tf.get_variable('bias', shape=[nodes], initializer=tf.zeros_initializer())

            fc = tf.matmul(input, weight)
            return tf.nn.sigmoid(tf.nn.bias_add(fc, bias))

        conv_layer_1_output = conv_layer(name='conv_layer_1', input=x_input, kernel_size=5, output_depth=2)

        pool_layer_1_output = pool_layer(name='pool_layer_1', input=conv_layer_1_output, kernel_size=2)

        conv_layer_2_output = conv_layer(name='conv_layer_2', input=pool_layer_1_output, kernel_size=5, output_depth=4)

        pool_layer_2_output = pool_layer(name='pool_layer_2', input=conv_layer_2_output, kernel_size=2)

        fc_input = tf.reshape(pool_layer_2_output, [BATCH_SIZE, 64])

        fc_layer_1_output = full_connect_layer(name='fc_layer_1', input=fc_input, nodes=1)

    return fc_layer_1_output


with tf.name_scope('input'):
    g_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 100])
    d_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 28, 28, 1])

with tf.name_scope('output'):
    g_output = generator(g_input)
    d_output_real = discriminator(d_input)
    d_output_fake = discriminator(g_output)

with tf.name_scope('g_loss'):
    g_loss = -tf.reduce_mean(tf.log(d_output_fake))

with tf.name_scope('d_loss'):
    d_loss = -tf.reduce_mean(tf.log(d_output_real) + tf.log(1. - d_output_fake))

with tf.name_scope('train'):
    g_train = tf.train.AdamOptimizer(learning_rate=G_LEARNING_RATE).minimize(g_loss, var_list=slim.get_variables('generator'))
    d_train = tf.train.AdamOptimizer(learning_rate=D_LEARNING_RATE).minimize(d_loss, var_list=slim.get_variables('discriminator'))

with tf.name_scope('summary'):
    tf.summary.image('generator_output', g_output, 10)
    tf.summary.histogram('discriminator_output_real', d_output_real)
    tf.summary.histogram('discriminator_output_fake', d_output_fake)
    tf.summary.scalar('d_output_fake', tf.reduce_mean(d_output_fake))
    tf.summary.scalar('generator_loss', g_loss)
    tf.summary.scalar('discriminator_loss', d_loss)

    merged = tf.summary.merge_all()

with tf.name_scope('init'):
    init = tf.global_variables_initializer()

with tf.name_scope('save'):
    saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # ckpt = tf.train.get_checkpoint_state(
    #     '/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)

    writer = tf.summary.FileWriter('tensorboard/train', sess.graph)

    for i in range(TRAIN_STEP):
        g_batch = get_generator_batch(BATCH_SIZE)
        d_batch = get_discriminator_batch(BATCH_SIZE)

        # 训练
        dl, _ = sess.run([d_loss, d_train], feed_dict={g_input: g_batch, d_input: d_batch})
        gl, _ = sess.run([g_loss, g_train], feed_dict={g_input: g_batch})

        if i % 50 == 0:
            print('After training %d round, generator_loss: %f' % (i, gl))

        if i != 0 and i % 200 == 0:
            saver.save(sess, '/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data/log.ckpt')

        if i % 10 == 0:
            summary = sess.run(merged, feed_dict={g_input: g_batch, d_input: d_batch})
            writer.add_summary(summary, i)

    writer.close()
