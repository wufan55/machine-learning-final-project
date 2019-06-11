import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('dataset/', one_hot=True)

# 常量定义
BATCH_SIZE = 128
TRAIN_STEP = 20001
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0001


def get_discriminator_batch(batch_size):
    x_batch, _ = mnist.train.next_batch(batch_size)
    return np.reshape(x_batch, [batch_size, 28, 28, 1])


def get_generator_batch(batch_size):
    return np.random.normal(0, 1., size=[batch_size, 100]).astype(np.float32)


# 全连接层
# 使用sigmoid激活函数
def full_connect_layer(name, input, nodes):
    input_shape = input.get_shape().as_list()

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weight', shape=[input_shape[1], nodes],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        bias = tf.get_variable('bias', shape=[nodes], initializer=tf.zeros_initializer())

    fc = tf.matmul(input, weight)
    return tf.nn.bias_add(fc, bias)


def generator(x_input):
    with tf.variable_scope('generator'):
        g_fc_layer_1_output = tf.nn.relu(full_connect_layer('generator_fc_layer_1', x_input, 128))

        g_fc_layer_2_output = tf.nn.relu(full_connect_layer('generator_fc_layer_2', g_fc_layer_1_output, 256))

        g_fc_layer_3_output = tf.nn.tanh(full_connect_layer('generator_fc_layer_3', g_fc_layer_2_output, 784))

    return tf.reshape(g_fc_layer_3_output, [x_input.get_shape().as_list()[0], 28, 28, 1])


def discriminator(x_input):
    with tf.variable_scope('discriminator'):
        fc_input = tf.reshape(x_input, [BATCH_SIZE, 784])

        fc_layer_1_output = tf.nn.relu(full_connect_layer(name='discriminator_fc_layer_1', input=fc_input, nodes=128))

        fc_layer_2_output = tf.nn.sigmoid(full_connect_layer(name='discriminator_fc_layer_2', input=fc_layer_1_output, nodes=1))

        # fc_layer_3_output = tf.nn.sigmoid(full_connect_layer('discriminator_fc_layer_3', fc_layer_2_output, 1))

    return fc_layer_2_output


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
    d_loss = -tf.reduce_mean(1.3*tf.log(d_output_real) + 0.7*tf.log(1. - d_output_fake))

with tf.name_scope('GAN'):
    g_train = tf.train.AdamOptimizer(learning_rate=G_LEARNING_RATE).minimize(g_loss, var_list=slim.get_variables('generator'))
    d_train = tf.train.AdamOptimizer(learning_rate=D_LEARNING_RATE).minimize(d_loss, var_list=slim.get_variables('discriminator'))

with tf.name_scope('summary'):
    tf.summary.image('generator_output', g_output, 10)
    tf.summary.scalar('d_output_fake', tf.reduce_mean(d_output_fake))
    tf.summary.scalar('generator_loss', g_loss)
    tf.summary.scalar('discriminator_loss', d_loss)

    merged = tf.summary.merge_all()

with tf.name_scope('init'):
    init = tf.global_variables_initializer()

with tf.name_scope('save'):
    saver = tf.train.Saver()

with tf.Session() as sess:
    # sess.run(init)
    ckpt = tf.train.get_checkpoint_state('/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    writer = tf.summary.FileWriter('tensorboard/GAN', sess.graph)

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
