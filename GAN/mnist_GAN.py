import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('dataset/', one_hot=True)

# 常量定义
BATCH_SIZE = 64
TRAIN_STEP = 10001
LEARNING_RATE = 0.0001


def get_discriminator_batch(batch_size):
    x_batch, _ = mnist.train.next_batch(batch_size)
    return np.reshape(x_batch, [batch_size, 784])


def get_generator_batch(batch_size):
    return np.random.uniform(-1, 1, size=[batch_size, 100]).astype(np.float32)


def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.zeros_initializer())


def generator(input):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        w1 = weight_var([100, 128], 'w1')
        b1 = bias_var([128], 'b1')
        generator_layer1 = tf.nn.relu(tf.matmul(input, w1) + b1)

        w2 = weight_var([128, 784], 'w2')
        b2 = bias_var([784], 'b2')
        generator_layer2 = tf.nn.relu(tf.matmul(generator_layer1, w2) + b2)

    return generator_layer2


def discriminator(input):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        w1 = weight_var([784, 128], 'w1')
        b1 = bias_var([128], 'b1')
        discriminator_layer1 = tf.nn.relu(tf.matmul(input, w1) + b1)

        w2 = weight_var([128, 1], 'w2')
        b2 = bias_var([1], 'b2')
        discriminator_layer2 = tf.nn.sigmoid(tf.matmul(discriminator_layer1, w2) + b2)

    return discriminator_layer2


with tf.name_scope('input'):
    D_input = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    G_input = tf.placeholder(tf.float32, shape=[None, 100], name='z')

with tf.name_scope('output'):
    G_output = generator(G_input)
    D_output_real = discriminator(D_input)
    D_output_fake = discriminator(G_output)

with tf.name_scope('var'):
    t_vars = tf.trainable_variables()
    # variable_names = [v.name for v in tf.trainable_variables()]
    # print(variable_names)
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

with tf.name_scope('loss'):
    D_loss = -tf.reduce_mean(tf.log(D_output_real) + tf.log(1. - D_output_fake))
    G_loss = -tf.reduce_mean(tf.log(D_output_fake))

with tf.name_scope('train'):
    D_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(D_loss, var_list=d_vars)
    G_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(G_loss, var_list=g_vars)

with tf.name_scope('summary'):
    tf.summary.image('generator_output', tf.reshape(G_output, [BATCH_SIZE, 28, 28, 1]), 10)
    tf.summary.scalar('discriminator_output_fake', tf.reduce_mean(D_output_fake))
    tf.summary.scalar('discriminator_loss_fake', -tf.reduce_mean(tf.log(1. - D_output_fake)))
    tf.summary.scalar('discriminator_loss_real', -tf.reduce_mean(tf.log(D_output_real)))
    tf.summary.scalar('discriminator_loss', D_loss)
    tf.summary.scalar('generator_loss', G_loss)

    merged = tf.summary.merge_all()

with tf.name_scope('init'):
    init = tf.global_variables_initializer()

with tf.name_scope('save'):
    saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data/GAN')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    writer = tf.summary.FileWriter('tensorboard/GAN', sess.graph)

    for i in range(TRAIN_STEP):
        g_batch = get_generator_batch(BATCH_SIZE)
        d_batch = get_discriminator_batch(BATCH_SIZE)

        # 训练
        dl, _ = sess.run([D_loss, D_optimizer], feed_dict={G_input: g_batch, D_input: d_batch})
        gl, _ = sess.run([G_loss, G_optimizer], feed_dict={G_input: g_batch})

        if i % 50 == 0:
            print('After training %d round, generator_loss: %f' % (i, gl))

        if i != 0 and i % 200 == 0:
            saver.save(sess, '/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data/GAN/log.ckpt')

        if i % 10 == 0:
            summary = sess.run(merged, feed_dict={G_input: g_batch, D_input: d_batch})
            writer.add_summary(summary, i)

    writer.close()
