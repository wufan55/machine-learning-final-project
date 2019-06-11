import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('dataset/', one_hot=True)

# 常量定义
BATCH_SIZE = 64
TRAIN_STEP = 5001
LEARNING_RATE = 0.0001


def get_discriminator_batch(batch_size):
    x_batch, _ = mnist.train.next_batch(batch_size)
    return np.reshape(x_batch, [batch_size, 28, 28, 1])


def get_generator_batch(batch_size):
    return np.random.uniform(-1, 1, size=[batch_size, 1, 1, 100]).astype(np.float32)


def generator(input, isTrain=True):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        # 1st hidden layer
        # shape (2, 2, 1024)
        conv1 = tf.layers.conv2d_transpose(input, 1024, [2, 2], strides=(1, 1), padding='valid')
        relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=isTrain))

        # 2nd hidden layer
        # shape (4, 4, 512)
        conv2 = tf.layers.conv2d_transpose(relu1, 512, [4, 4], strides=(2, 2), padding='same')
        relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=isTrain))

        # 3rd hidden layer
        # shape (7, 7, 256)
        conv3 = tf.layers.conv2d_transpose(relu2, 256, [4, 4], strides=(1, 1), padding='valid')
        relu3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=isTrain))

        # 4th hidden layer
        # shape (14, 14, 128)
        conv4 = tf.layers.conv2d_transpose(relu3, 128, [4, 4], strides=(2, 2), padding='same')
        relu4 = tf.nn.relu(tf.layers.batch_normalization(conv4, training=isTrain))

        # output layer
        # shape (28, 28, 1)
        conv5 = tf.layers.conv2d_transpose(relu4, 1, [4, 4], strides=(2, 2), padding='same')
        output = tf.nn.tanh(conv5)

        return output


def discriminator(input, isTrain=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        # 1st hidden layer
        # shape (14, 14, 128)
        conv1 = tf.layers.conv2d(input, 128, [4, 4], strides=(2, 2), padding='same')
        relu1 = tf.nn.relu(conv1)

        # 2nd hidden layer
        # shape (7, 7, 256)
        conv2 = tf.layers.conv2d(relu1, 256, [4, 4], strides=(2, 2), padding='same')
        relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=isTrain))

        # 3rd hidden layer
        # shape (4, 4, 512)
        conv3 = tf.layers.conv2d(relu2, 512, [4, 4], strides=(2, 2), padding='same')
        relu3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=isTrain))

        # 4th hidden layer
        # shape (2, 2, 1024)
        conv4 = tf.layers.conv2d(relu3, 1024, [4, 4], strides=(2, 2), padding='same')
        relu4 = tf.nn.relu(tf.layers.batch_normalization(conv4, training=isTrain))

        # output layer
        # shape (1, 1, 1)
        conv5 = tf.layers.conv2d(relu4, 1, [2, 2], strides=(1, 1), padding='valid')
        output = tf.nn.sigmoid(conv5)

        return tf.reshape(output, [BATCH_SIZE, 1])


with tf.name_scope('input'):
    D_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28, 28, 1], name='x')
    G_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1, 1, 100], name='z')

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
    tf.summary.image('generator_output', G_output, 10)
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
    ckpt = tf.train.get_checkpoint_state('/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data/DCGAN')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    writer = tf.summary.FileWriter('tensorboard/DCGAN', sess.graph)

    for i in range(TRAIN_STEP):
        g_batch = get_generator_batch(BATCH_SIZE)
        d_batch = get_discriminator_batch(BATCH_SIZE)

        # 训练
        dl, _ = sess.run([D_loss, D_optimizer], feed_dict={G_input: g_batch, D_input: d_batch})
        gl, _ = sess.run([G_loss, G_optimizer], feed_dict={G_input: g_batch})

        if i % 50 == 0:
            print('After training %d round, generator_loss: %f' % (i, gl))

        if i != 0 and i % 200 == 0:
            saver.save(sess, '/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data/DCGAN/log.ckpt')

        if i % 10 == 0:
            summary = sess.run(merged, feed_dict={G_input: g_batch, D_input: d_batch})
            writer.add_summary(summary, i)

    writer.close()
