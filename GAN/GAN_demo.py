import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 64
TRAIN_STEP = 15001

mnist = input_data.read_data_sets('dataset/', one_hot=True)


def get_discriminator_batch(batch_size):
    x_batch, _ = mnist.train.next_batch(batch_size)
    return np.reshape(x_batch, [batch_size, 784])


def get_generator_batch(batch_size):
    return np.random.uniform(-1, 1, size=[batch_size, 100]).astype(np.float32)


def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))


# discriminater net

X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = weight_var([784, 128], 'D_W1')
D_b1 = bias_var([128], 'D_b1')

D_W2 = weight_var([128, 1], 'D_W2')
D_b2 = bias_var([1], 'D_b2')


theta_D = [D_W1, D_W2, D_b1, D_b2]


# generator net

Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = weight_var([100, 128], 'G_W1')
G_b1 = bias_var([128], 'G_B1')

G_W2 = weight_var([128, 784], 'G_W2')
G_b2 = bias_var([784], 'G_B2')

theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

variable_names = [v.name for v in tf.trainable_variables()]
print(variable_names)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

with tf.name_scope('summary'):
    tf.summary.image('generator_output', tf.reshape(G_sample, [BATCH_SIZE, 28, 28, 1]), 10)
    tf.summary.scalar('discriminator_output_fake', tf.reduce_mean(D_fake))
    tf.summary.scalar('discriminator_loss_fake', -tf.reduce_mean(tf.log(1. - D_fake)))
    tf.summary.scalar('discriminator_loss_real', -tf.reduce_mean(tf.log(D_real)))
    tf.summary.scalar('discriminator_loss', D_loss)
    tf.summary.scalar('generator_loss', G_loss)

    merged = tf.summary.merge_all()

with tf.name_scope('init'):
    init = tf.global_variables_initializer()

with tf.name_scope('save'):
    saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    writer = tf.summary.FileWriter('tensorboard/train', sess.graph)

    for i in range(TRAIN_STEP):
        g_batch = get_generator_batch(BATCH_SIZE)
        d_batch = get_discriminator_batch(BATCH_SIZE)

        # 训练
        dl, _ = sess.run([D_loss, D_optimizer], feed_dict={Z: g_batch, X: d_batch})
        gl, _ = sess.run([G_loss, G_optimizer], feed_dict={Z: g_batch})

        if i % 50 == 0:
            print('After training %d round, generator_loss: %f' % (i, gl))

        if i != 0 and i % 200 == 0:
            saver.save(sess, '/Users/fan/Desktop/python/tensorflow/machine-learning-final-project/GAN/net_data/log.ckpt')

        if i % 10 == 0:
            summary = sess.run(merged, feed_dict={Z: g_batch, X: d_batch})
            writer.add_summary(summary, i)

    writer.close()
