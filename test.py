# import tensorflow as tf
#
# a = tf.ones([1, 3, 3, 1])
# w = tf.ones([3, 3, 3, 1])
# b = tf.nn.conv2d_transpose(a, w, output_shape=[1, 5, 5, 3], strides=[1, 1, 1, 1], padding='VALID')
#
# with tf.Session() as sess:
#     t = sess.run(b)
#     print(t)
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("GAN/dataset/", one_hot=True)

# print("Training data size: ", mnist.train.num_examples)
a = np.array(mnist.train.images[0])
print(a)
