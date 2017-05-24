# -*- coding: utf-8 -*-

import tensorflow as tf
import time

a = tf.Variable(tf.random_normal([8192, 8192], stddev=0.35))
b = tf.Variable(tf.random_normal([8192, 8192], stddev=0.35))
c = tf.matmul(a, b)

sess = tf.Session()
sess.run([tf.initialize_all_variables()])

start_time = time.time()
sess.run(c)
print(time.time() - start_time)

