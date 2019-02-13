#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:16:16 2018

@author: pmacias
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X =  tf.placeholder(tf.float32, shape=(1, 2))
data = np.random.randn(1000,1,2)
d_draw =  np.zeros([1000,2])

dense_1 = tf.layers.dense(inputs=X, units=100, activation=tf.nn.relu)
dense_2 = tf.layers.dense(inputs=dense_1, units=100, activation=tf.nn.relu,kernel_initializer =tf.random_normal_initializer())
dense_3 = tf.layers.dense(inputs=dense_2, units=100, activation=tf.nn.relu)
dense_4 = tf.layers.dense(inputs=dense_3, units=100, activation=tf.nn.relu)
dense_5 = tf.layers.dense(inputs=dense_4, units=100, activation=tf.nn.relu)
dense_6 = tf.layers.dense(inputs=dense_5, units=100, activation=tf.nn.relu)
dense_7 = tf.layers.dense(inputs=dense_6, units=100, activation=tf.nn.relu)
dense_8 = tf.layers.dense(inputs=dense_7, units=100, activation=tf.nn.relu)
dense_9 = tf.layers.dense(inputs=dense_8, units=100, activation=tf.nn.relu)
dense_10 = tf.layers.dense(inputs=dense_9, units=100, activation=tf.nn.relu)
dense_11 = tf.layers.dense(inputs=dense_10, units=100, activation=tf.nn.relu)
dense_12 = tf.layers.dense(inputs=dense_11, units=100, activation=tf.nn.relu)


logits = tf.layers.dense(inputs=dense_12, units=2,activation=tf.nn.sigmoid)
probs =  tf.nn.softmax(logits, name="softmax_tensor")

init_op = tf.global_variables_initializer()
out = np.zeros([1000,2])
with tf.Session() as sess:
    
    for i in range(len(data)):
        
        
        sess.run(init_op)
        #print(sess.run(dense_1))
        out[i] = sess.run(probs, feed_dict = {X:data[i]} )
        d_draw[i] = data[i]
    plt.scatter(d_draw[:,0], out[:,0])
    plt.figure()
    plt.scatter(d_draw[:,1], out[:,1])
    plt.figure()
    plt.scatter(out[:,0], out[:,1], c = 'g')
    sess.close()
    