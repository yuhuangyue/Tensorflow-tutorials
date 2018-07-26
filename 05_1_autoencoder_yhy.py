#!/usr/bin/env python
"""
自己写的一个autoencoder
用卷积神经网络写的
层次是这样的 1（灰度图）-- 32 -- 64 -- 128 -(中间可以提取出一个特征向量)- 128 -- 64 -- 32 --1

06那个教程里面给输入图片加了noise
正规做法是应该这样
保证模型能适应各种图片

***
特别注意这里的卷积和解卷积的过程

***
另外这个里面的vis函数可以借鉴一下

"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


import matplotlib # to plot images
# Force matplotlib to not use any X-server backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


batch_size = 128
test_size = 256

def init_weights(shape):

    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model_encode(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):


    with tf.name_scope("encode1"):
        l1 = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
        l1 = tf.nn.dropout(l1, p_keep_conv)

    with tf.name_scope("encode2"):
        l2 = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.dropout(l2, p_keep_conv)

    with tf.name_scope("encode3"):
        l3 = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.dropout(l3, p_keep_conv)

    #l4 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 7*7*128)

    return l3

def decov (X , w,output_shape):

    deconv = tf.nn.conv2d_transpose(X, w, output_shape=output_shape,
                                    strides=[1, 1, 1, 1],padding="SAME")

    return deconv

def model_decode (X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

   # X = tf.reshape(X , [-1,7,7,128])

    with tf.name_scope("decode1"):
        l1 = tf.nn.relu(decov(X, w,output_shape=[batch_size, 28, 28, 64]))
        l1 = tf.nn.dropout(l1, p_keep_conv)

    with tf.name_scope("decode2"):
        l2 = tf.nn.relu(decov(l1,w2, output_shape=[batch_size, 28, 28, 32]))
        l2 = tf.nn.dropout(l2, p_keep_conv)

    with tf.name_scope("decode1"):
        l3 = tf.nn.relu(decov(l2, w3,output_shape=[batch_size, 28, 28, 1]))
        #l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
        l3 = tf.nn.dropout(l3, p_keep_conv)


    return l3





mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])

Y = tf.placeholder("float", [None, 10])

with tf.name_scope("weight"):
#encode  变成一维的向量
    w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
    w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
    w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
    w4 = init_weights([128 * 7 * 7, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
    w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

#decode
    w4_ = init_weights([128 * 7 * 7, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
    w_ = init_weights([3, 3, 64, 128])       # 3x3x1 conv, 32 outputs
    w2_ = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
    w3_ = init_weights([3, 3, 1, 32])    # 3x3x32 conv, 128 outputs
    w_o_ = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)


p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x_ = model_encode(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
py_x = model_decode(py_x_, w_, w2_, w3_, w4_, w_o_, p_keep_conv, p_keep_hidden)

cost = tf.reduce_sum(tf.pow(py_x - X, 2))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
test_op = py_x


def vis(images, save_name):
    dim = images.shape[0]
    n_image_rows = int(np.ceil(np.sqrt(dim)))
    n_image_cols = int(np.ceil(dim * 1.0/n_image_rows))
    gs = gridspec.GridSpec(n_image_rows,n_image_cols,top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    for g,count in zip(gs,range(int(dim))):
        ax = plt.subplot(g)
        ax.imshow(images[count,:].reshape((28,28)))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(save_name + '_vis.png')



# Launch the graph in a session
with tf.Session() as sess:

    writer = tf.summary.FileWriter("./logs/05", sess.graph) # for 1.0
    merged = tf.summary.merge_all()

    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, 1000, batch_size),
                             range(batch_size, 1000+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:128]
        te_X_image = teX[test_indices]


        test_image = sess.run(test_op, feed_dict={X: te_X_image,
                                                  p_keep_conv: 1.0,
                                                  p_keep_hidden: 1.0})

        vis(test_image, 'pred')
        summary_op = tf.summary.image("image", test_image, 10)
        summary = sess.run(summary_op)
        writer.add_summary(summary)
        # 关闭
        writer.close()
        print(i, " done!")

