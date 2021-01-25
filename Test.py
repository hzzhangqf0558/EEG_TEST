#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import roc_auc_score


"""
data: label+data
    1. csv
    2. conv1+BN+pooling1+conv2+BN+pooling2+conv3+BN+pooling3+2*fc+softmax
"""

"""
    dataset parameters
"""
EEG_DEPTH = 1
EEG_HEIGHT = 1
EEG_WIDTH = 5000
BATCH_SIZE = 64
TEST_SIZE = 2000
learning_rate = 2 * 1e-3
SEED = 66478  # Set to None for random seed.
log_dir = './logs'
model_path = "./model/models"

"""
    prepare the dataset
"""
TRAIN_DIR = os.listdir('./train')
for i in range(len(TRAIN_DIR)):
    TRAIN_DIR[i] = './train/' + TRAIN_DIR[i]
np.random.shuffle(TRAIN_DIR)

TEST_DIR = os.listdir('./test')
for i in range(len(TEST_DIR)):
    TEST_DIR[i] = './test/' + TEST_DIR[i]
np.random.shuffle(TEST_DIR)

def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32

def Calculate_Sp_Se_Auc(pred, actual_y):
    assert len(pred) == len(actual_y), "length is not equal"
    Accuracy = 0.0
    for i in range(len(pred)):
        if round(pred[i]) == actual_y[i]:
            Accuracy += 1
    Accuracy /= len(pred)
    TP = FP = TN = FN = 0
    for i in range(len(pred)):
        if round(pred[i]) == 1 and actual_y[i] == 1:
            TP += 1
        elif round(pred[i]) == 1 and actual_y[i] == 0:
            FP += 1
        elif round(pred[i]) == 0 and actual_y[i] == 1:
            FN += 1
        elif round(pred[i]) == 0 and actual_y[i] == 0:
            TN += 1
    # print ("TP=%d, FP=%d, TN=%d, FN=%d" % (TP, FP, TN, FN))
    Specificity = (TN + 0.0) / (TN + FP)
    Sensitivity = (TP + 0.0) / (TP + FN)
    AUC = roc_auc_score(actual_y, pred)

    return Specificity, Sensitivity, AUC

"""
    
"""
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

"""
    
"""
def my_input_fn(file_path, perform_shuffle=True, batch_size = BATCH_SIZE,repeat_count=-1):
   def decode_csv(line):
       L = []
       for i in range(5000):  # float point float
           L.append([0.])
       L.append([0.])  # label point float

       lists = tf.decode_csv(line, L)

       preprocess_op = tf.case({
           tf.equal(lists[0], tf.constant(0.0)): lambda: tf.constant([1, 0, 0, 0, 0]),
           tf.equal(lists[0], tf.constant(1.0)): lambda: tf.constant([0, 1, 0, 0, 0]),
           tf.equal(lists[0], tf.constant(2.0)): lambda: tf.constant([0, 0, 1, 0, 0]),
           tf.equal(lists[0], tf.constant(3.0)): lambda: tf.constant([0, 0, 0, 1, 0]),
           tf.equal(lists[0], tf.constant(4.0)): lambda: tf.constant([0, 0, 0, 0, 1])
       }, lambda: [tf.constant([1, 0, 0, 0, 0])], exclusive=True)
       single_sample = tf.stack(lists[1:len(lists)])  # sample

       single_temp = tf.reshape(single_sample, [EEG_DEPTH, EEG_HEIGHT, EEG_WIDTH])  # reshape for depth,height,width
       single_ecg = tf.transpose(single_temp, [1, 2, 0])  # reshape for heigth,width,depth
       d = single_ecg, preprocess_op
       return d
   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       .map(decode_csv))
   if perform_shuffle:
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(BATCH_SIZE)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_1x2(x, name):
    """max_pool_1x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='VALID', name = name)


def max_pool_1x13(x, name ):
    """max_pool_1x4 downsamples a feature map by 13X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 13, 1],
                          strides=[1, 1, 13, 1], padding='VALID', name = name)

def max_pool_1x3(x, name):
    """max_pool_1x4 downsamples a feature map by 3X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 3, 1],
                          strides=[1, 1, 3, 1], padding='VALID', name = name)


def max_pool_1x5(x, name):
    """max_pool_1x5 downsamples a feature map by 5X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],
                          strides=[1, 1, 5, 1], padding='VALID', name = name)


def max_pool_1x7(x, name):
    """max_pool_1x5 downsamples a feature map by 7X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 7, 1],
                          strides=[1, 1, 7, 1], padding='VALID', name = name)


def max_pool_1x9(x, name):
    """max_pool_1x5 downsamples a feature map by 9X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 9, 1],
                          strides=[1, 1, 9, 1], padding='VALID', name = name)


def weight_variable(shape, name):
    """generate weights variables randomly for filter"""
    initial = tf.truncated_normal(shape, stddev=0.1, seed=SEED, dtype=data_type())
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """generate bias for layer in network"""
    initial = tf.constant(1, shape=shape, dtype=data_type())
    return tf.Variable(initial, name = name)


"""batch normalization
"""
def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, 0.001)

"""
    网络结构
"""
def deep_cnn(x):

    with tf.name_scope("conv1"):
        # the first convolutional layer mapping 5 maps to 10 maps
        W_conv1 = weight_variable([1, 13, 1, 16], name = "W_conv1")
        variable_summaries(W_conv1)
        b_conv1 = bias_variable([16], name =  "b_conv1")
        variable_summaries(b_conv1)
        bn_conv1 = batch_norm(conv2d(x, W_conv1) + b_conv1, True)
        h_conv1 = tf.nn.relu(bn_conv1, name="h_conv1")
    with tf.name_scope("pool1"):
    # the first pooling layer for down sample by 1/2
        h_pool1 = max_pool_1x13(h_conv1, name="h_pool1")
    with tf.name_scope("conv2"):

        # the second convolutional layer mapping 10 maps to 20maps
        W_conv2 = weight_variable([1, 7, 16, 24], name="W_conv2")
        variable_summaries(W_conv2)
        b_conv2 = bias_variable([24], name = "b_conv2")
        variable_summaries(b_conv2)
        bn_conv2 = batch_norm(conv2d(h_pool1, W_conv2) + b_conv2, True)
        h_conv2 = tf.nn.relu(bn_conv2, name = "h_conv2")

    with tf.name_scope("pool2"):
        # the second pooling layer for down sample by 1/2
        h_pool2 = max_pool_1x9(h_conv2, name="h_pool2")

    with tf.name_scope("conv3"):
        # the third convolutional layer mapping 20maps to 40maps
        W_conv3 = weight_variable([1, 5, 24, 32], name="W_conv3")
        variable_summaries(W_conv3)
        b_conv3 = bias_variable([32], name="b_conv3")
        variable_summaries(b_conv3)
        bn_conv3 = batch_norm(conv2d(h_pool2, W_conv3) + b_conv3, True)
        h_conv3 = tf.nn.relu(bn_conv3, name="h_conv3")
    with tf.name_scope("pool3"):
        # the third pooling layer for down sample by 1/2
        h_pool3 = max_pool_1x5(h_conv3, name="pool3")

    with tf.name_scope("fc1"):
        ## for st
        # the first full connection layer
        W_fc1 = weight_variable([7*32, 24], name="W_fc1")
        b_fc1 = bias_variable([24], name="b_fc1")

        variable_summaries(W_fc1)
        variable_summaries(b_fc1)
        h_pool3_flat_st = tf.reshape(h_pool3, [-1, 7*32])
        h_fc1_st = tf.nn.relu(tf.matmul(h_pool3_flat_st, W_fc1) + b_fc1, name="h_fc1")

    with tf.name_scope("fc2"):
        # Map the  features to 1 classes, one for each digit
        W_fc2 = weight_variable([24, 5], name="W_fc2")
        b_fc2 = bias_variable([5], name="b_fc2")
        variable_summaries(W_fc2)
        variable_summaries(b_fc2)

        y_conv = tf.matmul(h_fc1_st, W_fc2) + b_fc2

    regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))

    return y_conv,regularizers, h_conv1, h_conv2,h_pool3

def main():

        with tf.name_scope("input"):

            x = tf.placeholder(data_type(), [None, EEG_HEIGHT, EEG_WIDTH, EEG_DEPTH], name="x") #数据输入

            y = tf.placeholder(data_type(), [None,5], name="y")

        with tf.name_scope("deep_cnn"):

            pre,regulations, h_conv1,h_conv2,h_pool3 = deep_cnn(x)

        with tf.name_scope("cross_entropy"):

            # alterable learning rate
            global_step = tf.Variable(0, dtype=tf.float32, name="global_step")

            learning_rate1 = tf.train.exponential_decay(
                    learning_rate,  # Base learning rate.
                    global_step * BATCH_SIZE,  # Current index into the dataset.
                    10000,  # Decay step.
                    0.9405,  # Decay rate.
                    staircase=True, name="learning_rate1")

            # Define loss and optimizer
            _loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits= pre), name="_loss")
            loss = _loss
            #loss = _loss+regulations
            # loss =  alpha1[0]* loss_st + alpha2[0] * loss_t + tf.nn.l2_loss(alpha1) + tf.nn.l2_loss(alpha2)
            tf.summary.scalar("loss", loss)

            optimizer = tf.train.AdamOptimizer(learning_rate1).minimize(loss, global_step=global_step, name="optimizer")

                # Evaluate model
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits=pre), 1), tf.arg_max(y, 1), name="correct_prediction")
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuary")
                    tf.summary.scalar("accuracy",accuracy)

        train_features, train_labels = my_input_fn(tf.convert_to_tensor(TRAIN_DIR, tf.string))

        test_features, test_labels = my_input_fn(tf.convert_to_tensor(TEST_DIR, tf.string), batch_size= 2000)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', tf.get_default_graph())
        val_writer = tf.summary.FileWriter(log_dir + '/val', tf.get_default_graph())

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())  # init all the variable
        saver = tf.train.Saver()

        sess = tf.Session()

        sess.run(init)

        train_count = 0
        print('start...')
        while True:
            train_count += 1
            _x, _y = sess.run([train_features, train_labels])
            summary, glb_stp, l, opt, lr, acc = sess.run([merged,global_step,_loss, optimizer, learning_rate1, accuracy],
                                                    feed_dict={x: _x,y:_y})
            train_writer.add_summary(summary,glb_stp)
            print(
                '%d times, leraning_rate: %.4f, train loss rate: %.4f, acc: %.4f' % (
                    train_count * BATCH_SIZE, lr, l, acc))
            #for test
            if train_count % 10 == 0:
                _x, _y = sess.run([test_features, test_labels])
                summary, glb_stp, l, lr, acc = sess.run(
                    [merged, global_step, _loss, learning_rate1, accuracy],
                    feed_dict={x: _x, y: _y})
                print(
                    'leraning_rate: %.4f, test loss rate: %.4f, acc: %.4f' % (
                        lr, l, acc))
                val_writer.add_summary(summary, glb_stp)
                saver.save(sess= sess, save_path= model_path)

        train_writer.close()
        val_writer.close()

if __name__ == '__main__':
    main()
