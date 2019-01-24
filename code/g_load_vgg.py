# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import datetime
import matplotlib.pyplot as plt
from readthyroid import *

# 874 1840
img_channels = 1
iterations = 40
batch_size = 46
total_epoch = 150
test_iterations = 59
test_size = 46
weight_decay = 0.0003
dropout_rate = 0.8
momentum_rate = 0.9
# log_save_path = './log'
# model_save_path = './model_z/Attention_VGG/'
# exp_name = 'Attention_VGG'
num = 100
alpha = 1.5
beta = 1.0

def TP_def(c, y):
    c = np.array(c)
    y = np.array(y)
    # print(c.shape, y.shape)
    TP, FP, FN, TN = 0,0,0,0
    for i in range(len(y)):
        if c[i]:
            if y[i,0]:
                TP += 1
            else:
                TN += 1
        else:
            if y[i,0]:
                FN += 1
            else:
                FP += 1
    return TP, FP, FN, TN

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def conv2d(input, in_features, out_features, kernel_size, strides=1, with_bias=True):
#     W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
#     conv = tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding='SAME')
#     if with_bias:
#         return conv + bias_variable([ out_features ])
#     return conv

def conv2d(input, in_features, out_features, kernel_size, strides=1, with_bias=True):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.relu(batch_norm(tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding='SAME')))
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv

def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)

def avg_pool(input, k_size=1, stride=1, padding = 'SAME', name=None):
    return tf.nn.avg_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding=padding, name=name)

def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)

def index_max(con):
    # con = input
    print(con.shape[0])
    max = np.zeros(23)
    for i in range(23):
        max[i] = tf.reduce_max(con[i])
    return max


def pre_res_block(input, in_channel, channel_1, channel_2, channel_3, kernel_size_1=1, kernel_size_2=3, kernel_size_3=1, small=False):
    x = tf.nn.relu(batch_norm(input))
    x = tf.nn.dropout(x, keep_prob)
    conv1_1 = conv2d(x, in_channel, channel_1, kernel_size_1)
    conv1_1bn = tf.nn.relu(batch_norm(conv1_1))
    conv1_1bn = tf.nn.dropout(conv1_1bn, keep_prob)
    if small:
        conv1_2 = conv2d(conv1_1bn, channel_1, channel_2, kernel_size_2, 2)
        conv1_2bn = tf.nn.relu(batch_norm(conv1_2))
        conv1_2bn = tf.nn.dropout(conv1_2bn, keep_prob)
        conv1_3 = conv2d(conv1_2bn, channel_2, channel_3, kernel_size_3)
        conv2_1 = conv2d(x, in_channel, channel_3, kernel_size_3, 2)
    else:
        conv1_2 = conv2d(conv1_1bn, channel_1, channel_2, kernel_size_2)
        conv1_2bn = tf.nn.relu(batch_norm(conv1_2))
        conv1_2bn = tf.nn.dropout(conv1_2bn, keep_prob)
        conv1_3 = conv2d(conv1_2bn, channel_2, channel_3, kernel_size_3)
        conv2_1 = conv2d(x, in_channel, channel_3, kernel_size_3)
    res_output = tf.add(conv2_1, conv1_3)
    return res_output

def res_block(input, in_channel, channel_1, channel_2, channel_3, kernel_size_1=1, kernel_size_2=3, kernel_size_3=1, name=None):
    x = tf.nn.relu(batch_norm(input))
    x = tf.nn.dropout(x, keep_prob)
    conv1_1 = conv2d(x, in_channel, channel_1, kernel_size_1)
    conv1_1bn = tf.nn.relu(batch_norm(conv1_1))
    conv1_1bn = tf.nn.dropout(conv1_1bn, keep_prob)
    conv1_2 = conv2d(conv1_1bn, channel_1, channel_2, kernel_size_2)
    conv1_2bn = tf.nn.relu(conv1_2)
    conv1_2bn = tf.nn.dropout(conv1_2bn, keep_prob)
    conv1_3 = conv2d(conv1_2bn, channel_2, channel_3, kernel_size_3)
    res_output = tf.add(input, conv1_3, name)
    return res_output


def learning_rate_schedule(epoch_num):
    if epoch_num < 40:
        return 0.1
    elif epoch_num < 60:
        return 0.01
    elif epoch_num < 90:
        return 0.001
    elif epoch_num == 100:
        return 0.005
    elif epoch_num < 120:
        return 0.0001
    else:
        return 0.00001

def feature_cam(input, p=200.0, color_map=cv2.COLORMAP_JET):
    output = np.zeros((input.shape[0], input.shape[1], input.shape[2], 3))
    for j in range(input.shape[0]):
        sum = np.zeros((1, input.shape[1], input.shape[2], 1))
        for i in range(input.shape[3]):
            sum += input[j:j + 1, :, :, i:i + 1]
        img = sum - np.min(sum)
        img /= np.max(img)
        cam = np.uint8(p * img)
        # temp = np.reshape(cam, [input.shape[1], input.shape[2], 1])
        temp = cv2.applyColorMap(cam[0], color_map)
        # print('%d: temp = '%j, temp[10,10,2])
        output[j] += temp
        # print('output = ', output[j,10,10,2])
    return output

def run_testing(sess):
    # id = 1
    acc = 0.0
    loss = 0.0
    pre_index = 0
    ep_y_, correct_p, out = [], [], []
    for it in range(test_iterations):
        batch_x = test_x[pre_index:pre_index + test_size]
        batch_y = test_y[pre_index:pre_index + test_size]
        pre_index = pre_index + test_size

        loss_, acc_, ep_output_aa, out_f, ep_y_1, correct_p1 = sess.run([cross_entropy, accuracy, output_aa, output_f, y_,
                                                                   correct_prediction],
                                                                  feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0,
                                                                             train_flag: False})
        print(out_f.shape)
        if it == 0:
            out = ep_output_aa
            for i in range(10):
                for j in range(512):
                    temp = (out_f[i, :, :, j]-np.min(out_f[i, :, :, j]))/(np.max(out_f[i, :, :, j])-np.min(out_f[i, :, :, j]))
                    temp = cv2.resize(temp, (128, 128), interpolation=cv2.INTER_CUBIC)
                    temp = np.uint8(temp*255)
                    # print(i, np.max(temp), np.min(temp))
                    cv2.imwrite('./temp/whale/%d/%d.png'%(i, j), 0.7 * cv2.applyColorMap(temp, cv2.COLORMAP_JET) + 0.3 * test_x[i] * 255)
        else:
            out = np.concatenate((out, ep_output_aa), axis=0)
            print(it, out.shape)
        # loss_, acc_, ep_output, ep_y_1, correct_p1, \
        # output_1_, output_2_, output_3_, output_4_ = sess.run([cross_entropy, accuracy, output, y_,
        #                                             correct_prediction, output_1, output_2, output_3, output_4],
        #                                             feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0,
        #                                                                      train_flag: False})
        # loss_, acc_, ep_output, ep_y_1, correct_p1, \
        # a1_t, a1_m, a1, a2_t, a2_m, a2, a3_t, a3_m, a3, out_ = sess.run([cross_entropy, accuracy, output, y_,
        #                                                                  correct_prediction, a1_trunk, a1_mask, att_1,
        #                                                                  a2_trunk, a2_mask, att_2, a3_trunk, a3_mask,
        #                                                                  att_3, out],
        #                                                                 feed_dict={x: batch_x, y_: batch_y,
        #                                                                            keep_prob: 1.0,
        #                                                                            train_flag: False})

        loss += loss_ / float(test_iterations)
        acc += acc_ / float(test_iterations)

        ep_y_.extend(ep_y_1)
        correct_p.extend(correct_p1)

        # cam_img_a1_t = feature_cam(a1_t, 200.0)
        # cam_img_a1_m = feature_cam(a1_m, 200.0, cv2.COLORMAP_JET)
        # cam_img_a1 = feature_cam(a1, 200.0)
        #
        # cam_img_a2_t = feature_cam(a2_t, 200.0)
        # cam_img_a2_m = feature_cam(a2_m, 200.0, cv2.COLORMAP_JET)
        # cam_img_a2 = feature_cam(a2, 200.0)
        #
        # cam_img_a3_t = feature_cam(a3_t, 200.0)
        # cam_img_a3_m = feature_cam(a3_m, 200.0, cv2.COLORMAP_JET)
        # cam_img_a3 = feature_cam(a3, 200.0)
        #
        # cam_o = feature_cam(out_, 200.0)


        # half = a3_t * a3_m + a3_t
        # print("a3_t.shape, a3_m.shape, half.shape", a3_t.shape, a3_m.shape, half.shape)
        # half = np.reshape(half, [14, 14, 1])
        # half_ = np.uint8(200 * half)
        # temp = cv2.applyColorMap(half_, cv2.COLORMAP_JET)

        # sub = a3 - a3_t

        # print('mean::::', np.mean(a3_t), np.mean(a3), np.mean(a3_m))
        # print("np.max(sub), np.mean(sub)", np.max(sub), np.mean(sub), np.sum(sub))


    # ep_y_ = np.array(ep_y_).reshape(-1, 2)
    # correct_p = np.array(correct_p).reshape(-1))
    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
                                tf.Summary.Value(tag="test_accuracy", simple_value=acc)])

#     return acc, loss, summary, ep_y_, correct_p, cam_img_a1_t, cam_img_a1_m, cam_img_a1,\
# cam_img_a2_t, cam_img_a2_m, cam_img_a2, cam_img_a3_t, cam_img_a3_m, cam_img_a3, cam_o
#     return acc, loss, summary, ep_y_, correct_p, cam_o1, cam_o2, cam_o3, cam_o4
    return acc, loss, summary, ep_y_, correct_p, out


def cam_write(corg, img, po, pc, name):
    cam_img = cv2.resize(img, (112, 112))
    res = corg * po + cam_img * pc
    cv2.imwrite('/home/zrx/zrx/pyworkplace/output/Attention/' + name, res)


if __name__ == '__main__':
    org_x = np.load('img_100.npy')
    org_y = np.load('lab_100.npy')
    org_x = org_x[:, :, :, np.newaxis]
    org_x = segdata_preprocessing(org_x)
    # org_x, org_y = prepare_compdata()
    # org_x, org_y = prepare_compdata()
    acc_te, loss_te = [], []
    acc = 0.0

    graph = tf.Graph()
    sess = tf.Session(graph = graph)
    with graph.as_default():
        # define placeholder x, y_ , keep_prob, learning_rate
        x = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='x')
        y_ = tf.placeholder(tf.float32, [None, class_num], name='y_')
        keep_prob = tf.placeholder(tf.float32, name='keep_p')
        learning_rate = tf.placeholder(tf.float32, name='learning_r')
        train_flag = tf.placeholder(tf.bool, name='train_f')

        # build_network
        # 112*112
        output = conv2d(x, 1, 64, 3)
        output = conv2d(output, 64, 64, 3)
        output = max_pool(output, 2, 2, "pool1")
        # 56*56
        output = conv2d(output, 64, 128, 3)
        output = conv2d(output, 128, 128, 3)
        output = max_pool(output, 2, 2, "pool2")
        # 28*28
        output = conv2d(output, 128, 256, 3)
        output = conv2d(output, 256, 256, 3)
        output = conv2d(output, 256, 256, 3)
        output = conv2d(output, 256, 256, 3)
        output = max_pool(output, 2, 2, "pool3")
        # 14*14
        output = conv2d(output, 256, 512, 3)
        output = conv2d(output, 512, 512, 3)
        output = conv2d(output, 512, 512, 3)
        output = conv2d(output, 512, 512, 3)
        output = max_pool(output, 2, 2, 'pool4')
        # 7*7
        output = conv2d(output, 512, 512, 3)
        output = conv2d(output, 512, 512, 3)
        output = conv2d(output, 512, 512, 3)
        output_f = conv2d(output, 512, 512, 3)
        # output = max_pool(output, 2, 2, 'pool5')

        # output = tf.contrib.layers.flatten(output)
        output = tf.reshape(output_f, [-1, 8*8 * 512])
        print("=====》1", y_.shape, output.shape)
        W_fc1 = tf.get_variable('fc1', shape=[8*8 * 512, 4096], initializer=tf.contrib.keras.initializers.he_normal())
        b_fc1 = bias_variable([4096])
        output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1) + b_fc1))
        output = tf.nn.dropout(output, keep_prob)

        W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
        b_fc2 = bias_variable([4096])
        output_aa = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2))
        output = tf.nn.dropout(output_aa, keep_prob)

        W_fc3 = tf.get_variable('fc3', shape=[4096, class_num], initializer=tf.contrib.keras.initializers.he_normal())
        b_fc3 = bias_variable([class_num])
        output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3))
        # output  = tf.reshape(output,[-1,10])

        # loss function: cross_entropy
        # train_step: training operation
        print("=====》2", y_.shape, output.shape)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True). \
            minimize(cross_entropy + l2 * weight_decay)

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./model_z/g_VGG_10id/'))
        print('shape:', org_x.shape, org_y.shape)

        # test_x = segdata_preprocessing()
        # test_x = test_normalization(org_x)
        test_x = org_x
        test_y = org_y
        print('test_x.shape, test_y.shape:', test_x.shape, test_y.shape)


        ep_output, ep_y_, correct_p = [], [], []
        start_time = time.time()

        # val_acc, val_loss, test_summary, ep_y_, correct_p, cam_o1, cam_o2, cam_o3, cam_o4 = run_testing(sess)
        # val_acc, val_loss, test_summary, ep_y_, correct_p, max1_1_, max1_2_, max1_3_, max2_1_, max_res_ = run_testing(sess)
        # val_acc, val_loss, test_summary, ep_y_, correct_p, cam_img_a1_t, cam_img_a1_m, cam_img_a1, \
        # cam_img_a2_t, cam_img_a2_m, cam_img_a2, cam_img_a3_t, cam_img_a3_m, cam_img_a3, cam_out = run_testing(sess)
        val_acc, val_loss, test_summary, ep_y_, correct_p, out = run_testing(sess)
        print('out shape:', out.shape)
        np.save('features4096.npy', out)


        print('--org_x.shape', org_x.shape, org_y.shape)
        # print('--cam_img_a1_t.shape', cam_img_a3_t.shape)
        # for i in range(55):
        #     if correct_p[i] == True:
        #         flag = 'T'
        #     else:
        #         flag = 'F'
        #
        #     org = np.reshape(org_x[i], [112, 112, 3])
        #     org = np.uint8(org)
            # cv2.imwrite('/home/zrx/zrx/pyworkplace/output/Attention/%d_%s.png' % (i, flag), org)
            #
            # cam_write(org, cam_img_a1_t[i], 0.6, 0.4, '%d_a1_t.png' % i)
            # cam_write(org, cam_img_a1_m[i], 0.6, 0.4, '%d_a1_m.png' % i)
            # cam_write(org, cam_img_a1[i], 0.6, 0.4, '%d_a1.png' % i)
            #
            # cam_write(org, cam_img_a2_t[i], 0.6, 0.4, '%d_a2_t.png' % i)
            # cam_write(org, cam_img_a2_m[i], 0.6, 0.4, '%d_a2_m.png' % i)
            # cam_write(org, cam_img_a2[i], 0.6, 0.4, '%d_a2.png' % i)
            #
            # cam_write(org, cam_img_a3_t[i], 0.6, 0.4, '%d_a3_t.png' % i)
            # cam_write(org, cam_img_a3_m[i], 0.6, 0.4, '%d_a3_m.png' % i)
            # cam_write(org, cam_img_a3[i], 0.6, 0.4, '%d_a3.png' % i)
            # cam_write(org, cam_out[i], 0.6, 0.4, '%d_out.png' % i)

            # print('........%d........'%i)
            # print(np.mean(cam_img_a3_t[i]), np.max(cam_img_a3_t[i]), np.min(cam_img_a3_t[i]))
            # print(np.mean(cam_img_a3[i]), np.max(cam_img_a3[i]), np.min(cam_img_a3[i]))
            # sub = cam_img_a3_t[i] - cam_img_a3[i]
            # print(type(sub))
            # print(sub.shape)
            # print(np.mean(sub), np.max(sub), np.min(sub))
            # sub = cv2.resize(sub, (112, 112))
            # cv2.imwrite('/home/zrx/zrx/pyworkplace/output/Attention/%d_sub.png'%i, sub)

        #     acc += val_acc
        #
        # acc /= float(55)
        # print("cost_time: %ds, test_acc: %.4f" % (int(time.time() - start_time), acc))

        TP, FP, FN, TN = TP_def(correct_p, ep_y_)
        print('TP, FP, FN, TN', TP, FP, FN, TN)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        print('FPR, FNR', FPR, FNR)

        print('acc:', val_acc)

        # print(':max1_1_:', max1_1_)
        # print('max1_2_:', max1_2_)
        # print('max1_3_:', max1_3_)
        # print('max_2_1_:', max2_1_)
        # print('max_res_:', max_res_)