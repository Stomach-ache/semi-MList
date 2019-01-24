# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import datetime
import matplotlib.pyplot as plt
from skimage import measure
import math
from readthyroid import *
from kutils import *
from maskcam import *
from CRF import *


img_channels = 3
fsize = 1
fksize = 56
# model_path = '/home/zrx/zrx/pyworkplace/model_z/AttUnet_noInfo_trunk_GAP56-1_test1/AttUnet_noInfo_trunk_GAP56-1_test1'
# model_path = '/home/zrx/zrx/pyworkplace/model_z/AttUnet_noInfo_GAP56-1_conv_test5/Att_1m_noInfo_GAP56-1_conv_test5'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/att_semi/400/PA56-2-1/PA112-2'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/att_semi/att/112/400/test_semi_cam_64/test'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/test_comp_fcn_big/test_comp_fcn_big'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/unet_1/unet_1'
model_path1 = '/home/zrx/zrx/pyworkplace/trained/fcn-50.ckpt-0'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/segnet_2/segnet_2'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/test_CMU_little/test_CMU_little'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/att_semi/att/112/400/test_semi_cam_64/test'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/att_semi/400/AttVGG-2/AttVGG-2'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/att_semi/150/PA56-3-1/PA56-3-1'
# model_path1 = '/home/zrx/zrx/pyworkplace/model_z/semi_comp/big/224/test_comp_fcn_big_74-2/test_comp_fcn_big'
# data_dir = '/home/zrx/桌面/CMU/CMU_little/'
# data_dir = '/home/zrx/zrx/data/thyroid/data_20000/comp_semi/'
data_dir = '/home/zrx/zrx/data/thyroid/data_7216/207_big/224/'
# data_dir = '/home/zrx/zrx/data/thyroid/data_7216/test_semi_400-2600/224/'
centerType = 'mid'  # mid or find
overFlag = 'ex'  # hull 凸包 or ex 闭运算
bigFlag = True
separate = True
circle = True
mm = 0.25
bb = 0.09
'''
square:
    hull && mid = 38.75
        b: 0.15  m: 0.27
    ex && mid = 38.83
        b: 0.09  m: 0.27

circle:
    hull && mid = 39.38
        b: 0.14  m: 0.25
    ex && mid = 39.91
        b: 0.09  m: 0.25
'''

def findmid(i, img, threshold):
    img = np.uint8(img > threshold)
    mea = measure.regionprops(img)
    y = np.uint8(mea[0].centroid[0])
    x = np.uint8(mea[0].centroid[1])
    return x, y

def redeal(img, info):
    x, y, e =info
    img_ = np.zeros((image_size, image_size))
    img = cv2.resize(img, (e, e))
    img_[y:y+e, x:x+e] = img
    return img_

def f(x, mid, b):
    if x < mid:
        a = -b/mid
        return a * x + b
    else:
        return 0
    # return -pow(x-56, 2)*1/pow(56, 2)+1

def center(x=0, y=0):
    cen = np.zeros((image_size, image_size))
    for i in range(image_size):
        for j in range(image_size):
            mid = image_size//2
            if centerType == 'mid':
                if circle:
                    dis = math.sqrt(math.pow(i-mid, 2)+math.pow(j-mid, 2))
                else:
                    dis = max(abs(i-mid), abs(j-mid))
                cen[i, j] = f(dis, mid, 1.0)
            elif centerType == 'find':
                if circle:
                    dis = math.sqrt(math.pow(i - y, 2) + math.pow(j - x, 2))
                else:
                    dis = max(abs(i - y), abs(j - x))
                cen[i, j] = f(dis, mid, 1.0)
            # if (i > (image_size//3-1) and i < image_size*2/3) and(j > (image_size//3-1) and j < image_size*2/3):
            #     cen[i, j] = 1.0
            # elif (i > (image_size//4-1) and i < image_size*3/4) and(j > (image_size//4-1) and j < image_size*3/4):
            #     cen[i, j] = 0.8
            # elif (i > (image_size//8-1) and i < image_size*7/8) and(j > (image_size//8-1) and j < image_size*7/8):
            #     cen[i, j] = 0.6
            # else:
            #     cen[i, j] = 0.4
    return cen

def drawHist(b):
    # 创建直方图
    # 第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
    # 第二个参数为划分的区间个数
    plt.hist(b, 50, facecolor='b', alpha=0.4)
    # plt.hist(m, 200, facecolor='r', alpha=0.4)
    plt.xlabel('A/T')
    plt.ylabel('Frequency')
    plt.title('')
    plt.show()

def TP_def(c, y):
    c = np.array(c)
    y = np.array(y)
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

# def crop_segimg(img,):

if __name__ == '__main__':

    # org_x, org_y = prepare_testdata()
    # org_x = np.load(data_dir+'test_x.npy') # /home/zrx/zrx/data/thyroid/data_7216/test_semi_300-2700
    # org_x = np.load('/home/zrx/桌面/seg-whale/img.npy') # /home/zrx/zrx/data/thyroid/data_7216/test_semi_300-2700
    # cv2.imwrite('/home/zrx/zrx/data/thyroid/data_7216/207_big/test.png', org_x[0])
    # test_y = np.load(data_dir+'test_y.npy')
    mask = np.load(data_dir+'test_m.npy')
    # mask = np.uint8(mask>128)

    # print('org_x.shape', org_x.shape, mask.shape)
    # test_n = org_x.shape[0]
    acc_te, loss_te = [], []
    acc = 0.0

    # test_x = org_x/255.0
    # test_x = segdata_preprocessing(org_x)
    # print(np.mean(org_x))
    # print(np.mean(test_x))
    # print('test_x.shape, test_y.shape:', test_x.shape, mask.shape)
    # print(np.mean(test_x[0]))

    graph = tf.Graph()
    sess = tf.Session(graph = graph)

    with graph.as_default():
        start_time = time.time()
        saver = tf.train.import_meta_graph(model_path1 + '.meta')
        saver.restore(sess, model_path1)

        # x = graph.get_operation_by_name('x').outputs[0]
        # x1 = graph.get_operation_by_name('x1').outputs[0]
        x1 = graph.get_operation_by_name('input_imgs').outputs[0]
        # y_ = graph.get_operation_by_name('y_').outputs[0]
        # mask_ = graph.get_operation_by_name('mask_').outputs[0]
        mask_ = graph.get_operation_by_name('mask_imgs').outputs[0]
        # keep_prob = graph.get_operation_by_name('keep_p').outputs[0]
        # learning_rate = graph.get_operation_by_name('learning_r').outputs[0]
        # train_flag = graph.get_operation_by_name('train_f').outputs[0]
        train_flag = graph.get_operation_by_name('is_training').outputs[0]
        # prediction_op = graph.get_operation_by_name('prediction').outputs[0]
        # weight_op = graph.get_operation_by_name('fc1').outputs[0]
        # features_op = graph.get_collection('feature')[0]
        # features_op = graph.get_operation_by_name('feature').outputs[0]
        # accuracy = graph.get_operation_by_name('accuracy').outputs[0]
        # correct_prediction = graph.get_operation_by_name('correct_prediction').outputs[0]
        # anno = graph.get_operation_by_name('pre_mask').outputs[0]
        anno = graph.get_collection('pre_mask')[0]
        # pre_s = graph.get_operation_by_name('pre').outputs[0]
        dir = '/home/zrx/桌面/whale/train/'
        f = open('/home/zrx/zrx/pyworkplace/g_whale/images_10.txt', 'r')
        f_name = np.array(f.readlines())

        plot, list, result, alldice = [], [], [], []
        sum_rate, sum_m, sum_b, num = 0, 0, 0, 0
        sum_pa, num_zero, num_L = 0, 0, 0
        for key in f_name:
            name = key.strip('\n')
            name = name.strip('data_10')
            temp = Image.open(dir + name)
            temp = temp.resize((224, 224), Image.ANTIALIAS)
            # print(sum_pa, ':', temp.size)
            # temp = Image.fromarray(org_x[i])
            # temp.save('/home/zrx/桌面/img/%d.png' % (i))
            # cv2.imwrite('/home/zrx/桌面/img/%d.png' % (i), org_x[i])
            # batch_x = test_x[i*20:(i+1)*20]
            batch_x_ = np.array(temp)
            batch_x = batch_x_[np.newaxis, :]/255.
            # print(batch_x.shape)
            # batch_y = test_y[i*20:(i+1)*20]
            # batch_y = test_y[i]

            # anno_ = sess.run(anno, feed_dict={x1: batch_x, keep_prob: 1.0, train_flag: False})
            try:
                anno_ = sess.run(anno, feed_dict={x1: batch_x, train_flag: False})
                anno_ = np.uint8(anno_==0)

                sum_x = anno_[0].sum(axis=0)
                sum_y = np.sum(anno_[0], axis=1)
                # print(sum_x)
                # print(sum_y.shape)
                x_place = np.where(sum_x>0)
                y_place = np.where(sum_y>0)
                try:
                    crop_x0 = np.min(x_place)
                    crop_y0 = np.min(y_place)
                    crop_x1 = np.max(x_place)
                    crop_y1 = np.max(y_place)

                    # im = batch_x_
                    # im[crop_y0:crop_y0+2, crop_x0:crop_x1, 2] = 255
                    # im[crop_y1:crop_y0+2, crop_x0:crop_x1, 2] = 255
                    # im[crop_y0:crop_y1, crop_x0:crop_x0+2, 2] = 255
                    # im[crop_y0:crop_y1, crop_x1:crop_x1+2, 2] = 255
                    # cv2.imwrite('/home/zrx/zrx/pyworkplace/g_whale/crop_img/%d.png' % sum_pa, im)
                except:
                    crop_x0 = 0
                    crop_y0 = 0
                    crop_x1 = 224
                    crop_y1 = 224
                    num_zero += 1
            except:
                crop_x0 = 0
                crop_y0 = 0
                crop_x1 = 224
                crop_y1 = 224
                num_L += 1

            box = (crop_x0, crop_y0, crop_x1, crop_y1)
            img_crop = Image.fromarray(batch_x_)
            img_crop = img_crop.crop(box)
            img_crop.save('/home/zrx/zrx/pyworkplace/g_whale/crop/%s' % name)

            # with open('/home/zrx/zrx/pyworkplace/g_whale/crop_10.txt', 'a') as f:
            #     f.write(str(crop_x0)+','+str(crop_y0)+','+str(crop_x1)+','+str(crop_y1)+'\n')
            # print(crop_x0, crop_y0, crop_x1, crop_y1)

            # anno_, classify = sess.run([anno, prediction_op], feed_dict={x: batch_x, x1: batch_x, keep_prob: 1.0, train_flag: False})  # x1: batch_x1,
            # org = np.squeeze(org_x[i*20:(i+1)*20])
            # org = np.squeeze(org_x[i])
            # # cv2.imwrite('/home/zrx/zrx/pyworkplace/output/Attention/test_semi_400_64/org/%d.png' %i, org)
            #
            # org = np.uint8(org)
            # # classify = np.argmax(classify)
            #
            # # rate = get_normal_iu(0, i, org[:, :, :3], pre_mask, mask[i], flag)
            # # rate = get_normal_iu(0, i, org[:, :, :3], anno_[0], mask[i], 1, batch_y, classify)
            # rate, dice = get_normal_iu(0, i, org[:, :, :3], anno_[0], mask[i], 1)
            # # rate, dice = get_normal_ius(i, anno_, mask[i*20:(i+1)*20], org, batch_y)
            # # print('---------------------')
            # sum_rate = sum_rate + rate
            #
            # pa = get_normal_pa(anno_[0], mask[i])
            sum_pa += 1

            # if sum_pa % 100 == 0:
            print('%d done' % sum_pa)
        print(num_zero, num_L)
        #     list.append(rate)
        #     result.append(anno_[0])
        #
        #     alldice.append(dice)
        #
        # result = np.array(result)
        # print(result.shape)
        # # np.save('/home/zrx/zrx/pyworkplace/model_z/test2_4/test2_4_result.npy', result)
        # avg = sum_rate / test_n#
        # plot.append(avg)
        # # maavg = sum_m / m
        # # beavg = sum_b / b
        # print(num, "总平均: ", avg, sum_pa/test_n)
        # # print("恶性", maavg, "\n良性", beavg)
        #
        # print("cost_time: %ds, test_acc: %.4f" % (int(time.time() - start_time), num/test_n))
        # print(np.mean(list), np.mean(alldice))
        # drawHist(list)