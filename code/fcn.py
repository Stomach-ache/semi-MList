import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from matplotlib import pyplot as plt
import cv2

def weight_variable(shape, name=''):
    initial = tf.truncated_normal(shape, stddev=0.01)
    if name != '':
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def avg_pool(input, s):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'VALID')

def batch_norm(input, is_training):
    return tf.contrib.layers.batch_norm(input, is_training=is_training, decay=0.9, center=True, scale=True,
                                        epsilon=1e-3, updates_collections=None)

def deconv2d(inputs, filter_height, filter_width, output_shape, stride=(1, 1), padding='SAME', name='Deconv2D'):
    input_channels = int(inputs.get_shape()[-1])
    output_channels = output_shape[-1]
    fan_in = filter_height * filter_width * output_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [filter_height, filter_width, output_channels, input_channels]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        biases_init = tf.constant_initializer(0.1)

        filters = tf.get_variable('weights', shape=weights_shape, initializer=filters_init,
                                  collections=['weights', 'variables'])
        biases = tf.get_variable('biases', shape=biases_shape, initializer=biases_init,
                                 collections=['biases', 'variables'])

    return tf.nn.conv2d_transpose(inputs, filters, output_shape, strides=[1, 2, 2, 1], padding=padding,
                                  name=name) + biases

def iou(pre,mask):
    # print('aaa', pre.shape, mask.shape)
    # pre[pre<128]=0
    # mask[mask<128]=0
    pre = np.uint8(pre==0)
    mask = np.uint8(mask==0)

    join=(pre+mask)==2
    union=((pre+mask)>0)

    join=np.uint8(255*join)
    union=np.uint8(255*union)

    rate=np.sum(join)/np.maximum(np.sum(union), 1)
    return rate

class MnistModel:
    def __init__(self):
        self.train_imgs = "/home/zrx/桌面/seg-whale/img.npy"
        self.train_labels = "/home/zrx/桌面/seg-whale/mask.npy"
        # self.test_imgs = "/home/shq/PycharmProjects/gan-fcn/data/test_imgs.npy"
        # self.test_labels = "/home/shq/PycharmProjects/gan-fcn/data/test_masks.npy"
        self.gan_imgs = "/home/zrx/桌面/224/test_x.npy"
        self.gan_labels = "/home/zrx/桌面/224/test_m.npy"
        # 图片大小
        self.img_size = 224
        # 每步训练使用图片数量
        self.batch_size = 10
        self.test_batch_size = 10
        # 训练循环次数
        self.epoch_size = 16
        # 抽取样本数
        self.sample_size = 25
        # 生成器判别器隐含层数量
        self.units_size = 128
        # 学习率
        self.learning_rate = 0.001
        # 平滑参数
        self.smooth = 0.1
	
    @staticmethod
    def generator_graph(fake_imgs, is_training, reuse=False):
        # 生成器与判别器属于两个网络 定义不同scope
        with tf.variable_scope('fcn', reuse=reuse):
            print("生成器")
            print(fake_imgs.shape)
            W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv1_1 = bias_variable([128])
            output = tf.nn.relu(batch_norm(conv2d(fake_imgs, W_conv1_1) + b_conv1_1, is_training))

            W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv1_2 = bias_variable([128])
            output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2, is_training))

            W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv2_1 = bias_variable([128])
            output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1, is_training))

            W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv2_2 = bias_variable([128])
            conv224 = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2, is_training), name='conv224')
            pool1 = max_pool_2x2(conv224)
            print(pool1.shape)

            W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv3_1 = bias_variable([128])
            output = tf.nn.relu(batch_norm(conv2d(pool1, W_conv3_1) + b_conv3_1, is_training))

            W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv3_2 = bias_variable([128])
            conv112 = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2, is_training), name='conv112')
            pool2 = max_pool_2x2(conv112)
            # 56*56

            W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv4_1 = bias_variable([128])
            output = tf.nn.relu(batch_norm(conv2d(pool2, W_conv4_1) + b_conv4_1, is_training))

            W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv4_2 = bias_variable([128])
            conv56 = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2, is_training))
            pool3 = max_pool_2x2(conv56)
            # 28*28

            W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv5_1 = bias_variable([128])
            output = tf.nn.relu(batch_norm(conv2d(pool3, W_conv5_1) + b_conv5_1, is_training))

            W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv5_2 = bias_variable([128])
            conv28 = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2, is_training))
            pool4 = max_pool_2x2(conv28)
            # 14*14

            W_conv6_1 = tf.get_variable('conv6_1', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv6_1 = bias_variable([128])
            output = tf.nn.relu(batch_norm(conv2d(pool4, W_conv6_1) + b_conv6_1, is_training))

            W_conv6_2 = tf.get_variable('conv6_2', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv6_2 = bias_variable([128])
            conv14 = tf.nn.relu(batch_norm(conv2d(output, W_conv6_2) + b_conv6_2, is_training))
            pool5 = max_pool_2x2(conv14)
            # 7*7
            W_conv7_1 = tf.get_variable('conv7_1', shape=[3, 3, 128, 128],
                                        initializer=tf.contrib.keras.initializers.he_normal())
            b_conv7_1 = bias_variable([128])
            conv7 = tf.nn.relu(batch_norm(conv2d(pool5, W_conv7_1) + b_conv7_1, is_training))

            deconv14 = deconv2d(conv7, 4, 4, [tf.shape(output)[0], 14, 14, 128], stride=(2, 2), name='deconv14')
            fuse_14 = tf.add(deconv14, conv14, name="fuse_14")
            print(fuse_14.shape)

            deconv28 = deconv2d(fuse_14, 4, 4, [tf.shape(output)[0], 28, 28, 128], stride=(2, 2), name='deconv28')
            fuse_28 = tf.add(deconv28, conv28, name="fuse_28")
            print(fuse_28.shape)

            deconv56 = deconv2d(fuse_28, 4, 4, [tf.shape(output)[0], 56, 56, 2], stride=(2, 2), name='deconv56')
            deconv = tf.nn.relu(deconv56)

            deconv = tf.image.resize_bicubic(deconv, [224, 224])
            logits = deconv
            outputs = tf.argmax(deconv, axis=3)
            tf.add_to_collection('pre_mask', outputs)

            print("genlogits", logits.shape)
            return logits, outputs

    @staticmethod
    def loss_graph(pre_mask, lable_mask):

        fcn_loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_mask,
                                                                                  labels=tf.squeeze(
                                                                                      tf.cast(lable_mask, tf.int64)))))

        return fcn_loss
	
    @staticmethod
    def optimizer_graph(fcn_loss, learning_rate):

        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('fcn')]
        print("开始进行优化")
        gen_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fcn_loss, var_list=gen_vars)

        return gen_optimizer
	
    def train(self):

        lr = tf.placeholder("float", shape=[])
        is_training = tf.placeholder("bool", shape=[], name='is_training')
        input_imgs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3], name='input_imgs')
        mask_imgs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size], name='mask_imgs')

        logits, outputs = self.generator_graph(input_imgs, is_training)
        fcn_loss = self.loss_graph(logits, mask_imgs)
        gen_optimizer = self.optimizer_graph(fcn_loss, lr)

        # 开始训练
        saver = tf.train.Saver()

        step = 0
        # 指定占用GPU比例
        # tensorflow默认占用全部GPU显存 防止在机器显存被其他程序占用过多时可能在启动时报错
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state('trained/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored...")
            data, labels = np.load(self.train_imgs)/255.0,np.load(self.train_labels)
            train_data = data[:160]
            train_labels = labels[:160]
            test_data = data[160:]
            test_labels = labels[160:]
            # train_data = cv2.resize(train_data, (112, 112))
            # train_labels = cv2.resize(train_labels, (112, 112))
            train_labels = np.uint8(train_labels>128)
            print("train",train_data.shape)
            batch_count = len(train_data) // self.batch_size
            batches_data = np.split(train_data[:batch_count * self.batch_size], batch_count)
            batches_labels = np.split(train_labels[:batch_count * self.batch_size], batch_count)

            # test_data, test_labels = np.load(self.gan_imgs)/255.0,np.load(self.gan_labels)
            # test_data = cv2.resize(test_data, (112, 112))
            # test_labels = cv2.resize(test_labels, (112, 112))
            test_labels = np.uint8(test_labels>128)
            print()
            print("test",test_data.shape)
            batch_count_test = len(test_data) // self.test_batch_size
            batches_data_test = np.split(test_data[:batch_count_test * self.test_batch_size], batch_count_test)
            batches_labels_test = np.split(test_labels[:batch_count_test * self.test_batch_size], batch_count_test)


            learn_rate = 0.001
            for epoch in range(1, 51):

                for batch_idx in range(batch_count):
                    xs, ys = batches_data[batch_idx], batches_labels[batch_idx]

                    _ = sess.run([gen_optimizer],
                                 feed_dict={lr: learn_rate, input_imgs: xs, mask_imgs: ys,
                                            is_training: True})


                if epoch % 5 == 0:
                    sum = 0.0
                    num=0
                    for batch_test_idx in range(batch_count_test):
                        xs_test, ys_test = batches_data_test[batch_test_idx], batches_labels_test[batch_test_idx]

                        pre = sess.run(outputs,
                                       feed_dict={lr: learn_rate, input_imgs: xs_test, mask_imgs: ys_test,
                                               is_training: False})

                        for m in range(self.test_batch_size):
                            rate = iou(pre[m], ys_test[m])
                            cv2.imwrite("show/%d-%d.png" % (epoch, batch_test_idx * self.test_batch_size + m),
                                        np.uint8(pre[m] * 255.0))
                            cv2.imwrite("show/%d-%d-true.png" % (epoch, batch_test_idx * self.test_batch_size + m),
                                        np.uint8(ys_test[m] * 255.0))
                            # res1 = cv2.imread("show/%d-%d.png" % (epoch, batch_test_idx * self.test_batch_size + m))
                            # res2 = cv2.imread("show/%d-%d-true.png" % (epoch, batch_test_idx * self.test_batch_size + m))
                            # rate = iou(res1, res2)
                            sum = sum + rate
                            num = num+1


                    print(epoch, num, "交并比", sum / num)
                if epoch % 10 ==0:
                    saver.save(sess, 'trained/fcn-%d.ckpt' % epoch, global_step=step)

    def gen(self):
        # 生成图片
        sample_imgs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3], name='sample_imgs')
        data = np.load("/home/shq/PycharmProjects/thyroid/weakly-supervised-segment/data/segdata/data.npy") / 255.0
        gen_logits, gen_outputs = self.generator_graph(sample_imgs, self.units_size, self.img_size)
        saver = tf.train.Saver()
        list = []
        print("data", data.shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('fcn_train/.'))
            flag = 800
            n = 0
            for i in range(8):
                sample_noise = data[data.shape[0] - flag:data.shape[0] - flag + 100]
                print(flag, flag + 100)
                part = sess.run(gen_outputs, feed_dict={sample_imgs: sample_noise})
                flag = flag - 100
                print(part.shape[0])
                for m in range(part.shape[0]):
                    cv2.imwrite("fcn/%d.png" % n, np.uint8(part[m] * 255.0))
                    n = n + 1
                print(n)
            sess.close()

    @staticmethod
    def show():
        # 展示图片
        with open('samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        fig, axes = plt.subplots(figsize=(224, 224), nrows=5, ncols=5, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        plt.show()


model = MnistModel()
model.train()