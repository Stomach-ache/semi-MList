import keras.backend as K
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
from nts_model import get_nts_net, ranking_loss, part_cls_loss, part_cls_acc
from generator import Generator
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# from model import ParallelModelCheckpoint
from keras.utils import multi_gpu_model
from config import *
import numpy as np
import os
import sys
import math
from PIL import Image
from skimage.transform import resize
from keras.models import Model


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
# config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def load_image(image_file):
    image_path = image_file
    image = Image.open(image_path)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = resize(image_array, (image_dimension, image_dimension))
    return image_array

def transform_batch_images(X):
 #   assert X.shape == (image_dimension, cro, 3)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    X = (X - imagenet_mean) / imagenet_std
    return X



def step_decay(epoch):
    drop = 0.1
    for i in range(len(lr_steps)):
        if epoch + 1 <= lr_steps[i]:
            break
    lrate = initial_learning_rate * math.pow(drop, i)
    print('epoch {0} learning rate is {1}'.format(epoch + 1, lrate))
    return lrate


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='loss', verbose=0,
                 save_best_only=True, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)

def predict(X):

    assert num_gpu > 0
    multi_gpu = num_gpu > 1
    

        

    try:
        # with tf.device('/cpu:0'):
        model = get_nts_net(batch_size=batch_size)
        print(model.summary())
        if multi_gpu:
            model_parallel = multi_gpu_model(model, gpus=num_gpu)
            model_train = model_parallel
        else:
            model_train = model
        print("** compile model with class weights **")
        # optimizer = Adam(lr=initial_learning_rate)
        optimizer = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True)
        model_train.compile(optimizer=optimizer, loss={
            "cls_pred_global": "categorical_crossentropy",
            "cls_pred_part": part_cls_loss(num_classes),
            "cls_pred_concat": "categorical_crossentropy",
            "rank_concat": ranking_loss(PROPOSAL_NUM),
        },
                            loss_weights={
                                'cls_pred_global': 1.,
                                'cls_pred_part': 1,
                                'cls_pred_concat': 1,
                                'rank_concat': 1,
                            },
                            metrics={
                                'cls_pred_global': 'accuracy',
                                'cls_pred_part': part_cls_acc,
                                'cls_pred_concat': 'accuracy',
                            }
                            )

        for layer in model_train.layers:
            print layer.name, ':', layer.losses, '\n'

        model_train.load_weights('exp/model.h5')
        
        print("process image begin")
       

        #preprocess image
        X = load_image(X)
        X = transform_batch_images(X)




        model_test = Model(inputs=model_train.input,outputs=model_train.get_layer('feature_concat').output)
        X = X[np.newaxis, :, :, :]
        fea = model_test.predict(X)
        print(fea.shape)

        model_test = Model(inputs=model_train.input,outputs=model_train.get_layer('cls_pred_concat').output)
        labels = model_test.predict(X)
        print(labels)
        labels = np.argmax(np.array(labels))
        print("** done! **")

    except Exception as e:
        print(e)
    finally:
        pass


    return fea,labels



def main():
    assert num_gpu > 0
    multi_gpu = num_gpu > 1
    

        

    try:
        # with tf.device('/cpu:0'):
        model = get_nts_net(batch_size=batch_size)
        print(model.summary())
        if multi_gpu:
            model_parallel = multi_gpu_model(model, gpus=num_gpu)
            model_train = model_parallel
        else:
            model_train = model
        print("** compile model with class weights **")
        # optimizer = Adam(lr=initial_learning_rate)
        optimizer = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True)
        model_train.compile(optimizer=optimizer, loss={
            "cls_pred_global": "categorical_crossentropy",
            "cls_pred_part": part_cls_loss(num_classes),
            "cls_pred_concat": "categorical_crossentropy",
            "rank_concat": ranking_loss(PROPOSAL_NUM),
        },
                            loss_weights={
                                'cls_pred_global': 1.,
                                'cls_pred_part': 1,
                                'cls_pred_concat': 1,
                                'rank_concat': 1,
                            },
                            metrics={
                                'cls_pred_global': 'accuracy',
                                'cls_pred_part': part_cls_acc,
                                'cls_pred_concat': 'accuracy',
                            }
                            )

        for layer in model_train.layers:
            print layer.name, ':', layer.losses, '\n'

        model_train.load_weights('exp/model.h5')
        
        print("process image begin")
       

        #preprocess image
        count = 0
        filename = 'data_10_new/images.txt'
        img_dir = 'data_10_new/images/'

        size = 0
        with open(filename) as handle:
            for line in handle:
                size = size + 1
        fea_array = [[0 for i in range(2048*3)] for j in range(size)]

        with open(filename) as handle:
            for line in handle:
                img = img_dir + line[:-1]
                print(img)
                X = load_image(img)
                X = transform_batch_images(X)




                model_test = Model(inputs=model_train.input,outputs=model_train.get_layer('feature_concat').output)
                X = X[np.newaxis, :, :, :]
                fea = model_test.predict(X)
                print (fea.shape)
                fea_array[count] = fea[0, :2048*3]
                #print (fea_array[count].shape)
                #fea_array.append(fea[:2048])
                count = count + 1
                print(count)

        print("** done! **")

      #  np.save('feature10_new.npy', np.array(fea_array))
    except Exception as e:
        print(e)
    finally:
        pass


if __name__ == "__main__":
    #main()
    imgname = 'data_10_new/images/data_10_new/02bdec750.jpg'
    fea, label = predict(imgname)
    print(fea)
    print(label)
