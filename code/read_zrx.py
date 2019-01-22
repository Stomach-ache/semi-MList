import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
import cv2

csv_dir='/home/zrx/桌面/whale/train.csv'
train_img_dir = '/home/zrx/桌面/whale/train/'
num_of_id = 10
df = pd.read_csv(csv_dir)  #.set_index('Image')
new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training
unique_labels = np.unique(train_df.Id.values)
# choose_id = unique_labels[:num_of_id]

train_img, train_lab = [], []
id_img_num = np.zeros((num_of_id))
num = 0
flag = True
print(train_df.shape)
data = train_df.values

while flag:
    # labels_dict = dict()
    # labels_list = []
    choose_id = np.random.choice(unique_labels, num_of_id)
    print(choose_id)
    # for i in range(len(choose_id)):
    #     labels_dict[choose_id[i]] = i
    for i in range(len(data)):
        if train_df.Id.values[i] in choose_id:
            label = np.where(choose_id == train_df.Id.values[i])
            # train_df.Id.values[i] = train_df.Id.values[i].apply(lambda x: labels_dict[x])
            id_img_num[label] += 1
            num += 1
            # print(train_df.Id.values[i])
            # print(data[i,0])
            img = cv2.imread(train_img_dir + str(data[i,0]))
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            train_img.append(img)
            train_lab.append(label)
            if num % 1000 == 0:
                print(num, ' done')
            # cv2.imwrite('%d.png' % num, img)
    print(num)
    print(id_img_num)
    if np.min(id_img_num) >= 10:
        flag = False
train_img = np.array(train_img)
train_lab = np.array(train_lab)
train_lab = to_categorical(train_lab)
train_lab = np.squeeze(train_lab)
print(train_img.shape)
print(train_lab.shape)
np.save('train_img.npy', train_img)
np.save('train_lab.npy', train_lab)


