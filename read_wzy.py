#coding=utf-8


import numpy as np
import pandas as pd
import os
import shutil

csv_dir='D:/whale/semi-MList/code/train.csv'
num_of_id = 5004
df = pd.read_csv(csv_dir)  #.set_index('Image')
new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training
unique_labels = np.unique(train_df.Id.values)
print(unique_labels.shape)
# choose_id = unique_labels[:num_of_id]
datav = train_df.values
id_img_num = np.zeros((num_of_id))

choose_id = unique_labels
print(choose_id)
for i in range(len(datav)):
	if train_df.Id.values[i] in choose_id:
		label = np.where(choose_id == train_df.Id.values[i])
		id_img_num[label] += 1
	

print(id_img_num)

sort_list = np.argsort(id_img_num)




	
num_class = 10

old_dir = 'train/'
new_dir = 'data_10/'

record = []
imgs = []
labels = []
count = 0
for i in range(num_class):
	select_id = unique_labels[sort_list[num_of_id-i-25]]
	print(id_img_num[sort_list[num_of_id-i-25]])
	count = count + 1
	train_count = 0
	for j in range(len(datav)):
		if train_df.Id.values[j] == select_id:	
			old_name = old_dir + datav[j,0]
			new_name = new_dir + datav[j,0]
			#每个类15张训练，剩下的测试
			imgs.append(new_name)
			labels.append(count)
			train_count = train_count + 1
			if train_count > 15:
				record.append(0)
			else:
				record.append(1)
			shutil.copyfile(old_name,new_name)

f1 = open('images.txt','w')
for i in imgs:
	f1.write(i)
	f1.write('\n')
f1.close()	

f1 = open('image_class_labels.txt','w')
for i in labels:
	f1.write(str(i))
	f1.write('\n')
f1.close()	

f1 = open('train_test_split.txt','w')
for i in record:
	f1.write(str(i))
	f1.write('\n')
f1.close()	

				