import cv2
from collections import Counter
def readData(csv_dir, train_img_dir):
    num_of_id = 10
    df = pd.read_csv(csv_dir)  #.set_index('Image')
    new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
    train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training
    unique_labels = np.unique(train_df.Id.values)
    # choose_id = unique_labels[:num_of_id]
    train_img, train_lab = [], []
    num = 0
    flag = True
    print(train_df.shape)
    data = train_df.values
    count = Counter()
    for i in range(len(data)):
        if train_df.Id.values[i] not in count:
            count[train_df.Id.values[i]] = 1
        else:
            count[train_df.Id.values[i]] += 1
    common_whale = np.array([x[0] for x in count.most_common(10)])
    id_img_num = np.zeros((len(common_whale)))
    #print (common_whale)
    for i in range(len(data)):
        if train_df.Id.values[i] in common_whale:
    #        print (train_df.Id.values[i] )
            label = np.where(common_whale == train_df.Id.values[i])
    #        print (label)
            # train_df.Id.values[i] = train_df.Id.values[i].apply(lambda x: labels_dict[x])
            id_img_num[label] += 1
            num += 1
                # print(train_df.Id.values[i])
            # print(data[i,0])
            img = cv2.imread(train_img_dir + str(data[i,0]))
            if img is None:
                continue
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            train_img.append(img)
            train_lab.append(label)
            if num % 1000 == 0:
                print(num, ' done')
                # cv2.imwrite('%d.png' % num, img)
    train_img = np.array(train_img)
    train_lab = np.array(train_lab)
    #print ('======================')
    train_labl = np.squeeze(train_lab)
    #train_lab = to_categorical(train_lab)
    return train_img, train_lab
