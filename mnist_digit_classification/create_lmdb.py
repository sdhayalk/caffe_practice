'''
referred from: 
    http://deepdish.io/2015/04/28/creating-lmdb-in-python/
    https://github.com/sdhayalk/MNIST_Digit_Recognizer_Kaggle/blob/master/preprocessing.py
'''

import caffe
import numpy as np
import lmdb
import csv

def get_dataset_features_labels_np(dataset_path):
    dataset = []
    first_line_flag = True

    with open(dataset_path) as f:
        dataset_csv_reader = csv.reader(f, delimiter=",")
        for line in dataset_csv_reader:
            if first_line_flag:
                first_line_flag = False
                continue

            temp_line = []
            for element in line:
                temp_line.append(float(element))

            dataset.append(temp_line)

    dataset = np.array(dataset)
    dataset_features = np.array(dataset[:, 1:], dtype='float')
    dataset_labels_temp = np.array(dataset[:, 0], dtype='int')

    # dataset_labels = []
    # for element in dataset_labels_temp:
    #     temp = np.zeros(10, dtype='int')    # number of classes is 10
    #     temp[int(element)] = 1
    #     dataset_labels.append(temp)
    # dataset_labels = np.array(dataset_labels, dtype='int')

    return dataset_features.reshape((dataset_features.shape[0],1,28,28)), dataset_labels_temp.reshape((dataset_labels_temp.shape[0]))


dataset_train_path = 'G:/DL/mnist_data_for_caffe/train.csv'
lmdb_data_name = 'G:/DL/mnist_data_for_caffe/train_data_lmdb'
lmdb_label_name = 'G:/DL/mnist_data_for_caffe/train_label_lmdb'

dataset_train_features, dataset_train_labels = get_dataset_features_labels_np(dataset_train_path)
print(dataset_train_features.shape, dataset_train_labels.shape)


# Labels
env = lmdb.open(lmdb_data_name, map_size=int(6e8))

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(dataset_train_features.shape[0]):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = dataset_train_features.shape[1]
        datum.height = dataset_train_features.shape[2]
        datum.width = dataset_train_features.shape[3]
        datum.data = dataset_train_features[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(dataset_train_labels[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
