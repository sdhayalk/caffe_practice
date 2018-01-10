import numpy as np
import csv
import h5py

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
    dataset_features = np.array(dataset[:, 1:], dtype=np.float32)
    dataset_labels_temp = np.array(dataset[:, 0], dtype='int')

    dataset_labels = []
    for element in dataset_labels_temp:
        temp = np.zeros(10, dtype='int')    # number of classes is 10
        temp[int(element)] = 1
        dataset_labels.append(temp)
    dataset_labels = np.array(dataset_labels, dtype=np.float32)

    return dataset_features.reshape((dataset_features.shape[0],1,28,28)), dataset_labels

def convert_to_HDF5(dataset_path, hdf5_filename):
    dataset_features, dataset_labels = get_dataset_features_labels_np(dataset_path)
    print('dataset_features.shape:', dataset_features.shape,' dataset_labels.shape:', dataset_labels.shape)

    with h5py.File(hdf5_filename, 'w') as f:
        f['data'] = dataset_features
        f['label'] = dataset_labels

    print("Saved as HDF5")
