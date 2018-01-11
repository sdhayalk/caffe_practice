import numpy as np
import csv
import h5py

def get_dataset_features_labels_np(dataset_train_path, normalize, ratio_one_is_to):
    dataset_train = []
    dataset_validation = []
    first_line_flag = True
    counter = 0

    with open(dataset_train_path) as f:
        dataset_csv_reader = csv.reader(f, delimiter=",")
        for line in dataset_csv_reader:
            if first_line_flag:
                first_line_flag = False
                continue

            temp_line = []
            for element in line:
                temp_line.append(float(element))

            if counter % ratio_one_is_to == 0:
                dataset_validation.append(temp_line)
            else:
                dataset_train.append(temp_line)
            counter += 1

    dataset_train = np.array(dataset_train)
    dataset_validation = np.array(dataset_validation)
    dataset_train_features = np.array(dataset_train[:, 1:], dtype=np.float32)
    dataset_validation_features = np.array(dataset_validation[:, 1:], dtype=np.float32)
    if normalize:
        dataset_train_features = dataset_train_features / 255.0
        dataset_validation_features = dataset_validation_features / 255.0

    dataset_train_labels = np.array(dataset_train[:, 0], dtype='int')
    dataset_validation_lables = np.array(dataset_validation[:, 0], dtype='int')

    return dataset_train_features.reshape((dataset_train_features.shape[0],1,28,28)), dataset_validation_features.reshape((dataset_validation_features.shape[0],1,28,28)), dataset_train_labels, dataset_validation_lables


def convert_to_HDF5(dataset_train_path, hdf5_train_filename, hdf5_validation_filename, normalize=True, ratio_one_is_to=7):
    dataset_train_features, dataset_validation_features, dataset_train_labels, dataset_validation_labels = get_dataset_features_labels_np(dataset_train_path, normalize, ratio_one_is_to)
    print('dataset_train_features.shape:', dataset_train_features.shape,' dataset_train_labels.shape:', dataset_train_labels.shape)
    print('dataset_validation_features.shape:', dataset_validation_features.shape,' dataset_validation_labels.shape:', dataset_validation_labels.shape)

    with h5py.File(hdf5_train_filename, 'w') as f:
        f['data'] = dataset_train_features
        f['label'] = dataset_train_labels

    with h5py.File(hdf5_validation_filename, 'w') as f:
        f['data'] = dataset_validation_features
        f['label'] = dataset_validation_labels

    print("Saved as HDF5")


def get_test_dataset_features(dataset_path, normalize=True):
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

    dataset_features = np.array(dataset, dtype='float')
    if normalize:
        dataset_features = dataset_features / 255.0

    return dataset_features.reshape((dataset_features.shape[0],1,28,28))
