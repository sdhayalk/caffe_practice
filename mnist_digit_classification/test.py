import caffe
import numpy as np

from data_preprocessing import get_test_dataset_features

def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


DATASET_TEST_PATH = 'G:/DL/mnist_data_for_caffe/test.csv'
dataset_test_features = get_test_dataset_features(DATASET_TEST_PATH, normalize=True)
print('dataset_test_features.shape:', dataset_test_features.shape)
NUM_EXAMPLES = dataset_test_features.shape[0]
BATCH_SIZE = 100

USE_GPU = True
if USE_GPU:
    caffe.set_device(0)
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

CNN_NETWORK_PATH = "G:/DL/caffe_practice/mnist_digit_classification/cnn_deploy_network.prototxt"
CAFFEMODEL_PATH =  "G:/DL/mnist_data_for_caffe/snapshot_iter_2500.caffemodel"
net = caffe.Net(CNN_NETWORK_PATH, CAFFEMODEL_PATH, caffe.TEST)

i_= 0
first_flag = True
predicted_labels = []
for i in range(0, int(NUM_EXAMPLES/BATCH_SIZE)):
	i_=i
	batch_x = get_batch(dataset_test_features, i, BATCH_SIZE)
	predicted_batch_x = net.forward(data=batch_x)

	if first_flag:
		print(predicted_batch_x)
		predicted_labels = predicted_batch_x['prob']
		first_flag = False
	else:
		predicted_labels = np.concatenate((predicted_labels, predicted_batch_x['prob']), axis=0)

