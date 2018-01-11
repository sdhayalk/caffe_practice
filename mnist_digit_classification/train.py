'''
references from:
	http://caffe.berkeleyvision.org/gathered/examples/mnist.html
	https://github.com/nitnelave/pycaffe_tutorial
	https://stackoverflow.com/questions/32379878/cheat-sheet-for-caffe-pycaffe
	https://github.com/Franck-Dernoncourt/caffe_demos/blob/master/iris/iris_tuto.py
'''

'''
Steps for entire data preprocessing, train, validation, test
	Step 1) load dataset and save it as HDF5
	Step 2) define the network in .prototxt and include its path
	Step 3) define the solver in .prototxt and include its path
	Step 4) load the network, perform training using solver
	Step 5) test using test set
'''
import caffe
import os
from data_preprocessing import convert_to_HDF5
from utils import display_stats

#--- Step 1) ---

DATASET_TRAIN_PATH = 'G:/DL/mnist_data_for_caffe/train.csv'
DATASET_TRAIN_HDF5_PATH = 'G:/DL/mnist_data_for_caffe/dataset_train.hdf5'
DATASET_VALIDATION_HDF5_PATH = 'G:/DL/mnist_data_for_caffe/dataset_validation.hdf5'

if not os.path.exists(DATASET_TRAIN_HDF5_PATH):
	convert_to_HDF5(DATASET_TRAIN_PATH, DATASET_TRAIN_HDF5_PATH, DATASET_VALIDATION_HDF5_PATH, normalize=True, ratio_one_is_to=7)

USE_GPU = True
if USE_GPU:
    caffe.set_device(0)
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()


#--- Step 2) and Step 3) ---
CNN_NETWORK_PATH = "G:/DL/caffe_practice/mnist_digit_classification/cnn_network.prototxt"
CNN_SOLVER_PATH = "G:/DL/caffe_practice/mnist_digit_classification/cnn_solver.prototxt"


#--- Step 4) ---

net = caffe.Net(CNN_NETWORK_PATH, caffe.TRAIN) # caffe.TEST for testing
solver = caffe.get_solver(CNN_SOLVER_PATH)
display_stats(net)
solver.solve()
