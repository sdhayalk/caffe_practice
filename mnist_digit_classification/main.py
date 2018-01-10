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
	Step 2) define the network in .prototxt
	Step 3) define the solver in .prototxt
	Step 4) load the network, perform training using solver
	Step 5) test using test set
'''
import caffe
from data_preprocessing import convert_to_HDF5

#--- Step 1) ---

dataset_train_path = 'G:/DL/mnist_data_for_caffe/train.csv'
dataset_train_hdf5_path = 'G:/DL/mnist_data_for_caffe/dataset_train.hdf5'
convert_to_HDF5(dataset_train_path, dataset_train_hdf5_path)

# USE_GPU = True
# if USE_GPU:
#     caffe.set_device(0)
#     caffe.set_mode_gpu()
# else:
#     caffe.set_mode_cpu()

# CNN_NETWORK_PATH = "G:/DL/caffe_practice/mnist_digit_classification/cnn_network.prototxt"
# CNN_SOLVER_PATH = "G:/DL/caffe_practice/mnist_digit_classification/cnn_solver.prototxt"

# net = caffe.Net(CNN_NETWORK_PATH, caffe.TRAIN) # caffe.TEST for testing
# solver = caffe.AdamSolver(CNN_SOLVER_PATH)

# print("Network layers information:")
# for name, layer in zip(net._layer_names, net.layers):
#     print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
# print("Network blobs information:")
# for name, blob in net.blobs.items():
#     print("{:<7}: {}".format(name, blob.data.shape))
# print('net.inputs:', net.inputs)
# print('net.outputs:', net.outputs)

# solver.solve()

