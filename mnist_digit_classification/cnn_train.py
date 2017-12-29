'''
references from:
	http://caffe.berkeleyvision.org/gathered/examples/mnist.html
	https://github.com/nitnelave/pycaffe_tutorial
'''

import caffe

USE_GPU = True
if USE_GPU:
    caffe.set_device(0)
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

CNN_NETWORK_PATH = "cnn_network.prototxt"
loaded_network = caffe.Net(CNN_NETWORK_PATH, caffe.TEST) # caffe.TEST for testing
