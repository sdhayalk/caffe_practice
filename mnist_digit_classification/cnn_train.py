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

CNN_NETWORK_PATH = "G:/DL/caffe_practice/mnist_digit_classification/cnn_network.prototxt"
net = caffe.Net(CNN_NETWORK_PATH, caffe.TRAIN) # caffe.TEST for testing

print("Network layers information:")
for name, layer in zip(net._layer_names, net.layers):
    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
print("Network blobs information:")
for name, blob in net.blobs.items():
    print("{:<7}: {}".format(name, blob.data.shape))
print('net.inputs:', net.inputs)
print('net.outputs:', net.outputs)

