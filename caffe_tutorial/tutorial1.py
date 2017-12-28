'''
referred from: 
	https://prateekvjoshi.com/2016/02/02/deep-learning-with-caffe-in-python-part-i-defining-a-layer/
	http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
'''

import caffe

# tell caffe we want to use GPU
caffe.set_device(0)
caffe.set_mode_gpu()

# get the net from prototxt file
net = caffe.Net('tutorial1convnet.prototxt', caffe.TEST)

print("\nnet.inputs:", net.inputs)
print("\ndir(net.blobs):", dir(net.blobs))
print("\ndir(net.params):", dir(net.params))
print("\nnet.blobs['data']:", net.blobs['data'])
print("\nnet.blobs['data'].data:", net.blobs['data'].data)
print("\nnet.blobs['conv'].data.shape:", net.blobs['conv'].data.shape)
for i in range(0, len(net.params['conv'])):
	print("\nnet.params['conv']["+str(i)+"]:", net.params['conv'][i])
