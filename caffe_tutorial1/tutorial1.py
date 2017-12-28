'''
referred from: 
	https://prateekvjoshi.com/2016/02/02/deep-learning-with-caffe-in-python-part-i-defining-a-layer/
	http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
'''

import caffe
import numpy as np
import cv2
from PIL import Image


# tell caffe we want to use GPU
caffe.set_device(0)
caffe.set_mode_gpu()

# get the net from prototxt file
net = caffe.Net('tutorial1convnet.prototxt', caffe.TEST)

# printing some...
print("\nnet.inputs:", net.inputs)
print("\ndir(net.blobs):", dir(net.blobs))
print("\ndir(net.params):", dir(net.params))
print("\nnet.blobs['data']:", net.blobs['data'])
print("\nnet.blobs['data'].data:", net.blobs['data'].data)
print("\nnet.blobs['conv'].data.shape:", net.blobs['conv'].data.shape)
for i in range(0, len(net.params['conv'])):
	print("\nnet.params['conv']["+str(i)+"]:", net.params['conv'][i])

img = np.array(Image.open('cat_gray.jpg'))		# open and load the image in img
img_input = img[np.newaxis, np.newaxis, :, :]	# reshapes the input to shape (1,1,360,480)
print(img_input)								# prints the pixels value (which have shape (1,1,360,480))
print(img_input.shape)							# prints (1,1,360,480)
print(*img_input.shape)							# prints 1,1,360,480
net.blobs['data'].reshape(*img_input.shape)		# reshape 'data' to current image dimensions 1,1,360,480
net.blobs['data'].data[...] = img_input			# then assign actual data of 'data' as pixels value

# now make a forward pass since we have assigned the data
net.forward()

# Now net.blobs['conv'] is filled with data, and the 10 activation maps (net.blobs['conv'].data[0,i]) can be plotted.
cv2.imwrite('output_image_' + str(0) + '.jpg', 255*net.blobs['conv'].data[0,0]) # using i=0 only, but can iterate from 0 to 9

# To save the net parameters i.e. net.params, just call :
net.save('tutorial1.caffemodel')
