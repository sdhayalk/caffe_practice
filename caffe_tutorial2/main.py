import caffe
import numpy as np
import matplotlib.pyplot as plt
import os

def download_model():
	'''	this function does not work (maybe because of my OS is Windows. i have just added so that you can be familiar of using the script to download. 
	   	Therefore download it manually from : http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
	'''
	if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
		print('CaffeNet found.')
	else:
	    print('Downloading pre-trained CaffeNet model...')
	    os.system('python /scripts/download_model_binary.py /models/bvlc_reference_caffenet')


caffe_root = "G:/DL/caffe-master/caffe-master/"	# caffe path
caffe.set_mode_gpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)     # params: model structure from .prototxt , model weights from .caffemodel , use test mode

'''
Now, 
	CaffeNet is configured to take images in BGR format. 
	Values are expected to start in the range [0, 255] 
	Mean ImageNet pixel value subtracted from them. 
	The channel dimension is expected as the first (outermost) dimension.
Therefore,
	As matplotlib will load images with values in the range [0, 1] in RGB format with the channel as the innermost dimension, we are arranging for the needed transformations here.
'''

mean_image = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')	# load the mean ImageNet image (as distributed with Caffe) for subtraction
mean_pixel = mean_image.mean(1).mean(1)  										# average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', mean_pixel)									# prints array([104.00968, 116.66876, 122.6789])

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})	# create transformer for the input called 'data'
transformer.set_transpose('data', (2,0,1))  								# move image channels to outermost dimension
transformer.set_mean('data', mean_pixel)            						# subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      								# rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  								# swap channels from RGB to BGR

net.blobs['data'].reshape(50, 3, 227, 227)  	# params: batch_size, num_channels, dim_1, dim_2

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')	# load the image
transformed_image = transformer.preprocess('data', image)			# transform the image
plt.imshow(image)													# show the image
net.blobs['data'].data[...] = transformed_image						# copy the image data into the memory allocated for the net

output = net.forward()		# forward pass

output_prob = output['prob'][0]  					# the output probability vector for the first image in the batch
print('predicted class is:', output_prob.argmax())	# predicted class number
