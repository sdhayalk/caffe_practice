import caffe

def display_stats(net):
	print("Network layers information:")
	for name, layer in zip(net._layer_names, net.layers):
	    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
	print("Network blobs information:")
	for name, blob in net.blobs.items():
	    print("{:<7}: {}".format(name, blob.data.shape))
	print('net.inputs:', net.inputs)
	print('net.outputs:', net.outputs)
