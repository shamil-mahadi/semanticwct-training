from models_tf.model_conv import VggEncoder
import numpy as np

a = VggEncoder()

a.load_weights("ckpts/ckpts-conv/encoder")
block = a.layers[0]
convlayer = block.layers[-1] #two convs in this block, first is custom, last is conv1_1 from vgg
#print(convlayer.name, convlayer.dtype, convlayer.input, convlayer.output)
#print(convlayer.kernel.shape)
w = convlayer.kernel.numpy()
pre = a.layers[0].layers[1]  # this is preprocess Conv2D
print(pre.kernel, pre.bias)

np.save("tfw.npy", w)