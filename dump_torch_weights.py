from model_pytorch import Encoder
import numpy as np

a = Encoder()
layer = a.features[0]
print(layer)
print(layer.weight.shape)
w = layer.weight.data.detach().cpu().numpy()
np.save("torchw.npy", w)
