# save_tf_encoded.py

import numpy as np
import tensorflow as tf
from model_tf import VggEncoder as TFEncoder

def main():
    # 1) Fix seeds
    np.random.seed(0)
    tf.random.set_seed(0)

    # 2) Load exactly the same input image created by the PyTorch script
    input_np = np.load("input.npy").astype(np.float32)  # shape = (1, 224, 224, 3)

    # 3) Instantiate the TensorFlow encoder
    tf_encoder = TFEncoder()
    tf_encoder.load_weights("ckpts/ckpts-conv/encoder")

    # 4) Convert to a TF tensor (NHWC)
    input_tf = tf.convert_to_tensor(input_np)
    """
    dummy_input = tf.random.normal([1, 224, 224, 3])
    _ = tf_encoder(0, dummy_input)
    _,skip = tf_encoder(1, _)
    _, skip = tf_encoder(2, _)
    _ = tf_encoder(3, _)"""

    def zero_weights_tf(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):  # recurse into nested models
                zero_weights_tf(layer)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                w_shape = layer.kernel.shape
                b_shape = layer.bias.shape if layer.use_bias else None
                layer.set_weights([
                    tf.zeros(w_shape),
                    tf.zeros(b_shape) if b_shape is not None else None
                ])

    #zero_weights_tf(tf_encoder)

    # 5) Run through TF encoder blocks 0→3, collecting features & skips
    out_tf = input_tf
    for block_idx in range(4):
        if block_idx == 0:
            feat_tf = tf_encoder.call(block_idx, out_tf)  # returns one tensor
            out_tf = feat_tf
        else:
            feat_tf, skip_tf = tf_encoder.call(block_idx, out_tf)
            out_tf = feat_tf

    # 6) Now feat_tf is block 3 output (shape = [1, 28, 28, 512])
    tf_encoded = feat_tf.numpy()
    print(tf_encoded)
    np.save("tf_encoded.npy", tf_encoded)
    print(f"Saved tf_encoded.npy (shape {tf_encoded.shape})")

if __name__ == "__main__":
    # Optional: disable GPU for TF to reduce nondeterminism
    # tf.config.set_visible_devices([], "GPU")
    main()
