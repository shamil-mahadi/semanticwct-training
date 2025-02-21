import numpy as np

# Load PyTorch weights
pt_weights = np.load("torchw.npy")
print("PyTorch weights shape:", pt_weights.shape)  # (64, 3, 3, 3)

# Transpose PyTorch weights to TensorFlow format
pt_weights_tf_format = pt_weights.transpose(2, 3, 1, 0)
print("PyTorch weights transposed to TF format:", pt_weights_tf_format.shape)  # (3, 3, 3, 64)

# Load TensorFlow weights
tf_weights = np.load("tfw.npy")
print("TensorFlow weights shape:", tf_weights.shape)  # (3, 3, 3, 64)

# Compare weights
diff = pt_weights_tf_format - tf_weights
max_diff = np.max(np.abs(diff))
min_diff = np.min(np.abs(diff))
mean_diff = np.mean(diff)
std_diff = np.std(diff)

print(f"Max abs difference: {max_diff}")
print(f"Min abs difference: {min_diff}")
print(f"Mean difference: {mean_diff}")
print(f"Std difference: {std_diff}")
