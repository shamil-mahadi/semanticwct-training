import numpy as np

def main():
    # 1) Load both encoded feature maps
    torch_enc = np.load("torch_encoded.npy")  # shape = (1, 28, 28, 512)
    tf_enc    = np.load("tf_encoded.npy")     # shape = (1, 28, 28, 512)

    # 2) Sanity check: shapes must match
    if torch_enc.shape != tf_enc.shape:
        raise ValueError(f"Shape mismatch: torch {torch_enc.shape} vs tf {tf_enc.shape}")

    # 3) Compute difference metrics
    diff = torch_enc - tf_enc

    l2_diff = np.linalg.norm(diff.ravel())
    l2_tf   = np.linalg.norm(tf_enc.ravel())
    rel_percent = (l2_diff / (l2_tf + 1e-12)) * 100

    mae    = np.mean(np.abs(diff))
    max_abs = np.max(np.abs(diff))

    # 4) Stats on the original tensors
    def print_stats(name, arr):
        print(f"{name} stats:")
        print(f"  shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")
        print(f"  min:   {np.min(arr):.6e}")
        print(f"  max:   {np.max(arr):.6e}")
        print(f"  mean:  {np.mean(arr):.6e}")
        print(f"  std:   {np.std(arr):.6e}")

    print_stats("Torch encoded", torch_enc)
    print_stats("TF encoded", tf_enc)
    print_stats("Difference (torch - tf)", diff)

    # 5) Error distribution details
    abs_diff = np.abs(diff)
    zero_diff_count = np.sum(abs_diff == 0)
    small_diff_count = np.sum(abs_diff < 1e-6)
    large_diff_count = np.sum(abs_diff > 1.0)

    total_elements = diff.size

    print("\nDifference distribution:")
    print(f"  Total elements        : {total_elements}")
    print(f"  Exactly zero diff      : {zero_diff_count} ({100*zero_diff_count/total_elements:.4f}%)")
    print(f"  Diff < 1e-6           : {small_diff_count} ({100*small_diff_count/total_elements:.4f}%)")
    print(f"  Diff > 1.0            : {large_diff_count} ({100*large_diff_count/total_elements:.4f}%)")

    # 6) Optional: percentiles of absolute differences
    percentiles = [50, 75, 90, 95, 99, 99.9]
    abs_diff_percentiles = np.percentile(abs_diff, percentiles)

    print("\nAbsolute difference percentiles:")
    for p, val in zip(percentiles, abs_diff_percentiles):
        print(f"  {p}th percentile: {val:.6e}")

if __name__ == "__main__":
    main()
