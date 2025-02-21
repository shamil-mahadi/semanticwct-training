import tensorflow as tf
import numpy as np
import time

def inv_sqrt_cov_original(cov, inverse=False):
    """Original implementation"""
    # Ensure tf.eye has the same dtype as cov
    s, u, _ = tf.linalg.svd(cov + tf.eye(tf.shape(cov)[-1], dtype=cov.dtype))
    # Original logic calculates a single scalar n_s for the entire batch
    n_s = tf.reduce_sum(tf.cast(tf.greater(s, 1e-5), tf.int32))
    s = tf.sqrt(s[:,:n_s])
    if inverse:
        s = 1 / s
    d = tf.linalg.diag(s)
    u = u[:,:,:n_s]
    return tf.matmul(u, tf.matmul(d, u, adjoint_b=True))

"""def inv_sqrt_cov_optimized(cov, inverse=False, threshold=1e-5):
    # Regularize covariance matrix
    reg_cov = cov + tf.eye(tf.shape(cov)[-1], dtype=cov.dtype)

    # Use eigendecomposition (more efficient for symmetric matrices)
    # s will have shape (batch_size, num_eigenvalues)
    # u will have shape (batch_size, num_eigenvalues, num_eigenvalues)
    s, u = tf.linalg.eigh(reg_cov)

    # Replicate the original function's behavior of calculating a single scalar n_s
    # for the entire batch to ensure comparable results.
    n_s = tf.reduce_sum(tf.cast(tf.greater(s, threshold), tf.int32))

    # Slice s and u using this global n_s
    s_truncated = s[:, :n_s]
    u_truncated = u[:, :, :n_s]

    # Compute sqrt or inverse sqrt
    if inverse:
        s_transformed = tf.math.rsqrt(s_truncated)
    else:
        s_transformed = tf.sqrt(s_truncated)

    # Reconstruct the matrix
    # tf.linalg.diag expects a 1D or 2D tensor for the diagonal.
    # If s_transformed is (B, N_s), tf.linalg.diag(s_transformed) will create a (B, N_s, N_s) diagonal matrix.
    d = tf.linalg.diag(s_transformed)

    return tf.matmul(u_truncated, tf.matmul(d, u_truncated, adjoint_b=True))
"""
import numpy as np

import numpy as np

import numpy as np

import numpy as np

import numpy as np

def inv_sqrt_cov_optimized(cov, inverse=False):
    # Ensure input is numpy array
    if not isinstance(cov, np.ndarray):
        cov = cov.numpy()

    # Add identity - broadcast to shape if needed
    I = np.eye(cov.shape[-1], dtype=cov.dtype)
    if cov.ndim == 2 and cov.shape[0] != cov.shape[1]:
        # For rectangular, add identity accordingly
        # For safety, add identity of shape (min_dim, min_dim)
        min_dim = min(cov.shape)
        I = np.eye(min_dim, dtype=cov.dtype)
        cov_plus_eye = cov + I
    else:
        cov_plus_eye = cov + I

    U, S, Vt = np.linalg.svd(cov_plus_eye, full_matrices=False)

    # Filter small singular values like original
    mask = S > 1e-5
    S = S[mask]
    U = U[:, mask]

    S = np.sqrt(S)
    if inverse:
        S = 1.0 / S

    return U @ np.diag(S) @ U.T



def benchmark_functions(matrix_sizes, num_runs=100):
    """Benchmark both implementations"""
    results = {}

    for size in matrix_sizes:
        print(f"\nTesting matrix size: {size}x{size}")

        # Generate random positive definite covariance matrix
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32)
        cov_np = A @ A.T + 0.1 * np.eye(size)  # Ensure positive definite
        cov = tf.constant(cov_np)

        # Add batch dimension for testing
        cov_batch = tf.expand_dims(cov, 0)

        # Warm up
        _ = inv_sqrt_cov_original(cov_batch, inverse=False)
        _ = inv_sqrt_cov_optimized(cov_batch, inverse=False)

        # Benchmark original
        start_time = time.time()
        for _ in range(num_runs):
            result_orig = inv_sqrt_cov_original(cov_batch, inverse=False)
        orig_time = (time.time() - start_time) / num_runs

        # Benchmark optimized
        start_time = time.time()
        for _ in range(num_runs):
            result_opt = inv_sqrt_cov_optimized(cov_batch, inverse=False)
        opt_time = (time.time() - start_time) / num_runs

        # Test with inverse=True as well
        start_time = time.time()
        for _ in range(num_runs):
            result_orig_inv = inv_sqrt_cov_original(cov_batch, inverse=True)
        orig_time_inv = (time.time() - start_time) / num_runs

        start_time = time.time()
        for _ in range(num_runs):
            result_opt_inv = inv_sqrt_cov_optimized(cov_batch, inverse=True)
        opt_time_inv = (time.time() - start_time) / num_runs

        # Check if outputs match
        diff = tf.reduce_max(tf.abs(result_orig - result_opt))
        diff_inv = tf.reduce_max(tf.abs(result_orig_inv - result_opt_inv))

        results[size] = {
            'orig_time': orig_time * 1000,  # Convert to ms
            'opt_time': opt_time * 1000,
            'orig_time_inv': orig_time_inv * 1000,
            'opt_time_inv': opt_time_inv * 1000,
            'speedup': orig_time / opt_time,
            'speedup_inv': orig_time_inv / opt_time_inv,
            'max_diff': float(diff.numpy()),
            'max_diff_inv': float(diff_inv.numpy()),
            'outputs_match': float(diff.numpy()) < 1e-5,
            'outputs_match_inv': float(diff_inv.numpy()) < 1e-5
        }

        print(f"Original time (inverse=False): {orig_time*1000:.3f} ms")
        print(f"Optimized time (inverse=False): {opt_time*1000:.3f} ms")
        print(f"Speedup (inverse=False): {orig_time/opt_time:.2f}x")
        print(f"Original time (inverse=True): {orig_time_inv*1000:.3f} ms")
        print(f"Optimized time (inverse=True): {opt_time_inv*1000:.3f} ms")
        print(f"Speedup (inverse=True): {orig_time_inv/opt_time_inv:.2f}x")
        print(f"Max difference (inverse=False): {float(diff.numpy()):.2e}")
        print(f"Max difference (inverse=True): {float(diff_inv.numpy()):.2e}")
        print(f"Outputs match (inverse=False): {float(diff.numpy()) < 1e-5}")
        print(f"Outputs match (inverse=True): {float(diff_inv.numpy()) < 1e-5}")

    return results

def test_correctness():
    """Test correctness with various edge cases"""
    print("Testing correctness with edge cases...")

    # Test 1: Small matrix
    cov_small = tf.constant([[2.0, 1.0], [1.0, 2.0]], dtype=tf.float32)
    cov_small = tf.expand_dims(cov_small, 0)

    orig_result = inv_sqrt_cov_original(cov_small, inverse=False)
    opt_result = inv_sqrt_cov_optimized(cov_small, inverse=False)
    print(f"Small matrix test - Max diff: {tf.reduce_max(tf.abs(orig_result - opt_result)):.2e}")

    # Test 2: Nearly singular matrix
    A = tf.constant([[1.0, 0.999], [0.999, 1.0]], dtype=tf.float32)
    A = tf.expand_dims(A, 0)

    orig_result = inv_sqrt_cov_original(A, inverse=False)
    opt_result = inv_sqrt_cov_optimized(A, inverse=False)
    print(f"Nearly singular matrix test - Max diff: {tf.reduce_max(tf.abs(orig_result - opt_result)):.2e}")

    # Test 3: Identity matrix
    eye = tf.expand_dims(tf.eye(3, dtype=tf.float32), 0)

    orig_result = inv_sqrt_cov_original(eye, inverse=False)
    opt_result = inv_sqrt_cov_optimized(eye, inverse=False)
    print(f"Identity matrix test - Max diff: {tf.reduce_max(tf.abs(orig_result - opt_result)):.2e}")

if __name__ == "__main__":
    print("=== TensorFlow inv_sqrt_cov Optimization Benchmark ===")

    # Test correctness first
    test_correctness()

    # Benchmark different matrix sizes
    matrix_sizes = [10, 50, 100, 200, 500]
    num_runs = 50 if max(matrix_sizes) > 200 else 100

    print(f"\n=== Benchmarking with {num_runs} runs per test ===")
    results = benchmark_functions(matrix_sizes, num_runs)

    print("\n=== Summary ===")
    print(f"{'Size':<6} {'Speedup':<8} {'Speedup(inv)':<12} {'Match':<6} {'Match(inv)':<10}")
    print("-" * 50)
    for size, result in results.items():
        print(f"{size:<6} {result['speedup']:<8.2f} {result['speedup_inv']:<12.2f} "
              f"{result['outputs_match']:<6} {result['outputs_match_inv']:<10}")

    avg_speedup = np.mean([r['speedup'] for r in results.values()])
    avg_speedup_inv = np.mean([r['speedup_inv'] for r in results.values()])
    all_match = all(r['outputs_match'] and r['outputs_match_inv'] for r in results.values())

    print(f"\nAverage speedup (inverse=False): {avg_speedup:.2f}x")
    print(f"Average speedup (inverse=True): {avg_speedup_inv:.2f}x")
    print(f"All outputs match exactly: {all_match}")