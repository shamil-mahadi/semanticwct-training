import os
import torch

def check_chunks(chunk_dir, expected_shape=(3, 256, 256), expected_dtype=torch.float32):
    chunk_paths = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".pt")])
    bad_samples = []

    for chunk_idx, path in enumerate(chunk_paths):
        print(f"Checking {path} ...")
        try:
            chunk = torch.load(path, map_location="cpu")
            if not isinstance(chunk, torch.Tensor):
                raise TypeError(f"Expected Tensor in {path}, got {type(chunk)}")

            if chunk.dim() != 4:
                raise ValueError(f"Expected 4D Tensor (NCHW), got shape {chunk.shape} in {path}")

            for i, img in enumerate(chunk):
                if img.shape != expected_shape:
                    bad_samples.append((chunk_idx, i, "bad_shape", tuple(img.shape)))
                if img.dtype != expected_dtype:
                    bad_samples.append((chunk_idx, i, "bad_dtype", img.dtype))

        except Exception as e:
            bad_samples.append((chunk_idx, -1, "load_error", str(e)))

    if not bad_samples:
        print("✅ All chunks passed.")
    else:
        print("❌ Issues found:")
        for entry in bad_samples:
            print(f" - Chunk {entry[0]}, Index {entry[1]}, Issue: {entry[2]}, Detail: {entry[3]}")

    return bad_samples

print(check_chunks("chunks"))