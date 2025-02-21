import os
import torch
from glob import glob

def check_chunk_contiguity(chunk_dir):
    bad = 0
    total = 0
    chunk_paths = sorted(glob(os.path.join(chunk_dir, "*.pt")))

    for chunk_path in chunk_paths:
        try:
            chunk = torch.load(chunk_path, map_location="cpu")
            if not isinstance(chunk, torch.Tensor):
                print(f"{chunk_path} is not a single tensor, skipping.")
                continue

            for i in range(chunk.size(0)):
                total += 1
                if not chunk[i].is_contiguous():
                    print(f"{chunk_path} index {i} is not contiguous")
                    bad += 1

        except Exception as e:
            print(f"Error loading {chunk_path}: {e}")

    print(f"Checked {total} tensors, {bad} non-contiguous")

# Example usage
check_chunk_contiguity("chunks")
