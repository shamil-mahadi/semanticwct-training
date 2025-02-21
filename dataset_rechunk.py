import os
import glob
import torch
from tqdm import tqdm

# Define chunk size
CHUNK_SIZE = 2000

def rechunk_tensors(input_dir, output_dir):
    """
    Read individual .pt tensor files and rechunk them into files of 2000 tensors each
    
    Args:
        input_dir (str): Path to directory containing individual .pt files
        output_dir (str): Path to directory where chunked tensors will be saved
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .pt file paths
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
    pt_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(pt_files)} .pt files to rechunk")
    
    if len(pt_files) == 0:
        print("No .pt files found in input directory!")
        return
    
    # Initialize chunking variables
    tensor_batch = []
    chunk_count = 0
    successful = 0
    failed = 0
    
    def save_chunk(tensors, chunk_idx):
        """Save a chunk of tensors"""
        tensor_stack = torch.stack(tensors)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.pt")
        torch.save(tensor_stack, chunk_path)
        print(f"Saved chunk {chunk_idx} with {len(tensors)} tensors to {chunk_path}")
    
    # Process each .pt file
    for pt_file in tqdm(pt_files, desc="Processing tensor files"):
        try:
            # Load tensor
            tensor = torch.load(pt_file)
            
            # Verify it's a single tensor (not already chunked)
            if tensor.dim() == 3:  # Expected: (3, H, W)
                tensor_batch.append(tensor)
                successful += 1
            elif tensor.dim() == 4:  # Might be a batch: (N, 3, H, W)
                # If it's already a batch, add each tensor individually
                for i in range(tensor.shape[0]):
                    tensor_batch.append(tensor[i])
                    successful += 1
                    
                    # Check if we need to save a chunk
                    if len(tensor_batch) >= CHUNK_SIZE:
                        save_chunk(tensor_batch, chunk_count)
                        tensor_batch = []
                        chunk_count += 1
            else:
                print(f"Unexpected tensor shape in {pt_file}: {tensor.shape}")
                failed += 1
                continue
            
            # Check if we need to save a chunk (for single tensors)
            if tensor.dim() == 3 and len(tensor_batch) >= CHUNK_SIZE:
                save_chunk(tensor_batch, chunk_count)
                tensor_batch = []
                chunk_count += 1
                
        except Exception as e:
            print(f"Failed to process {pt_file}: {str(e)}")
            failed += 1
    
    # Save remaining tensors if any
    if tensor_batch:
        save_chunk(tensor_batch, chunk_count)
        chunk_count += 1
    
    print(f"\nRechunking complete!")
    print(f"Successfully processed: {successful} tensors")
    print(f"Failed to process: {failed} files")
    print(f"Total chunks created: {chunk_count}")
    print(f"Output chunks saved to: {output_dir}")

def verify_chunks(output_dir):
    """
    Verify the created chunks and show statistics
    
    Args:
        output_dir (str): Directory containing chunk files
    """
    chunk_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*.pt")))
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    total_tensors = 0
    print(f"\nVerification - Found {len(chunk_files)} chunk files:")
    
    for chunk_file in chunk_files:
        try:
            tensors = torch.load(chunk_file)
            num_tensors = len(tensors)
            total_tensors += num_tensors
            chunk_name = os.path.basename(chunk_file)
            
            # Also verify tensor shapes
            if len(tensors) > 0:
                shape_info = f" (shape: {tensors[0].shape})"
            else:
                shape_info = " (empty)"
            
            print(f"  {chunk_name}: {num_tensors} tensors{shape_info}")
            
        except Exception as e:
            print(f"  Error loading {chunk_file}: {str(e)}")
    
    print(f"\nTotal tensors across all chunks: {total_tensors}")
    
    # Calculate expected vs actual chunks
    expected_full_chunks = total_tensors // CHUNK_SIZE
    remainder = total_tensors % CHUNK_SIZE
    expected_total_chunks = expected_full_chunks + (1 if remainder > 0 else 0)
    
    print(f"Expected chunks: {expected_total_chunks} ({expected_full_chunks} full + {1 if remainder > 0 else 0} partial)")
    print(f"Actual chunks: {len(chunk_files)}")
    
    if remainder > 0:
        print(f"Last chunk size: {remainder} tensors")

def load_sample_from_chunks(output_dir, chunk_idx=0, tensor_idx=0):
    """
    Load a sample tensor from the chunks for inspection
    
    Args:
        output_dir (str): Directory containing chunk files
        chunk_idx (int): Index of chunk to load from
        tensor_idx (int): Index of tensor within the chunk
    """
    chunk_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*.pt")))
    
    if chunk_idx >= len(chunk_files):
        print(f"Chunk index {chunk_idx} out of range. Found {len(chunk_files)} chunks.")
        return None
    
    chunk_path = chunk_files[chunk_idx]
    tensors = torch.load(chunk_path)
    
    if tensor_idx >= len(tensors):
        print(f"Tensor index {tensor_idx} out of range. Chunk has {len(tensors)} tensors.")
        return None
    
    tensor = tensors[tensor_idx]
    print(f"Sample tensor from {os.path.basename(chunk_path)}[{tensor_idx}]:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    
    return tensor

if __name__ == "__main__":
    # Set your paths here
    INPUT_DIR = "tensor_dataset"  # Directory with individual .pt files
    OUTPUT_DIR = "rechunked_tensors"  # Output directory for chunks
    
    # Rechunk the tensors
    rechunk_tensors(INPUT_DIR, OUTPUT_DIR)
    
    # Verify the results
    print("\n" + "="*60)
    verify_chunks(OUTPUT_DIR)
    
    # Load a sample tensor for inspection
    print("\n" + "="*60)
    print("SAMPLE TENSOR INSPECTION:")
    load_sample_from_chunks(OUTPUT_DIR, chunk_idx=0, tensor_idx=0)
