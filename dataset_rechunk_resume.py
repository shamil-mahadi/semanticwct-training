import os
import glob
import torch
from tqdm import tqdm

# Define chunk size
CHUNK_SIZE = 2000

def get_last_chunk_info(output_dir):
    """
    Find the last chunk and determine how many tensors have been processed
    
    Args:
        output_dir (str): Directory containing existing chunks
        
    Returns:
        tuple: (last_chunk_number, total_tensors_processed)
    """
    chunk_files = glob.glob(os.path.join(output_dir, "chunk_*.pt"))
    
    if not chunk_files:
        return -1, 0
    
    # Sort to find the highest numbered chunk
    chunk_numbers = []
    for chunk_file in chunk_files:
        chunk_name = os.path.basename(chunk_file)
        # Extract number from chunk_XXXX.pt
        chunk_num = int(chunk_name.split('_')[1].split('.')[0])
        chunk_numbers.append(chunk_num)
    
    last_chunk_num = max(chunk_numbers)
    
    # Count total tensors processed
    total_tensors = (last_chunk_num + 1) * 2000
    
    return last_chunk_num, total_tensors

def resume_rechunk_tensors(input_dir, output_dir):
    """
    Resume rechunking process from where it left off
    
    Args:
        input_dir (str): Path to directory containing individual .pt files
        output_dir (str): Path to directory where chunked tensors are being saved
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find where we left off
    last_chunk_num, tensors_processed = get_last_chunk_info(output_dir)
    
    print(f"Found existing chunks up to chunk_{last_chunk_num:04d}.pt")
    print(f"Total tensors already processed: {tensors_processed}")
    
    # Get all .pt file paths
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
    pt_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(pt_files)} total .pt files")
    
    if tensors_processed >= len(pt_files):
        print("All files have already been processed!")
        return
    
    # Skip files that have already been processed
    remaining_files = pt_files[tensors_processed:]
    print(f"Resuming from file {tensors_processed}: {os.path.basename(remaining_files[0])}")
    print(f"Remaining files to process: {len(remaining_files)}")
    
    # Initialize chunking variables
    tensor_batch = []
    chunk_count = last_chunk_num + 1
    successful = 0
    failed = 0
    
    def save_chunk(tensors, chunk_idx):
        """Save a chunk of tensors"""
        tensor_stack = torch.stack(tensors)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.pt")
        torch.save(tensor_stack, chunk_path)
        print(f"Saved chunk {chunk_idx} with {len(tensors)} tensors to {chunk_path}")
    
    # Process remaining files
    for pt_file in tqdm(remaining_files, desc="Processing remaining tensor files"):
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
    
    print(f"\nResume rechunking complete!")
    print(f"Successfully processed: {successful} additional tensors")
    print(f"Failed to process: {failed} files")
    print(f"Final chunk number: {chunk_count - 1}")
    print(f"Total tensors now processed: {tensors_processed + successful}")

def verify_complete_chunks(output_dir, expected_total_files=None):
    """
    Verify all chunks and show final statistics
    
    Args:
        output_dir (str): Directory containing chunk files
        expected_total_files (int): Expected total number of original files (optional)
    """
    chunk_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*.pt")))
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    total_tensors = 0
    print(f"\nFinal verification - Found {len(chunk_files)} chunk files:")
    
    for chunk_file in chunk_files:
        try:
            tensors = torch.load(chunk_file)
            num_tensors = len(tensors)
            total_tensors += num_tensors
            chunk_name = os.path.basename(chunk_file)
            print(f"  {chunk_name}: {num_tensors} tensors")
            
        except Exception as e:
            print(f"  Error loading {chunk_file}: {str(e)}")
    
    print(f"\nTotal tensors across all chunks: {total_tensors}")
    
    if expected_total_files:
        print(f"Expected total files: {expected_total_files}")
        if total_tensors == expected_total_files:
            print("✓ All files successfully processed!")
        else:
            print(f"⚠ Mismatch: {expected_total_files - total_tensors} files missing")

if __name__ == "__main__":
    # Set your paths here (same as before)
    INPUT_DIR = "tensor_dataset"  # Directory with individual .pt files
    OUTPUT_DIR = "rechunked_tensors"  # Output directory for chunks
    
    # Resume the rechunking process
    resume_rechunk_tensors(INPUT_DIR, OUTPUT_DIR)
    
    # Verify the final results
    print("\n" + "="*60)
    verify_complete_chunks(OUTPUT_DIR, expected_total_files=118287)  # Update this number if different
