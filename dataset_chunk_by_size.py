import os
import torch
import glob
from tqdm import tqdm
import gc
import psutil
import math

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def estimate_tensor_size_gb(tensor):
    """Estimate tensor size in GB"""
    num_elements = tensor.numel()
    bytes_per_element = tensor.element_size()
    return (num_elements * bytes_per_element) / (1024**3)

def chunk_dataset(data_folder, output_folder, target_chunk_size_gb=2.0, max_ram_usage_gb=20.0):
    """
    Chunk .pt files into larger chunks for efficient loading
    
    Args:
        data_folder: Path to folder containing .pt files
        output_folder: Path to save chunked files
        target_chunk_size_gb: Target size for each chunk in GB
        max_ram_usage_gb: Maximum RAM to use during processing
    """
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all .pt files
    pt_files = glob.glob(os.path.join(data_folder, "*.pt"))
    print(f"Found {len(pt_files)} .pt files")
    
    if not pt_files:
        print("No .pt files found!")
        return
    
    # Sample a few files to estimate average size
    print("Estimating file sizes...")
    sample_files = pt_files[:min(10, len(pt_files))]
    total_sample_size = 0
    
    for file_path in tqdm(sample_files, desc="Sampling"):
        try:
            tensor = torch.load(file_path, map_location='cpu')
            total_sample_size += estimate_tensor_size_gb(tensor)
            del tensor
            gc.collect()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    avg_file_size_gb = total_sample_size / len(sample_files)
    estimated_total_size_gb = avg_file_size_gb * len(pt_files)
    files_per_chunk = max(1, int(target_chunk_size_gb / avg_file_size_gb))
    estimated_chunks = math.ceil(len(pt_files) / files_per_chunk)
    
    print(f"Average file size: {avg_file_size_gb:.3f} GB")
    print(f"Estimated total dataset size: {estimated_total_size_gb:.1f} GB")
    print(f"Files per chunk: {files_per_chunk}")
    print(f"Estimated number of chunks: {estimated_chunks}")
    print(f"Available RAM: {max_ram_usage_gb} GB")
    
    # Process files in chunks
    chunk_idx = 0
    current_chunk = []
    current_chunk_size = 0
    files_processed = 0
    
    # Main progress bar
    pbar = tqdm(total=len(pt_files), desc="Processing files")
    
    for file_path in pt_files:
        try:
            # Check memory usage
            current_memory = get_memory_usage()
            if current_memory > max_ram_usage_gb:
                print(f"\nMemory usage ({current_memory:.1f} GB) exceeds limit. Saving current chunk...")
                if current_chunk:
                    save_chunk(current_chunk, output_folder, chunk_idx)
                    chunk_idx += 1
                    current_chunk = []
                    current_chunk_size = 0
                    gc.collect()
            
            # Load tensor
            tensor = torch.load(file_path, map_location='cpu')
            tensor_size = estimate_tensor_size_gb(tensor)
            
            # Check if adding this tensor would exceed chunk size
            if (current_chunk_size + tensor_size > target_chunk_size_gb and 
                len(current_chunk) > 0):
                
                # Save current chunk
                save_chunk(current_chunk, output_folder, chunk_idx)
                chunk_idx += 1
                current_chunk = []
                current_chunk_size = 0
                gc.collect()
            
            # Add tensor to current chunk
            current_chunk.append(tensor)
            current_chunk_size += tensor_size
            files_processed += 1
            
            # Update progress
            pbar.set_postfix({
                'chunk': chunk_idx,
                'chunk_size': f'{current_chunk_size:.2f}GB',
                'memory': f'{get_memory_usage():.1f}GB'
            })
            pbar.update(1)
            
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            pbar.update(1)
            continue
    
    # Save final chunk
    if current_chunk:
        save_chunk(current_chunk, output_folder, chunk_idx)
        chunk_idx += 1
    
    pbar.close()
    print(f"\nCompleted! Created {chunk_idx} chunks")
    print(f"Processed {files_processed}/{len(pt_files)} files")
    
    # Print chunk summary
    print("\nChunk Summary:")
    for i in range(chunk_idx):
        chunk_path = os.path.join(output_folder, f"chunk_{i:04d}.pt")
        if os.path.exists(chunk_path):
            size_mb = os.path.getsize(chunk_path) / (1024**2)
            print(f"chunk_{i:04d}.pt: {size_mb:.1f} MB")

def save_chunk(chunk_data, output_folder, chunk_idx):
    """Save a chunk of tensors"""
    chunk_path = os.path.join(output_folder, f"chunk_{chunk_idx:04d}.pt")
    
    try:
        # Stack tensors if they have the same shape, otherwise save as list
        if len(chunk_data) > 1:
            try:
                # Try to stack if all tensors have same shape
                if all(t.shape == chunk_data[0].shape for t in chunk_data):
                    stacked_data = torch.stack(chunk_data)
                else:
                    stacked_data = chunk_data
            except:
                stacked_data = chunk_data
        else:
            stacked_data = chunk_data[0] if len(chunk_data) == 1 else chunk_data
        
        torch.save(stacked_data, chunk_path)
        print(f"\nSaved chunk_{chunk_idx:04d}.pt with {len(chunk_data)} tensors")
        
    except Exception as e:
        print(f"\nError saving chunk {chunk_idx}: {e}")

def main():
    # Configuration
    DATA_FOLDER = "tensor_dataset"  # Change this to your data folder
    OUTPUT_FOLDER = "chunks"         # Change this to desired output folder
    TARGET_CHUNK_SIZE_GB = 10.0             # Target size per chunk
    MAX_RAM_USAGE_GB = 2.0                # Maximum RAM to use (leave some buffer)
    
    print("Dataset Chunking Script")
    print("=" * 50)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Target chunk size: {TARGET_CHUNK_SIZE_GB} GB")
    print(f"Max RAM usage: {MAX_RAM_USAGE_GB} GB")
    print("=" * 50)
    
    # Validate input folder
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder '{DATA_FOLDER}' does not exist!")
        return
    
    # Start chunking
    chunk_dataset(
        data_folder=DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        target_chunk_size_gb=TARGET_CHUNK_SIZE_GB,
        max_ram_usage_gb=MAX_RAM_USAGE_GB
    )
    
    print("\nChunking completed!")

if __name__ == "__main__":
    main()