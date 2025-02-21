import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging
import hashlib
import time

# Define your constants (adjust these to match your original values)
TRAINING_IMG_DIM = (512, 512)  # Resize dimension
TRAINING_CROP_SIZE = (256, 256)  # Final crop size

# VGG19 normalization values (ImageNet pretrained)
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]

def get_safe_filename(img_path):
    """
    Generate a safe, unique filename for the output tensor
    
    Args:
        img_path (str): Original image path
        
    Returns:
        str: Safe filename for output tensor
    """
    # Get the full filename
    filename = os.path.basename(img_path)
    
    # Create a hash of the full path to ensure uniqueness
    # This handles cases where images in different subdirs have same names
    path_hash = hashlib.md5(img_path.encode()).hexdigest()[:8]
    
    # Remove extension and replace problematic characters
    safe_name = os.path.splitext(filename)[0]
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in safe_name)
    
    # Combine safe name with hash to ensure uniqueness
    return f"{safe_name}_{path_hash}.pt"

def process_single_image(img_path, output_dir, transform):
    """
    Process a single image through the pipeline and save as tensor
    
    Args:
        img_path (str): Path to input image
        output_dir (str): Directory to save tensor
        transform: torchvision transform pipeline
        
    Returns:
        tuple: (success_bool, img_path, error_message)
    """
    try:
        # Check if input file exists and is readable
        if not os.path.exists(img_path):
            return (False, img_path, "Input file does not exist")
        
        if not os.access(img_path, os.R_OK):
            return (False, img_path, "Input file is not readable")
        
        # Load and convert image
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Apply transforms
            img_tensor = transform(img)
        
        # Verify tensor properties
        if not isinstance(img_tensor, torch.Tensor):
            return (False, img_path, "Transform did not produce a tensor")
        
        expected_shape = (3, TRAINING_CROP_SIZE[0], TRAINING_CROP_SIZE[1])
        if img_tensor.shape != expected_shape:
            return (False, img_path, f"Shape mismatch: got {img_tensor.shape}, expected {expected_shape}")
        
        # Check for invalid values
        if torch.isnan(img_tensor).any():
            return (False, img_path, "Tensor contains NaN values")
        
        if torch.isinf(img_tensor).any():
            return (False, img_path, "Tensor contains infinite values")
        
        # Generate safe output filename
        safe_filename = get_safe_filename(img_path)
        output_path = os.path.join(output_dir, safe_filename)
        
        # Ensure output directory exists (in case of race condition during creation)
        os.makedirs(output_dir, exist_ok=True)
        
        # Atomic save to prevent race conditions
        pid = os.getpid()
        timestamp = int(time.time() * 1000000)  # Microsecond timestamp
        temp_path = f"{output_path}.tmp_{pid}_{timestamp}"
        
        try:
            # Save to temporary file
            torch.save(img_tensor, temp_path)
            
            # Verify the saved file
            if not os.path.exists(temp_path):
                return (False, img_path, "Temporary file was not created")
            
            # Atomic rename
            os.rename(temp_path, output_path)
            
            # Final verification
            if not os.path.exists(output_path):
                return (False, img_path, "Final output file was not created")
            
        except Exception as save_error:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors
            raise save_error
        
        return (True, img_path, None)
        
    except Exception as e:
        return (False, img_path, str(e))

def worker_init():
    """Initialize worker process - sets up transform pipeline"""
    global transform
    try:
        transform = transforms.Compose([
            transforms.Resize(TRAINING_IMG_DIM),
            transforms.ToTensor(),
            transforms.RandomCrop(TRAINING_CROP_SIZE),
            transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)
        ])
    except Exception as e:
        print(f"Failed to initialize worker: {e}")
        raise

def worker_process_image(args):
    """Worker function that uses global transform"""
    try:
        img_path, output_dir = args
        return process_single_image(img_path, output_dir, transform)
    except Exception as e:
        return (False, args[0] if args else "unknown", f"Worker error: {str(e)}")

def get_image_paths(input_dir):
    """
    Get all image paths from input directory
    
    Args:
        input_dir (str): Input directory path
        
    Returns:
        list: List of image file paths
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Get all image paths
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    img_paths = []
    
    for ext in img_extensions:
        # Add both lowercase and uppercase versions
        img_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    # Remove duplicates and sort
    img_paths = sorted(list(set(img_paths)))
    
    return img_paths

def validate_parameters(input_dir, output_dir, num_workers, batch_size):
    """Validate input parameters"""
    if not input_dir or not isinstance(input_dir, str):
        raise ValueError("input_dir must be a non-empty string")
    
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError("output_dir must be a non-empty string")
    
    if num_workers is not None:
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError("num_workers must be a positive integer or None")
    
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

def preprocess_images_multiprocess(input_dir, output_dir, num_workers=None, batch_size=100):
    """
    Process all images in input_dir through the pipeline using multiprocessing
    
    Args:
        input_dir (str): Path to directory containing images
        output_dir (str): Path to directory where tensors will be saved
        num_workers (int): Number of worker processes. If None, uses CPU count
        batch_size (int): Number of images to process in each batch
    """
    
    # Validate parameters
    validate_parameters(input_dir, output_dir, num_workers, batch_size)
    
    # Set number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"Using {num_workers} worker processes")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output directory {output_dir}: {e}")
    
    # Get all image paths
    try:
        img_paths = get_image_paths(input_dir)
    except Exception as e:
        raise ValueError(f"Error getting image paths: {e}")
    
    print(f"Found {len(img_paths)} images to process")
    
    if not img_paths:
        print("No images found!")
        return
    
    # Prepare arguments for worker processes
    worker_args = [(img_path, output_dir) for img_path in img_paths]
    
    # Process images in batches using multiprocessing
    successful = 0
    failed = 0
    failed_files = []
    
    try:
        # Use Pool with initializer to set up transform in each worker
        with mp.Pool(processes=num_workers, initializer=worker_init) as pool:
            # Process in batches to avoid memory issues
            for i in tqdm(range(0, len(worker_args), batch_size), desc="Processing batches"):
                batch = worker_args[i:i + batch_size]
                
                try:
                    # Process batch
                    results = pool.map(worker_process_image, batch)
                    
                    # Count results
                    for success, img_path, error in results:
                        if success:
                            successful += 1
                        else:
                            failed += 1
                            failed_files.append((img_path, error))
                            
                except Exception as batch_error:
                    print(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                    # Mark all images in this batch as failed
                    for img_path, _ in batch:
                        failed += 1
                        failed_files.append((img_path, f"Batch error: {str(batch_error)}"))
    
    except Exception as pool_error:
        raise RuntimeError(f"Multiprocessing pool error: {pool_error}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
    
    if failed_files:
        print(f"\nFirst 10 failed files:")
        for img_path, error in failed_files[:10]:
            print(f"  {os.path.basename(img_path)}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print(f"Output tensors saved to: {output_dir}")
    
    return successful, failed

def preprocess_images_single_thread(input_dir, output_dir):
    """
    Single-threaded version for comparison or debugging
    """
    # Validate parameters
    validate_parameters(input_dir, output_dir, 1, 1)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output directory {output_dir}: {e}")
    
    # Get all image paths
    try:
        img_paths = get_image_paths(input_dir)
    except Exception as e:
        raise ValueError(f"Error getting image paths: {e}")
    
    print(f"Found {len(img_paths)} images to process (single-threaded)")
    
    if not img_paths:
        print("No images found!")
        return 0, 0
    
    # Define the transform pipeline
    transform = transforms.Compose([
        transforms.Resize(TRAINING_IMG_DIM),
        transforms.ToTensor(),
        transforms.RandomCrop(TRAINING_CROP_SIZE),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)
    ])
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_path in tqdm(img_paths, desc="Processing images"):
        success, _, error = process_single_image(img_path, output_dir, transform)
        if success:
            successful += 1
        else:
            failed += 1
            if failed <= 10:  # Only print first 10 errors
                print(f"Failed to process {os.path.basename(img_path)}: {error}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
    
    return successful, failed

def load_tensor_example(tensor_path):
    """
    Example function to load and inspect a saved tensor
    
    Args:
        tensor_path (str): Path to saved tensor file
        
    Returns:
        torch.Tensor: Loaded tensor
    """
    try:
        tensor = torch.load(tensor_path, map_location='cpu')
        print(f"Loaded tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"Memory usage: {tensor.numel() * tensor.element_size() / 1024:.2f} KB")
        return tensor
    except Exception as e:
        print(f"Error loading tensor from {tensor_path}: {e}")
        return None

if __name__ == "__main__":
    # Set your paths here
    INPUT_DIR = "mscoco/train2017"  # Change this to your image directory
    OUTPUT_DIR = "tensor_dataset"  # Change this to your desired output directory
    
    # Choose processing method
    USE_MULTIPROCESSING = True  # Set to False for single-threaded processing
    NUM_WORKERS = None  # None = auto-detect CPU count, or specify number
    BATCH_SIZE = 500  # Process images in batches to manage memory
    
    try:
        if USE_MULTIPROCESSING:
            successful, failed = preprocess_images_multiprocess(
                INPUT_DIR, OUTPUT_DIR, NUM_WORKERS, BATCH_SIZE
            )
        else:
            successful, failed = preprocess_images_single_thread(INPUT_DIR, OUTPUT_DIR)
        
        # Optional: Load and inspect the first tensor
        output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.pt"))
        if output_files and successful > 0:
            print(f"\nExample tensor inspection:")
            load_tensor_example(output_files[0])
            
    except Exception as e:
        print(f"Error during processing: {e}")
        exit(1)
