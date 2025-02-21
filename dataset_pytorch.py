import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image
import glob
from torchvision import transforms

IMG_SIZE = (512, 512)
CROP_SIZE = [256, 256, 3]
BATCH_SIZE = 8


class MSCOCODataset(Dataset):
    def __init__(self, path, img_size=IMG_SIZE, crop_size=CROP_SIZE):
        self.img_size = img_size
        self.crop_size = crop_size
        
        # Get all image files
        self.img_paths = glob.glob(os.path.join(path, '*'))
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),  # Converts to [0, 1] range and CHW format
            transforms.RandomCrop((self.crop_size[0], self.crop_size[1]))
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Convert to PIL Image for transforms
        img = Image.fromarray(img)
        
        # Apply transforms
        img = self.transform(img)
        
        return img


# Alternative implementation using cv2 for resize (closer to original)
class MSCOCODatasetCV2(Dataset):
    def __init__(self, path, img_size=IMG_SIZE, crop_size=CROP_SIZE):
        self.img_size = img_size
        self.crop_size = crop_size
        self.img_paths = glob.glob(os.path.join(path, '*'))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Read and resize image (matching your cv2.resize)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size, cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Random crop
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size[0], self.crop_size[1]
        
        if h >= crop_h and w >= crop_w:
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            img = img[top:top + crop_h, left:left + crop_w]
        
        # Convert to tensor and change from HWC to CHW
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img


def mscoco_dataset(path, shuffle=True, batch_size=BATCH_SIZE, num_workers=4):
    """
    Create a PyTorch DataLoader for MSCOCO dataset
    
    Args:
        path: Path to image directory
        shuffle: Whether to shuffle the dataset
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
    
    Returns:
        DataLoader object
    """
    dataset = MSCOCODataset(path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=False   # Set to True if you need consistent batch sizes
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    # Create dataset
    data_path = "path/to/your/images"
    dataloader = mscoco_dataset(data_path, shuffle=True, batch_size=8)
    
    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: {batch.shape}")
        if batch_idx == 2:  # Just show first few batches
            break
    
    # Alternative: Direct dataset usage
    dataset = MSCOCODataset(data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Get a single sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")  # Should be [3, 256, 256]