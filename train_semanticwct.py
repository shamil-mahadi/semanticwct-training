# PhotoWCT2 PyTorch Training Script - Complete Beginner's Tutorial
# ================================================================
# 
# This script implements the training procedure for PhotoWCT2 in PyTorch, ported from TensorFlow.
# PhotoWCT2 is a neural style transfer method that can transfer the style of one image 
# to another while preserving photorealistic quality.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import argparse
from PIL import Image
import glob
from model_pytorch import Encoder, Decoder  # Import the provided model classes

# GPU Configuration
# =================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset Class for MS-COCO
# ==========================
class MSCOCODataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for loading MS-COCO images for training
    """
    def __init__(self, dataset_path, transform=None):
        """
        Initialize the dataset
        
        Parameters:
        -----------
        dataset_path : str
            Path to the MS-COCO training images directory
        transform : torchvision.transforms
            Image preprocessing transformations
        """
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Get all image files (jpg, jpeg, png)
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            self.image_paths.extend(glob.glob(os.path.join(dataset_path, ext)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {dataset_path}")
        
        print(f"Found {len(self.image_paths)} images in dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and preprocess an image
        
        Parameters:
        -----------
        idx : int
            Index of the image to load
            
        Returns:
        --------
        torch.Tensor
            Preprocessed image tensor
        """
        img_path = self.image_paths[idx]
        
        try:
            # Load image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random tensor if image loading fails
            return torch.randn(3, 256, 256)

# Loss Function Definition
# ========================
mse_loss = nn.MSELoss()

def train_step(enc_dec, block, input_img):
    """
    Core training function that performs forward pass through encoder-decoder network
    
    Parameters:
    -----------
    enc_dec : EncDecModel
        The encoder-decoder model containing both VGG encoder and decoder networks
    block : int
        Current block number being trained (0, 1, 2, or 3)
    input_img : torch.Tensor
        Input image tensor with shape [batch_size, channels, height, width]
    
    Returns:
    --------
    list of torch.Tensor
        Loss values for different training objectives
    """
    
    # Storage for encoder features at each level
    enc_feats = []
    
    # Storage for skip connections from encoder to decoder
    skip_feats = [None]

    # Forward Pass Through Encoder
    # ============================
    
    # Process input through first encoder block (block 0)
    x = enc_dec.encoder(0, input_img)
    enc_feats.append(x)
    
    # Process through remaining encoder blocks (1 to current block)
    for l in range(1, block + 1):
        result = enc_dec.encoder(l, x)
        
        if isinstance(result, tuple):
            # Unpack main features and skip connection
            x, skip = result
            enc_feats.append(x)
            skip_feats.append(skip)
        else:
            # Only main features returned
            x = result
            enc_feats.append(x)
            skip_feats.append(None)

    # Forward Pass Through Decoder
    # ============================
    
    dec_feats = []
    
    # Process through decoder blocks in reverse order
    for l in reversed(range(block + 1)):
        x = enc_dec.decoder(l, x, skip_feats[l])
        dec_feats.append(x)

    # Loss Calculation
    # ================
    
    # Reconstruction Loss: How well can we reconstruct the original input?
    loss_rec = mse_loss(input_img, dec_feats[-1])

    # For block 0 training, only reconstruction loss is needed
    if block == 0:
        return [loss_rec]

    # Feature Reconstruction Loss: Can we reconstruct encoder features in decoder?
    loss_feat_rec = mse_loss(enc_feats[-2], dec_feats[0])

    # Perceptual Loss: Do re-encoded features match original encoding?
    x_reenc = enc_dec.encoder(0, dec_feats[-1])
    for l in range(1, block + 1):
        x_reenc = enc_dec.encoder(l, x_reenc)
        if isinstance(x_reenc, tuple):
            x_reenc = x_reenc[0]
    
    loss_percept = mse_loss(enc_feats[-1], x_reenc)

    return [loss_rec, loss_percept, loss_feat_rec]

# Combined Encoder-Decoder Model
# ==============================
class EncDecModel(nn.Module):
    """
    Combined Encoder-Decoder Model for PhotoWCT2
    """
    def __init__(self, encoder_path=None):
        """
        Initialize the encoder-decoder model
        
        Parameters:
        -----------
        encoder_path : str
            Path to pre-trained encoder weights (optional)
        """
        super(EncDecModel, self).__init__()
        
        # Initialize encoder and decoder
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # Load pre-trained encoder weights if provided
        if encoder_path and os.path.exists(encoder_path):
            try:
                self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
                print(f"Loaded encoder weights from {encoder_path}")
            except Exception as e:
                print(f"Warning: Could not load encoder weights: {e}")
        
        # Move to device
        self.to(device)
        
        # Freeze encoder parameters (only train decoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

# Command Line Interface Setup
# ============================
def parse_arguments():
    parser = argparse.ArgumentParser(description='PhotoWCT2 PyTorch Training')
    
    parser.add_argument('--pair', '-p', type=int, choices=[0, 1, 2, 3], default=0,
                        help="which pair of encoder-decoder blocks to begin training from")
    
    parser.add_argument('--epochs', '-i', type=int, nargs=4, default=[10, 10, 10, 10],
                        help="number of epochs for training each of the four block pairs")
    
    parser.add_argument("--encoder", '-e', type=str, default='./checkpoints/encoder.pth',
                        help="path to the pretrained VGG encoder")
    
    parser.add_argument("--saveto", '-s', type=str, default='./saved_decoder/',
                        help="directory to save the trained decoder blocks")
    
    parser.add_argument("--dataset", '-d', type=str, default='./MSCOCO/train2017',
                        help="path to the training dataset (MSCOCO)")
    
    parser.add_argument("--batch_size", '-b', type=int, default=4,
                        help="batch size for training")
    
    parser.add_argument("--learning_rate", '-lr', type=float, default=1e-4,
                        help="learning rate for Adam optimizer")
    
    parser.add_argument("--num_workers", '-w', type=int, default=4,
                        help="number of workers for data loading")
    
    return parser.parse_args()

def create_data_loader(dataset_path, batch_size, num_workers):
    """
    Create data loader for training
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
    batch_size : int
        Batch size for training
    num_workers : int
        Number of workers for data loading
        
    Returns:
    --------
    DataLoader
        PyTorch DataLoader for training
    """
    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((286, 286)),  # Slightly larger for random crop
        transforms.RandomCrop((256, 256)),  # Random crop to 256x256
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and data loader
    dataset = MSCOCODataset(dataset_path, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    return dataloader

def save_checkpoint(model, optimizer, epoch, block, save_path):
    """
    Save training checkpoint
    
    Parameters:
    -----------
    model : nn.Module
        Model to save
    optimizer : torch.optim.Optimizer
        Optimizer to save
    epoch : int
        Current epoch
    block : int
        Current block being trained
    save_path : str
        Path to save checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'block': block,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load training checkpoint
    
    Parameters:
    -----------
    model : nn.Module
        Model to load weights into
    optimizer : torch.optim.Optimizer
        Optimizer to load state into
    checkpoint_path : str
        Path to checkpoint file
        
    Returns:
    --------
    int, int
        Starting epoch and block
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['block']
    return 0, 0

def main():
    """
    Main training function
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create save directory
    os.makedirs(args.saveto, exist_ok=True)
    
    # Create data loader
    print("Loading dataset...")
    dataloader = create_data_loader(args.dataset, args.batch_size, args.num_workers)
    print(f"Dataset loaded with {len(dataloader)} batches per epoch")
    
    # Create model
    print("Initializing model...")
    model = EncDecModel(args.encoder)
    
    # Define which blocks to train
    blocks = range(args.pair, 4)
    
    # Load previously trained blocks if not starting from block 0
    if args.pair != 0:
        for i in range(0, args.pair):
            block_checkpoint_path = os.path.join(args.saveto, f'block_{i}_final.pth')
            if os.path.exists(block_checkpoint_path):
                try:
                    # Load specific decoder block
                    checkpoint = torch.load(block_checkpoint_path, map_location=device)
                    model.decoder.blocks[i].load_state_dict(checkpoint['decoder_block_state_dict'])
                    print(f"Loaded decoder block {i} from {block_checkpoint_path}")
                except Exception as e:
                    print(f"Error loading decoder block {i}: {e}")
                    print(f"Block {i} must be trained before training block {args.pair}")
                    return
            else:
                print(f"Checkpoint for block {i} not found at {block_checkpoint_path}")
                print(f"Block {i} must be trained before training block {args.pair}")
                return
    
    # Main Training Loop
    # ==================
    for block in blocks:
        print(f"\n{'='*50}")
        print(f"Training Block {block}")
        print(f"{'='*50}")
        
        # Create optimizer for current decoder block only
        if hasattr(model.decoder.blocks[block], 'parameters'):
            trainable_params = list(model.decoder.blocks[block].parameters())
        else:
            # Fallback to all decoder parameters
            trainable_params = list(model.decoder.parameters())
        
        optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
        
        # Model warm-up (initialize with random data)
        model.train()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            try:
                _ = train_step(model, block, dummy_input)
                print("Model warm-up completed successfully")
            except Exception as e:
                print(f"Warning: Model warm-up failed: {e}")
        
        # Training loop for current block
        for epoch in range(args.epochs[block]):
            print(f"\nBlock {block}, Epoch {epoch+1}/{args.epochs[block]}")
            
            epoch_losses = []
            
            for batch_idx, images in enumerate(dataloader):
                # Move batch to device
                images = images.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass and loss calculation
                try:
                    losses = train_step(model, block, images)
                    total_loss = sum(losses)
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    
                    # Update weights
                    optimizer.step()
                    
                    # Track losses
                    epoch_losses.append([loss.item() for loss in losses])
                    
                    # Print progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        loss_str = ', '.join([f'{loss.item():.6f}' for loss in losses])
                        print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: [{loss_str}]")
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            # Calculate average losses for the epoch
            if epoch_losses:
                avg_losses = [sum(losses[i] for losses in epoch_losses) / len(epoch_losses) 
                             for i in range(len(epoch_losses[0]))]
                loss_str = ', '.join([f'{loss:.6f}' for loss in avg_losses])
                print(f"  Epoch {epoch+1} Average Loss: [{loss_str}]")
            
            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(args.saveto, f'block_{block}_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, block, checkpoint_path)
        
        # Save final block checkpoint
        final_checkpoint_path = os.path.join(args.saveto, f'block_{block}_final.pth')
        final_checkpoint = {
            'block': block,
            'decoder_block_state_dict': model.decoder.blocks[block].state_dict(),
            'full_model_state_dict': model.state_dict(),
        }
        torch.save(final_checkpoint, final_checkpoint_path)
        print(f"Block {block} training completed! Saved to {final_checkpoint_path}")
    
    # Save complete model after training all blocks
    if blocks and max(blocks) == 3:
        final_model_path = os.path.join(args.saveto, 'complete_model.pth')
        torch.save({
            'encoder_state_dict': model.encoder.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'full_model_state_dict': model.state_dict(),
        }, final_model_path)
        print(f"Complete model saved to {final_model_path}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()