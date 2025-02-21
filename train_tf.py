# PhotoWCT2 Training Script - Complete Beginner's Tutorial
# =======================================================
# 
# This script implements the training procedure for PhotoWCT2, a neural style transfer method
# that can transfer the style of one image to another while preserving photorealistic quality.
# 
# Key Concepts for Beginners:
# - Neural Style Transfer: A technique that combines the content of one image with the style of another
# - Encoder-Decoder Architecture: A network structure where the encoder compresses input into features,
#   and the decoder reconstructs the output from those features
# - VGG Network: A pre-trained convolutional neural network originally designed for image classification,
#   but here repurposed for feature extraction
# - Blockwise Training: Training the network in stages, one block at a time, rather than all at once

import numpy as np  # Library for numerical operations and array handling
import tensorflow as tf  # Google's machine learning framework
from models_tf.DatasetAPI import mscoco_dataset  # Custom function to load MS-COCO dataset
import os, sys  # Operating system interface and system-specific parameters
import argparse  # Command-line argument parsing library

# GPU Configuration
# =================
# These environment variables configure how TensorFlow uses the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # Use only GPU #2 (if you have multiple GPUs)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"  # Allow GPU memory to grow as needed instead of allocating all at once

# Loss Function Definition
# ========================
# Mean Squared Error (MSE) measures the average squared difference between predicted and actual values
# Formula: MSE = (1/n) * Σ(predicted - actual)²
# Lower MSE means better reconstruction quality
mse = tf.keras.losses.MeanSquaredError()

def train(enc_dec, block, input_img):
    """
    Core training function that performs forward pass through encoder-decoder network
    
    Parameters:
    -----------
    enc_dec : VggEncDec
        The encoder-decoder model containing both VGG encoder and decoder networks
    block : int
        Current block number being trained (0, 1, 2, or 3)
        Block 0 is closest to input, Block 3 is closest to bottleneck
    input_img : tf.Tensor
        Input image tensor with shape [batch_size, height, width, channels]
        Typically [batch_size, 256, 256, 3] for RGB images
    
    Returns:
    --------
    list of tf.Tensor
        Loss values for different training objectives
    """
    
    # Storage for encoder features at each level
    # These capture different levels of abstraction (edges, textures, objects, etc.)
    enc_feats = []
    
    # Storage for skip connections from encoder to decoder
    # Skip connections help preserve fine details during reconstruction
    # Initialize with None for block 0 (no skip connection at input level)
    skip_feats = [None]

    # Forward Pass Through Encoder
    # ============================
    
    # Process input through first encoder block (block 0)
    # This captures low-level features like edges and basic textures
    x = enc_dec.encoder(0, input_img)
    enc_feats.append(x)  # Store features for later comparison
    
    # Process through remaining encoder blocks (1 to current block)
    for l in range(1, block+1):
        # Each encoder block returns: [main_features, skip_connection_features]
        x = enc_dec.encoder(l, x)
        
        # Store the main features for reconstruction loss calculation
        enc_feats.append(x[0])
        
        # Store skip connection features for decoder
        # Skip connections preserve spatial information lost during downsampling
        skip_feats.append(x[1])
        
        # Continue with main feature path
        x = x[0]

    # Forward Pass Through Decoder
    # ============================
    
    # Storage for decoder features at each level
    dec_feats = []
    
    # Process through decoder blocks in reverse order (from bottleneck back to output)
    for l in reversed(range(block+1)):
        # Each decoder block upscales features and combines with skip connections
        # skip=skip_feats[l] provides fine-grained spatial information
        x = enc_dec.decoder(l, x, skip=skip_feats[l])
        dec_feats.append(x)  # Store for loss calculation

    # Loss Calculation
    # ================
    
    # Reconstruction Loss: How well can we reconstruct the original input?
    # Compares final decoder output with original input image
    # This ensures the network learns to preserve the overall image structure
    loss_rec = mse(input_img, dec_feats[-1])

    # For block 0 training, only reconstruction loss is needed
    if block == 0:
        return [loss_rec]

    # Feature Reconstruction Loss: Can we reconstruct encoder features in decoder?
    # Compares encoder features from previous block with first decoder features
    # This ensures decoder learns to properly interpret encoder representations
    loss_feat_rec = mse(enc_feats[-2], dec_feats[0])

    # Perceptual Loss: Do re-encoded features match original encoding?
    # Re-encode the decoder output and compare with original encoder features
    # This ensures semantic content is preserved through the encode-decode process
    x = enc_dec.encoder(0, x)  # Re-encode decoder output
    for l in range(1, block+1):
        x = enc_dec.encoder(l, x)
        x = x[0]
    loss_percept = mse(enc_feats[-1], x)  # Compare with original encoding

    # Return all three loss components for optimization
    return loss_rec, loss_percept, loss_feat_rec

# Command Line Interface Setup
# ============================
# argparse allows users to specify training parameters via command line
parser = argparse.ArgumentParser()

# Block Pair Selection
# ====================
'''
Architecture Explanation:
The VGG encoder is split into 4 blocks (enc0, enc1, enc2, enc3) from input to bottleneck:
- enc0: Processes raw input images, captures basic features (edges, colors)
- enc1: Captures more complex patterns (textures, simple shapes)  
- enc2: Captures object parts and spatial relationships
- enc3: Captures high-level semantic features (objects, scenes)

The decoder mirrors this with (dec3, dec2, dec1, dec0) from bottleneck to output:
- dec3: Begins reconstruction from semantic features
- dec2: Adds spatial and structural details
- dec1: Refines textures and patterns
- dec0: Produces final pixel-level output

Blockwise training trains these encoder-decoder pairs progressively:
N=0: Train enc0+dec0 (basic input/output handling)
N=1: Train enc1+dec1 (texture processing) 
N=2: Train enc2+dec2 (structural understanding)
N=3: Train enc3+dec3 (semantic processing)

Benefits of blockwise training:
1. Stable training: Each block learns one level of representation
2. Faster convergence: Smaller networks train faster
3. Better feature learning: Each block specializes in its abstraction level
4. Recovery from failures: Can restart from any block if training fails
'''
parser.add_argument('--pair', '-p', type=int, choices=[0, 1, 2, 3], default=0, 
                    help="which pair of an encoder block and a decoder block to begin training from")

# Training Duration Configuration
# ===============================
'''
Epoch: One complete pass through the entire training dataset
Default: 10 epochs per block pair (10, 10, 10, 10 for blocks 0, 1, 2, 3)

Why different blocks might need different epochs:
- Block 0: Simple features, might converge quickly
- Block 3: Complex features, might need more training

If starting from block N>0, earlier epoch values are ignored
Example: Starting from block 2 ignores first two epoch values
'''
parser.add_argument('--epochs', '-i', type=int, nargs=4, default=[10, 10, 10, 10],
                    help="the numbers of epochs for training of the four block pairs")

# Feature Type Selection
# ======================
'''
VGG network has two types of layers:
1. Convolutional layers (conv): Raw feature maps before activation
2. ReLU layers (relu): Feature maps after ReLU activation function

ReLU (Rectified Linear Unit): f(x) = max(0, x)
- Removes negative values, keeping only positive activations
- Creates sparser, more interpretable features
- Generally better for style transfer applications

Choice affects what features the decoder tries to reproduce:
- 'relu': Reproduce activated features (more common choice)
- 'conv': Reproduce raw convolutional features
'''
parser.add_argument('--feature', '-f', type=str, choices=['relu', 'conv'], default='relu')

# Model Paths Configuration
# =========================
parser.add_argument("--encoder", '-e', type=str, default='./ckpts/ckpts-relu/encoder',
                    help="the path to the pretrained VGG encoder")
parser.add_argument("--saveto", '-s', type=str, default='./saved_decoder/relu/',
                    help="the path to the directory where to save the trained decoder blocks")

# Dataset Configuration
# =====================
'''
MS-COCO (Microsoft Common Objects in Context):
- Large-scale dataset with ~120,000 training images
- Contains diverse scenes, objects, and compositions
- Good for learning general image representations
- Images are preprocessed to 256x256 pixels for training

Alternative datasets can be used, but preprocessing pipeline might need adjustment
'''
parser.add_argument("--dataset", '-d', type=str, default='./MSCOCO/train2017', 
                    help="the path to the training dataset (MSCOCO)")

# Parse command line arguments into args object
args = parser.parse_args()

# Dynamic Model Import
# ====================
# Import different model architectures based on feature type selection
if args.feature == 'relu':
    from models_tf.model_relu import VggDecoder, VggEncoder
else:
    from models_tf.model_conv import VggDecoder, VggEncoder

# Model Definition
# ================
class VggEncDec(tf.keras.Model):
    """
    Combined Encoder-Decoder Model for PhotoWCT2
    
    This class wraps both the VGG encoder and decoder into a single model.
    The encoder is pre-trained on ImageNet for feature extraction.
    The decoder is trained from scratch to reconstruct images from VGG features.
    """
    def __init__(self, enc_path):
        """
        Initialize the encoder-decoder model
        
        Parameters:
        -----------
        enc_path : str
            Path to pre-trained encoder weights
        """
        super(VggEncDec, self).__init__()
        
        # Initialize VGG encoder (will be loaded with pre-trained weights)
        self.encoder = VggEncoder()
        
        # Initialize decoder (will be trained from scratch)
        self.decoder = VggDecoder()
        
        # Load pre-trained encoder weights
        # These weights come from VGG trained on ImageNet classification
        self.encoder.load_weights(enc_path)

# Model Instantiation and Training Setup
# ======================================

# Create the encoder-decoder model with pre-trained encoder
enc_dec = VggEncDec(args.encoder)

# Define which blocks to train based on starting point
# range(args.pair, 4) creates sequence from start_block to 3
# Example: args.pair=1 gives blocks [1, 2, 3]
blocks = range(args.pair, 4)

# Load and preprocess the training dataset
# mscoco_dataset returns batched, preprocessed images ready for training
ds = mscoco_dataset(args.dataset)

# Checkpoint Recovery
# ===================
# If not starting from block 0, need to load previously trained decoder blocks
if args.pair != 0:
    # Load each previously trained decoder block
    for i in range(0, args.pair):
        # Create checkpoint object for decoder block i
        # tf.train.Checkpoint manages saving/loading of model weights
        ckpt = tf.train.Checkpoint(dec_block = enc_dec.decoder.btnecks[i])
        
        try:
            # Attempt to restore the latest checkpoint for this block
            # assert_existing_objects_matched() ensures all expected weights are found
            ckpt.restore(tf.train.latest_checkpoint(os.path.join(args.saveto, str(i)))).assert_existing_objects_matched()
        except:
            # If checkpoint loading fails, training cannot proceed
            print(f"The decoder block {i} has to be trained before training the decoder block {args.pair}.")
            quit()

# Main Training Loop
# ==================
# Train each block sequentially
for block in blocks:
    print(f"\n=== Training Block {block} ===")
    
    # Optimizer Configuration
    # =======================
    # Adam optimizer: Adaptive learning rate optimization algorithm
    # Combines benefits of AdaGrad and RMSProp
    # lr=1e-4: Learning rate of 0.0001 (small steps for stable training)
    lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Checkpoint Management
    # =====================
    # Set up checkpointing for this specific block
    # Saves both optimizer state and decoder block weights
    ckpt = tf.train.Checkpoint(optimizer=optimizer, dec_block=enc_dec.decoder.btnecks[block])
    
    # CheckpointManager handles automatic saving and cleanup
    # max_to_keep=1: Only keep the most recent checkpoint to save disk space
    manager = tf.train.CheckpointManager(ckpt, f"{os.path.join(args.saveto, str(block))}", max_to_keep=1)
    
    # Model Warm-up
    # =============
    # Run one forward pass with random data to initialize all variables
    # This ensures all tensor shapes and operations are properly set up
    # tf.random.normal([1, 256, 256, 3]): Random batch of 1 RGB image, 256x256 pixels
    train(enc_dec, block, tf.random.normal([1, 256, 256,3])) 
    
    # Get trainable weights for current decoder block
    # Only these weights will be updated during training
    weights = enc_dec.decoder.btnecks[block].trainable_weights
    
    # Training Loop for Current Block
    # ===============================
    for epoch in range(args.epochs[block]):
        print(f"Epoch {epoch+1}/{args.epochs[block]}")
        
        # Process each batch in the dataset
        for i, imgs in enumerate(ds):
            # Forward Pass and Loss Calculation
            # =================================
            # tf.GradientTape records operations for automatic differentiation
            with tf.GradientTape() as tape:
                # Compute all loss components for current block
                loss = train(enc_dec, block, imgs) 
                
                # Sum all loss components for total loss
                # This creates a single scalar value for gradient computation
                _loss = sum(loss)

            # Backward Pass and Weight Update
            # ===============================
            # Compute gradients of total loss with respect to trainable weights
            # Gradients indicate how much each weight should change to reduce loss
            grads = tape.gradient(_loss, weights)
            
            # Apply gradients to update weights using Adam optimizer
            # This is where the actual learning happens
            optimizer.apply_gradients(zip(grads, weights))
            
            # Progress Reporting
            # ==================
            # Print training progress every 10 iterations
            if (i+1) % 10 == 0:
                # Format loss values for display
                # .numpy() converts TensorFlow tensors to NumPy arrays for printing
                loss_values = ', '.join(map(lambda x: str(x.numpy()), loss))
                to_show = f"Epoch: {epoch+1}, iter: {i+1}, loss: {loss_values}"
                print(to_show)
        
        # Save checkpoint after each epoch
        # This allows recovery if training is interrupted
        manager.save()

    # Final Model Saving
    # ==================
    # After training the final block (block 3), save the complete decoder
    if block == 3:
        # Save entire decoder weights for inference use
        # This creates a complete model file that can be loaded independently
        enc_dec.decoder.save_weights(f"{os.path.join(args.saveto, 'decoder')}")
        print("Training completed! Full decoder model saved.")

# Training Complete
# =================
# At this point, the PhotoWCT2 decoder has been trained to reconstruct images
# from VGG encoder features. The model can now be used for style transfer by:
# 1. Encoding a content image with the VGG encoder
# 2. Modifying the encoded features (e.g., with WCT transformation)
# 3. Decoding the modified features back to an image with the trained decoder