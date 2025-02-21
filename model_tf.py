# PhotoWCT2: Photorealistic Wavelet Color Transfer - Model Definition
# This file implements the core neural network architecture for PhotoWCT2, a state-of-the-art method
# for photorealistic style transfer that combines wavelet transforms with neural style transfer.
#
# WHAT IS PhotoWCT2?
# PhotoWCT2 is an advanced neural style transfer technique that creates photorealistic stylized images.
# Unlike traditional neural style transfer that often produces painterly results, PhotoWCT2 maintains
# the photorealistic quality of the content image while applying the color and tone of a style image.
#
# KEY INNOVATIONS OF PhotoWCT2:
# 1. Wavelet Color Transfer (WCT): Matches color statistics in wavelet domain for better photorealism
# 2. Progressive Style Transfer: Applies style transfer at multiple resolution levels
# 3. Skip Connections: Preserves fine structural details from the content image
# 4. VGG-based Feature Extraction: Uses pre-trained VGG features for robust style representation

# IMPORTS SECTION
# These are the essential libraries we need for building the PhotoWCT2 neural network
import tensorflow as tf  # TensorFlow: Google's machine learning framework for PhotoWCT2 implementation
from tensorflow.keras import layers, Model  # Keras: High-level API for building the encoder-decoder architecture
import csv  # For reading VGG layer configuration from CSV file
import numpy as np  # NumPy: Numerical operations for wavelet transforms and color statistics

# LOADING VGG LAYER CONFIGURATION FOR PhotoWCT2
# PhotoWCT2 uses pre-trained VGG-19 features to extract style and content representations
# The CSV file contains the exact layer configuration from the original VGG-19 model
csvfile = open('./models_tf/vgg_layers.csv', newline='')  # VGG-19 layer definitions for PhotoWCT2
vgg_layers = csv.DictReader(csvfile)  # Parse VGG layers as dictionaries for easy access

# PhotoWCT2 VGG ENCODER CLASS DEFINITION
# The encoder extracts multi-scale features that will be used for wavelet color transfer
class VggEncoder(tf.keras.Model):
    """
    PhotoWCT2 VGG-based Encoder for multi-scale feature extraction.
    
    ROLE IN PhotoWCT2:
    The encoder is responsible for extracting hierarchical features from both content and style images.
    These features are used at different stages of the wavelet color transfer process:
    - Early layers (conv1_1): Capture low-level features like edges and textures
    - Middle layers (conv2_1, conv3_1): Capture mid-level features like patterns and shapes  
    - Deeper layers (conv4_1): Capture high-level semantic content
    
    WHY VGG FOR PhotoWCT2?
    - VGG features are well-suited for style transfer due to their hierarchical nature
    - Pre-trained on ImageNet, so they already understand natural image statistics
    - The layered structure allows for progressive style transfer at multiple scales
    - Skip connections from VGG encoder help preserve photorealistic details
    
    WAVELET CONNECTION:
    The features extracted here will later be processed through wavelet transforms
    to perform color transfer in the frequency domain, which preserves photorealism.
    """
    
    def __init__(self, nblocks=3):
        """
        Initialize the PhotoWCT2 VGG Encoder.
        
        PARAMETERS:
        nblocks (int): Number of encoding blocks for multi-scale processing (default=3)
                      PhotoWCT2 typically uses 3-4 blocks for optimal quality-speed tradeoff:
                      Block 0: Fine details and textures (conv1_1 level)
                      Block 1: Medium-scale patterns (conv2_1 level)  
                      Block 2: Large-scale structures (conv3_1 level)
                      Block 3: Semantic content (conv4_1 level)
        """
        super(VggEncoder, self).__init__()  # Initialize the parent Keras Model class
        
        # DEFINE PhotoWCT2 FEATURE EXTRACTION POINTS
        # These specific VGG layers are chosen for PhotoWCT2's multi-scale approach
        # Each layer captures features at a different level of abstraction
        output_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'][:nblocks+1]
        
        # INITIALIZE MULTI-SCALE BOTTLENECKS
        # PhotoWCT2 processes features at multiple scales simultaneously
        # Each bottleneck corresponds to one scale in the wavelet decomposition
        btnecks = [[] for _ in range(nblocks+1)]  # One bottleneck per scale level
        
        # INITIALIZE WEIGHT STORAGE FOR PhotoWCT2
        # Store VGG weights that will be frozen during PhotoWCT2 training
        # These pre-trained weights provide robust feature representations
        self.kernels = [{} for _ in range(len(output_layers))]  # Convolution filter weights
        self.biases = [{} for _ in range(len(output_layers))]   # Bias terms for each layer
        
        # BUILD PhotoWCT2 ENCODER FROM VGG CONFIGURATION
        # This creates the exact VGG architecture needed for PhotoWCT2 feature extraction
        for idx, vgg_layer in enumerate(vgg_layers):
            name = vgg_layer['name']        # Layer name from VGG (e.g., 'conv1_1', 'relu1_1')
            typename = vgg_layer['_typename']  # Layer type (e.g., 'nn.SpatialConvolution')
            
            # PhotoWCT2 PREPROCESSING LAYER
            # VGG preprocessing is crucial for PhotoWCT2 to work with the pre-trained features
            if idx == 0:
                name = 'preprocess'  # VGG preprocessing: scale to [0,255] and subtract BGR mean
                i = 0  # Start with the first bottleneck block
            
            # LAYER TYPE HANDLING FOR PhotoWCT2
            # Each layer type serves a specific purpose in PhotoWCT2's feature extraction pipeline
            
            # REFLECTION PADDING FOR PhotoWCT2
            # PhotoWCT2 uses reflection padding to avoid border artifacts in style transfer
            # This is especially important for photorealistic results
            if typename == 'nn.SpatialReflectionPadding':
                # Reflection padding mirrors border pixels instead of using zeros
                # This prevents color bleeding and maintains photorealistic boundaries
                btnecks[i].append(tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')))
                
            # CONVOLUTIONAL LAYERS FOR PhotoWCT2 FEATURE EXTRACTION
            # These layers extract the hierarchical features needed for wavelet color transfer
            elif typename == 'nn.SpatialConvolution':
                filters = int(vgg_layer['filters'])  # Number of feature maps at this level
                kernel_size = (int(vgg_layer['kernel_size']), int(vgg_layer['kernel_size']))  # Filter size (3x3 for VGG)
                # Create frozen VGG convolution layer for PhotoWCT2
                # trainable=False: Keep pre-trained VGG weights fixed during PhotoWCT2 training
                btnecks[i].append(tf.keras.layers.Conv2D(filters, kernel_size, padding='valid', name=name, trainable=False))
                
            # ReLU ACTIVATION FOR PhotoWCT2
            # ReLU activations help PhotoWCT2 capture non-linear feature relationships
            # Essential for representing complex style patterns
            elif typename == 'nn.ReLU':
                btnecks[i].append(tf.keras.layers.ReLU(name=name))
                
            # POOLING LAYERS FOR PhotoWCT2 MULTI-SCALE PROCESSING
            # PhotoWCT2 uses pooling to create features at different spatial scales
            # Note: Uses AveragePooling instead of MaxPooling for smoother feature transitions
            elif typename == 'nn.SpatialMaxPooling':
                # AveragePooling provides smoother downsampling for photorealistic style transfer
                # This helps maintain color coherence across different scales
                btnecks[i].append(tf.keras.layers.AveragePooling2D(name=name))
            else:
                # Unknown layer type - PhotoWCT2 requires specific VGG architecture
                raise NotImplementedError(typename)
            
            # TRACK PhotoWCT2 FEATURE EXTRACTION POINTS
            # Move to next bottleneck when we reach a target feature extraction layer
            if name in output_layers:
                i += 1  # Advance to next scale level
            
            # STOP AT PhotoWCT2 TARGET DEPTH
            # PhotoWCT2 doesn't need features deeper than conv4_1 for optimal results
            if name == output_layers[-1]:
                print("Reached target layer: {}".format(name))
                break
        
        # BUILD PhotoWCT2 MULTI-SCALE BOTTLENECKS
        # Each bottleneck processes features at one scale for the wavelet color transfer
        self.btnecks = []  # Store bottleneck models for each scale
        
        for i, layers in enumerate(btnecks):
            # DEFINE CHANNEL DIMENSIONS FOR PhotoWCT2 SCALES
            # Channel count increases with depth: 3(RGB) -> 64 -> 128 -> 256
            # This matches the VGG architecture that PhotoWCT2 is built upon
            shape = [3, 64, 128, 256][i]
            
            # CREATE SCALE-SPECIFIC INPUT
            # PhotoWCT2 processes variable-size images, so height/width are flexible
            input_tensor = tf.keras.layers.Input(shape=[None, None, shape])
            x = input_tensor  # Start processing at this scale
            
            # PROCESS THROUGH PhotoWCT2 SCALE LAYERS
            for layer in layers:
                # SPECIAL HANDLING FOR PhotoWCT2 SKIP CONNECTIONS
                # Skip connections are crucial for preserving photorealistic details
                if type(layer) is tf.keras.layers.AveragePooling2D:
                    skip = x  # Save high-resolution features before pooling
                    x = layer(x)  # Apply pooling (reduces spatial resolution)
                    # COMPUTE PhotoWCT2 SKIP CONNECTION
                    # Skip connections preserve fine details lost during downsampling
                    # This is essential for photorealistic reconstruction
                    skip = skip - tf.image.resize(x, tf.shape(skip)[1:3])
                else:
                    # Apply regular layers (convolution, ReLU, padding)
                    x = layer(x)
            
            # DEFINE PhotoWCT2 BOTTLENECK OUTPUTS
            # First bottleneck outputs only features, others include skip connections
            outputs = [x] if i == 0 else [x, skip]
            
            # CREATE PhotoWCT2 SCALE-SPECIFIC MODEL
            self.btnecks.append(tf.keras.Model(inputs=input_tensor, outputs=outputs))
    
    def call(self, block, input_tensor):
        """
        PhotoWCT2 forward pass through a specific encoding block.
        
        This method is called during PhotoWCT2's multi-scale processing pipeline.
        Different blocks extract features at different scales for wavelet color transfer.
        
        PARAMETERS:
        block (int): Which scale/block to process (0=finest, higher=coarser)
        input_tensor: Image or feature tensor to encode
        
        RETURNS:
        Encoded features (and skip connections for blocks > 0)
        
        PhotoWCT2 USAGE:
        - Block 0: Process original image for fine detail features
        - Block 1: Process downsampled features for medium-scale patterns
        - Block 2+: Process further downsampled features for coarse structures
        """
        return self.btnecks[block](input_tensor)


# PhotoWCT2 DECODER BOTTLENECK CREATION
def create_btneck(bt_layers, n_channels):
    """
    Creates a decoder bottleneck block for PhotoWCT2 reconstruction.
    
    ROLE IN PhotoWCT2:
    After wavelet color transfer is applied to the encoded features, the decoder
    reconstructs the stylized image. Each bottleneck handles reconstruction at
    one scale level, combining upsampled features with skip connections.
    
    WHY BOTTLENECKS IN PhotoWCT2?
    - Efficient processing at multiple scales
    - Skip connections preserve photorealistic details from content image
    - Gradual reconstruction prevents artifacts common in neural style transfer
    
    PARAMETERS:
    bt_layers: Sequence of layers for this reconstruction scale
    n_channels: [skip_channels, input_channels] for combining features
    
    RETURNS:
    Keras Model for one scale of PhotoWCT2 reconstruction
    """
    # DEFINE PhotoWCT2 DUAL INPUTS
    # input_layer: Stylized features from wavelet color transfer
    # skip_layer: Original content features to preserve photorealism
    input_layer = layers.Input(shape=(None, None, n_channels[1]))  # Stylized features
    skip_layer = layers.Input(shape=(None, None, n_channels[0]))   # Content preservation
    
    # PROCESS PhotoWCT2 STYLIZED FEATURES
    # Apply initial reconstruction layers to the stylized features
    x = bt_layers[0](input_layer)  # Usually ReLU activation
    x = bt_layers[1](x)            # Reflection padding for boundary handling
    x = bt_layers[2](x)            # Convolution for feature refinement
    x = bt_layers[3](x)            # Upsampling to increase resolution
    
    # APPLY PhotoWCT2 SKIP CONNECTION
    # This is where PhotoWCT2 preserves photorealistic details from the content image
    # Skip connections are essential for maintaining photorealism in the final result
    x = x + skip_layer
    
    # FINAL PhotoWCT2 RECONSTRUCTION PROCESSING
    # Apply remaining layers for final feature refinement
    for layer in bt_layers[4:]:
        x = layer(x)
    
    # RETURN PhotoWCT2 BOTTLENECK MODEL
    return tf.keras.Model(inputs=[input_layer, skip_layer], outputs=x)


# PhotoWCT2 VGG DECODER CLASS DEFINITION
class VggDecoder(tf.keras.Model):
    """
    PhotoWCT2 VGG-based Decoder for photorealistic image reconstruction.
    
    ROLE IN PhotoWCT2 PIPELINE:
    1. Receives features that have undergone wavelet color transfer
    2. Progressively reconstructs the image from coarse to fine scales
    3. Combines stylized features with content skip connections
    4. Produces final photorealistic stylized image
    
    KEY PhotoWCT2 DESIGN PRINCIPLES:
    - Multi-scale reconstruction matches the multi-scale wavelet processing
    - Skip connections preserve photorealistic content structure
    - Progressive upsampling avoids common style transfer artifacts
    - Reflection padding maintains natural image boundaries
    
    DIFFERENCE FROM STANDARD DECODERS:
    PhotoWCT2's decoder is specifically designed to work with wavelet-transformed
    features and maintain photorealistic quality, not just artistic stylization.
    """
    
    def __init__(self):
        """
        Initialize the PhotoWCT2 Decoder with multi-scale reconstruction blocks.
        
        The decoder architecture mirrors the encoder but in reverse, with
        additional skip connections for preserving photorealistic content.
        """
        super(VggDecoder, self).__init__()  # Initialize parent Keras Model
        self.btnecks = []  # Store PhotoWCT2 reconstruction bottlenecks
        
        # PhotoWCT2 FINAL OUTPUT BLOCK (Block 0)
        # This block produces the final photorealistic stylized RGB image
        n1 = 64  # Intermediate channel count before final RGB output
        self.btnecks.append(tf.keras.Sequential([
            layers.ReLU(),  # Final activation for color output
            # PhotoWCT2 reflection padding for natural image boundaries
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            # Final convolution: convert features to RGB image (64 -> 3 channels)
            layers.Conv2D(3, (3,3), padding='valid')  # Output RGB channels
        ]))
        
        # PhotoWCT2 RECONSTRUCTION BLOCK 1 (128 -> 64 channels)
        # Handles medium-scale feature reconstruction after wavelet processing
        n2 = 128  # Input channels from wavelet-processed features
        self.btnecks.append(create_btneck([
            layers.ReLU(),  # Activation for feature processing
            # PhotoWCT2 reflection padding for artifact-free reconstruction
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            # Feature refinement convolution (128 -> 64 channels)
            layers.Conv2D(n1, (3,3), padding='valid', activation='relu'),
            # PhotoWCT2 upsampling for progressive resolution increase
            layers.UpSampling2D(interpolation='bilinear'),  # Smooth upsampling for photorealism
            # Additional refinement after upsampling
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            layers.Conv2D(n1, (3,3), padding='valid', activation=None)
        ], [n1, n2]))  # Channel config: [skip=64, input=128]
        
        # PhotoWCT2 RECONSTRUCTION BLOCK 2 (256 -> 128 channels)
        # Handles coarse-scale feature reconstruction
        n3 = 256  # Input channels from deeper wavelet-processed features
        self.btnecks.append(create_btneck([
            layers.ReLU(),  # Activation for feature processing
            # PhotoWCT2 reflection padding
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            # Feature refinement (256 -> 128 channels)
            layers.Conv2D(n2, (3,3), padding='valid', activation='relu'),
            # Progressive upsampling for PhotoWCT2 multi-scale reconstruction
            layers.UpSampling2D(interpolation='bilinear'),
            # Post-upsampling refinement
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            layers.Conv2D(n2, (3,3), padding='valid', activation=None)
        ], [n2, n3]))  # Channel config: [skip=128, input=256]
        
        # PhotoWCT2 RECONSTRUCTION BLOCK 3 (512 -> 256 channels)
        # Handles the coarsest scale reconstruction from deepest VGG features
        n4 = 512  # Input channels from deepest VGG level
        self.btnecks.append(create_btneck([
            layers.ReLU(),  # Initial activation
            # First reconstruction convolution with padding
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
            # PhotoWCT2 progressive upsampling
            layers.UpSampling2D(interpolation='bilinear'),
            # Multiple refinement convolutions for high-quality reconstruction
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
            # Final convolution without activation (allows for residual learning)
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
            layers.Conv2D(n3, (3,3), padding='valid', activation=None)
        ], [n3, n4]))  # Channel config: [skip=256, input=512]
    
    def call(self, block, input_tensor, skip):
        """
        PhotoWCT2 forward pass through a specific decoder block.
        
        This method is called during PhotoWCT2's reconstruction phase, after
        wavelet color transfer has been applied to the encoded features.
        
        PARAMETERS:
        block (int): Which reconstruction block to use
                    0 = Final RGB output block
                    1,2,3 = Progressive reconstruction blocks (coarse to fine)
        input_tensor: Stylized features from wavelet color transfer
        skip: Skip connection from encoder (preserves content structure)
        
        RETURNS:
        Reconstructed features/image at the current scale
        
        PhotoWCT2 RECONSTRUCTION PROCESS:
        1. Start with deepest stylized features (block 3)
        2. Progressively reconstruct through blocks 2, 1
        3. Generate final RGB image (block 0)
        4. Skip connections preserve photorealistic content throughout
        """
        if block != 0:
            # FOR INTERMEDIATE BLOCKS: Combine stylized features with content skip connections
            # This is where PhotoWCT2 preserves the photorealistic structure of the content
            # while applying the color/tone from the style image
            
            # Alternative skip connection handling (commented out):
            # skip = tf.image.resize_with_crop_or_pad(skip, tf.shape(input_tensor)[1] * 2, tf.shape(input_tensor)[2] * 2)
            # This would resize skip connections to match upsampled feature dimensions
            
            # Combine stylized features with content preservation
            input_tensor = [input_tensor, skip]
            
        # FOR FINAL BLOCK (block 0): Only use stylized features, no skip needed
        # The final RGB output doesn't need skip connections
        
        # PROCESS THROUGH PhotoWCT2 RECONSTRUCTION BOTTLENECK
        bt_out = self.btnecks[block](input_tensor)
        return bt_out

# COMPLETE PhotoWCT2 ARCHITECTURE SUMMARY:
#
# PhotoWCT2 PIPELINE OVERVIEW:
# 1. ENCODING: VggEncoder extracts multi-scale features from content and style images
# 2. WAVELET COLOR TRANSFER: Features undergo wavelet-based color transfer (not in this file)
# 3. DECODING: VggDecoder reconstructs photorealistic stylized image
#
# PhotoWCT2 KEY INNOVATIONS IMPLEMENTED HERE:
# 
# 1. MULTI-SCALE PROCESSING:
#    - Encoder extracts features at multiple VGG levels (conv1_1 through conv4_1)
#    - Decoder reconstructs progressively from coarse to fine scales
#    - Enables better control over stylization at different detail levels
#
# 2. SKIP CONNECTIONS FOR PHOTOREALISM:
#    - Preserve fine structural details from content image
#    - Prevent over-stylization that destroys photorealistic quality
#    - Connect corresponding encoder and decoder levels
#
# 3. VGG-BASED FEATURE EXTRACTION:
#    - Uses pre-trained VGG-19 features (frozen weights)
#    - Provides robust, semantically meaningful feature representations
#    - Enables effective style transfer without training from scratch
#
# 4. REFLECTION PADDING:
#    - Prevents border artifacts common in style transfer
#    - Maintains natural image boundaries for photorealistic results
#    - Used throughout both encoder and decoder
#
# PhotoWCT2 ADVANTAGES OVER TRADITIONAL NEURAL STYLE TRANSFER:
# - Maintains photorealistic quality instead of painterly effects
# - Better preservation of content image structure and details
# - More stable and controllable stylization process
# - Works well with diverse content and style image types
#
# TYPICAL PhotoWCT2 USAGE:
# 1. Load content and style images
# 2. Extract multi-scale features using VggEncoder
# 3. Apply wavelet color transfer to match style statistics
# 4. Reconstruct stylized image using VggDecoder with skip connections
# 5. Result: Photorealistic image with transferred style characteristics

