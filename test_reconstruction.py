# test_torch_encoder_decoder.py

import torch
from model_pytorch import Encoder, Decoder  # assumes model_pytorch.py is in the same directory

def test_torch_pipeline():
    # 1) Instantiate encoder & decoder
    encoder = Encoder().eval()   # freeze parameters
    decoder = Decoder().eval()

    # 2) Create a random “image” tensor: shape = [1, 3, 224, 224]
    x = torch.randn(1, 3, 300, 300)

    # 3) Run through each encoder block (0..3), storing features & skips
    features = {}
    skips = {}

    out = x
    for block_idx in range(4):
        if block_idx == 0:
            # Block 0 only takes x; returns a single tensor
            feat = encoder(block_idx, out)  # no skip for block 0
            # block 0’s “skip” is implicitly not used
        else:
            # For block > 0, pass in (prev_feat, prev_skip)
            feat, skip = encoder(block_idx, out)
            skips[block_idx] = skip

        features[block_idx] = feat
        out = feat  # feed into next block

        print(f"[Encoder] Block {block_idx} output shape: {feat.shape}")
        if block_idx > 0:
            print(f"           Skip {block_idx} shape: {skip.shape}")

    # 4) Run through decoder blocks in reverse (3→0)
    dec_in = features[3]
    for block_idx in reversed(range(4)):
        if block_idx == 0:
            # Block 0 only needs the “styled” features
            dec_out = decoder(block_idx, dec_in, None)
        else:
            # Blocks 1..3 need (feature, skip)
            dec_out = decoder(block_idx, dec_in, skips[block_idx])
        print(f"[Decoder] Block {block_idx} output shape: {dec_out.shape}")
        dec_in = dec_out  # feed into next (previous) block

    print(f"[Decoder] Final reconstructed image shape: {dec_out.shape}")

if __name__ == "__main__":
    test_torch_pipeline()
