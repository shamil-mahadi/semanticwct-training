# save_torch_encoded.py

import numpy as np
import torch
from model_pytorch import Encoder as TorchEncoder

def main():
    # 1) Fix seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # 2) Create a random “image” of shape (1, 224, 224, 3)
    input_np = np.random.randn(1, 224, 224, 3).astype(np.float32)

    # 3) Save the input so the TF script reuses exactly the same array
    np.save("input.npy", input_np)

    # 4) Instantiate the PyTorch encoder in eval mode
    encoder = TorchEncoder().eval()

    # 5) Convert input to PyTorch tensor (NCHW)
    input_torch = torch.from_numpy(input_np).permute(0, 3, 1, 2)  # shape = (1, 3, 224, 224)

    # 6) Run through the encoder, collecting features & skips
    #    We only need block 3’s “feat” output, so we can discard earlier skips
    out = input_torch
    for block_idx in range(4):
        if block_idx == 0:
            feat = encoder(block_idx, out)  # returns one tensor
            out = feat
        else:
            feat, skip = encoder(block_idx, out)
            prev_skip = skip
            out = feat

    # 7) Now ‘feat’ is block 3 output (shape = [1, 512, 28, 28])
    #    Convert to NHWC so it matches TF’s [1, 28, 28, 512]
    torch_encoded = feat.detach().cpu().numpy().transpose(0, 2, 3, 1)
    np.save("torch_encoded.npy", torch_encoded)
    print(f"Saved input.npy and torch_encoded.npy (shape {torch_encoded.shape})")

if __name__ == "__main__":
    main()
