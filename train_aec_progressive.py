import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import glob
from model_pytorch import Encoder, Decoder, preprocess
import numpy as np

device = "cpu"

mse = torch.nn.MSELoss()
TRAINING_IMG_DIM = (512, 512)
TRAINING_CROP_SIZE = (256, 256)
BATCH_SIZE = 8


class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.img_paths = glob.glob(os.path.join(root_dir, "*"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        img = torch.load(img_path)
        img.to(device)
        assert img.shape == (
            3,
            TRAINING_CROP_SIZE[0],
            TRAINING_CROP_SIZE[1],
        ), f"Image shape mismatch: {img.shape}, {img_path}"
        return img


class AEC(torch.nn.Module):
    def __init__(self):
        super(AEC, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.to(device=device)


def forward_pass(aec, block, x, encode=True, skips=None):
    if skips is None:
        skips = []
    features = []
    if encode:
        start = 0
        stop = block + 1
        step = 1
    else:
        start = block
        stop = -1
        step = -1

    # (0) enc1 skip, 
    # (1) enc2 skip, 
    # (2) enc3 skip
    # i = 3
    # x = enc3 output
    # x + enc3 skip -> dec3
    for i in range(start, stop, step):
        if encode:
            output = aec.encoder(i, x)
            if i == 0:
                x = output
            else:
                x, skip = output
                skips.append(skip)
            features.append(x)
        else:
            if i == 0:
                x = aec.decoder(i, x)
            else:
                x = aec.decoder(i, x, skips[i - 1])
            features.append(x)
    return x, features, skips


def train_block(aec, block, x):
    input_img = x
    x, encoded, skips = forward_pass(aec, block, x)
    x, decoded, _ = forward_pass(aec, block, x, encode=False, skips=skips)

    reconstruction_loss = mse(input_img, x)

    if block == 0:
        return [reconstruction_loss]

    feature_loss = mse(encoded[-2], decoded[0])

    x_reencoded, _, _ = forward_pass(aec, block, x)
    perceptual_loss = mse(encoded[-1], x_reencoded)
    return (reconstruction_loss, feature_loss, perceptual_loss)


if __name__ == "__main__":
    dataset = MSCOCO("tensor_dataset")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    aec = AEC()
    aec.compile()

    for block in range(4):
        trainable_params = list(aec.decoder.blocks[block].parameters())

        optimizer = optim.Adam(trainable_params, lr=1e-4)
        aec.train()
        with torch.no_grad():
            x = torch.randn(1, 3, 256, 256).to(device)
            _ = train_block(aec, 0, x)
        print("warmup done")

        for epoch in range(2):
            print("starting poch")
            epoch_losses = []
            # imgs = torch.randn(10, 5,3,224,224)
            for i, batch in enumerate(dataloader):
                print("bax", i)

                optimizer.zero_grad()
                losses = train_block(aec, block, batch)
                print(losses)
                total_loss = sum(losses)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                epoch_losses.append(total_loss)
                print(total_loss)
            print(epoch_losses)
