# semanticwct-training

The training side of semanticwct, the wavelet colour transfer model that
[`../semanticwct`](../semanticwct) runs at inference. It has the TensorFlow/Keras reference I
started from, the PyTorch port I wrote by hand, the parity checks I used to convince myself the
port was numerically right, the progressive training procedure, and the dataset preprocessing
scripts.

The autoencoder design is the PhotoWCT2 one, which is what the reference implements. The port, the
parity harness and the dataset pipeline are mine.

## TensorFlow reference

| File | What it does |
|---|---|
| `model_tf.py` | Keras VGG-19 encoder and decoder, built from the layer specs in `models_tf/vgg_layers.csv` |
| `train_tf.py` | The reference training script, progressive decoder training. MS-COCO loader lives in `models_tf/DatasetAPI.py` |
| `models_tf/` | Keras model definitions (`model_conv.py`, `model_relu.py`), the dataset API, the VGG layer table |
| `ckpts/` | Trained TF checkpoints, `ckpts-conv/` and `ckpts-relu/`, encoder and decoder in each. Both loaded by prefix path |
| `benchmark_tf_svd.py` | The `inv_sqrt_cov` kernel, inverse square root of a covariance matrix. Correctness checks plus a speed table for sizes 10 to 500 |

## PyTorch port

| File | What it does |
|---|---|
| `model_pytorch.py` | `Encoder`, `Decoder` and `preprocess` |
| `train_semanticwct.py` | semanticwct training, four stages one at a time (`--pair`), checkpoints between them |
| `train_aec_progressive.py` | The block-wise autoencoder training. Reconstruction loss, feature matching against the encoder, and a perceptual loss from re-encoding the output. Only the decoder blocks for the stage being trained get updated, gradients clipped |
| `dataset_pytorch.py` | Two `Dataset` classes, one for images and one for already-preprocessed tensors |
| `test_reconstruction.py` | Quick encoder/decoder round trip at a few resolutions |

## Parity checks

A port isn't worth much unless you can show both frameworks agree, so there are four dump scripts
and two comparisons. Run them from the repo root in this order:

```bash
python dump_torch_features.py    # seeded input.npy and torch_encoded.npy
python dump_tf_features.py       # reads the same input.npy, writes tf_encoded.npy
python compare_features.py       # shapes, max abs diff, MSE, cosine

python dump_torch_weights.py     # writes torchw.npy
python dump_tf_weights.py        # writes tfw.npy, loads ./ckpts/ckpts-conv/encoder
python compare_weights.py        # transposes OIHW to HWIO, then compares
```

`input.npy`, `torch_encoded.npy`, `torchw.npy` and `tfw.npy` are committed from the original run.
`tf_encoded.npy` is not, so the TF side needs one run to produce it.

## Dataset pipeline

```bash
python dataset_build_tensors.py    # mscoco/train2017 into tensor_dataset/
python dataset_chunk_by_size.py    # tensor_dataset/ into chunks/, sized to sit in RAM
python dataset_rechunk.py          # tensor_dataset/ into rechunked_tensors/, 2000 samples each
python dataset_rechunk_resume.py   # the same, but resumable if it dies part way through
python dataset_check_chunks.py     # checks shapes and dtypes in chunks/
python dataset_check_contiguity.py # checks chunks are still contiguous after save and load
```

`dataset_build_tensors.py` resizes to 512, normalises with VGG mean and std and takes 256px random
crops, with multiprocessing and hashed filenames so a parallel run can't collide on names.

`benchmark_preprocessing.py` is what I used to check that three preprocessing implementations
(manual OpenCV, `torchvision.transforms` and `albumentations`) really do produce the same tensor
from a fixed crop seed. I wasn't going to trust the parallel version until they did.

## Notes

- Every script resolves paths against the working directory (`./ckpts`, `./models_tf/vgg_layers.csv`), so run them from the repo root.
- The `np.save` line in `dump_tf_weights.py` was commented out in my scratch copy. It's live here so the weight comparison works without editing anything.
- Needs either `tensorflow` and `keras` for the reference track, or `torch`, `torchvision`, `opencv-python`, `albumentations`, `tqdm` and `psutil` for the port and the dataset scripts.
- A few experiments and a half-finished second port aren't here. They never reached the final pipeline.
