# HIP-MRI VQ-VAE(PyTorch)
Lightweight VQ-VAE with EMA codebook for HIP-MRI 2D slice reconstruction.Turn-key pipeline covering environment → training → inference → panel generation → run packaging with strong SSIM/PSNR under a small model.

## Project Overview(What&Why)
This repo implements a compact VQ-VAE that learns discrete latent codes for MRI slices.<br>
**Goals**: 
- robust reconstruction with limited compute; 
- stable quantizer training; 
-  fully reproducible scripts and artifacts for grading.

**Highlights**
- encoder/decoder with residual blocks.
- VectorQuantizerEMA (K=256 by default), warm-up schedule for stability.
- Ready-to-run scripts: train.py, predict.py
> Repo path for this project:`recognition/hipmri-VAQVAE-48302304`



## Data Layout
Organize  dataset as 2D slices under three splits：
```bash
recognition/hipmri-VAQVAE-48302304/
└─ data_hipmri_2d/
   ├─ keras_slices_train/      # .png/.jpg/.jpeg or .nii/.nii.gz
   ├─ keras_slices_validate/   # same formats as above
   └─ keras_slices_test/       # same formats as above
```
-Images are grayscale (1-channel). If you have RGB PNG/JPG, we use the first channel.
-NIfTI volumes (.nii/.nii.gz) are auto-normalized per file and converted to 2D (first slice/channel if 3D).
-If your raw data are 3D volumes, a typical preprocessing is slice extraction + per-slice min-max normalization to −1,1. You can follow the same naming pattern {case}_{week}_{slice}.nii(.gz|.png|.jpg) as in examples.
## Method
This project use a standard encoder-decoder CNN with stride-2 downsamples to map a 128*128 grayscale slice to a \( \beta \left\lVert \mathrm{sg}[z_e] - z_q \right\rVert^2 \)


## Environment
we recommend to set the conda:

```bash
conda create -y -n hipmri_vq python=3.10
conda activate hipmri_vq
#CUDA build (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nibabel numpy scipy scikit-image opencv-python matplotlib tqdm pyyaml

pip install torchmetrics==1.4.1

```

















