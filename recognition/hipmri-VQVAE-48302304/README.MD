# HIP-MRI VQ-VAE(PyTorch)
Lightweight VQ-VAE with EMA codebook for HIP-MRI 2D slice reconstruction.Turn-key pipeline covering environment → training → inference → panel generation → run packaging with strong SSIM/PSNR under a small model.

## Project Overview(What&Why)
This repo implements a compact VQ-VAE that learns discrete latent codes for MRI slices.<br>
**Goals**: 
- robust reconstruction with limited compute; 
- stable quantizer training; 
-  fully reproducible scripts and artifacts for grading.

**Highlights**
> Repo path for this project:`recognition/hipmri-VAQVAE-48302304`



## Data Layout
Organize your dataset as 2D slices under three splits：
```bash
recognition/hipmri-VAQVAE-48302304/
└─ data_hipmri_2d/
   ├─ keras_slices_train/      # .png/.jpg/.jpeg or .nii/.nii.gz
   ├─ keras_slices_validate/   # same formats as above
   └─ keras_slices_test/       # same formats as above
```
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














