# HIP-MRI VQ-VAE(PyTorch)
This project aims to build a reproducible implementation of VQVAE model for 2D HIP MRI slices with training, inference,metric reporting(SSIM/PSNR),panel visualization, and run packaging.
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







