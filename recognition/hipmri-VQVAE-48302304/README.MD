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
This project use a standard encoder-decoder CNN with stride-2 downsamples to map a 128*128 grayscale slice to a `z ∈ ℝ^{C × H' × W'}`
A VQEMA module replace each latent vector with its nearst code from a learnable codebook(size K), updated via exponential moving averages to avoid codebook collapse. The training objective is:
-  Image reconstruction loss (L1 by default).
- Commitment loss `β ‖sg[z_e] − z_q‖²` to encourage encoder outputs to stay near the chosen codewords.

We linearly warm-up the VQ loss weight (vq_w) for the first N steps to stabilize early training.
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

## Training
### **What this step does**
Train an EMA-VQ-VAE on 128×128 2D slices. The encoder maps images to a latent grid, the EMA codebook discretizes it, and the decoder reconstructs the slice. We log reconstruction loss (L1/L2), VQ commitment loss, codebook perplexity, and SSIM on the validation set.
``` bash
python -u train.py \
  --data_root data_hipmri_2d \
  --train_dir keras_slices_train \
  --val_dir   keras_slices_validate \
  --test_dir  keras_slices_test \
  --work_dir  workdir/hipmri_vqvae \
  --img_size 128 128 \
  --batch_size 48 \
  --num_workers 2 \
  --epochs 20 \
  --lr 5e-5 \
  --recon_loss l1 \
  --clip_norm 1.0 \
  --vq_warmup 2000 \
  --codebook_size 256 \
  --commit_beta 0.2 \
  --ema_decay 0.99
```
### **why these defaults**
- --recon_loss l1: L1 is more robust to outliers and tends to preserve edges in medical images.
- --vq_warmup 2000: Delay/anneal the VQ loss in early steps to avoid codebook collapse.
- --clip_norm 1.0: Stabilizes training with mixed batches and medical intensity ranges.
- --ema_decay 0.99: Standard EMA momentum for stable codebook updates.
- --commit_beta 0.2: A conservative commitment strength that works well for 64-D latents.

### Expect runtime & memory
- 128×128, batch 48 on a single A100 ~20 epochs in ~6–7 minutes per epoch; VRAM < 8–10 GB.
- BF16 mixed precision can further reduce memory; disable on older GPUs.

### Outputs format
- highest val_SSIM store at `workdir/hipmri_vqvae/best_vq.pt`.
- Curves and logs in `workdir/hipmri_vqvae/`.



## Predict(test&validate)
### Test set
``` bash
python -u predict.py \
  --data_root data_hipmri_2d \
  --input_dir keras_slices_test \
  --ckpt workdir/hipmri_vqvae/best_vq.pt \
  --out_dir workdir/hipmri_vqvae/predict_test_png \
  --img_size 128 128 \
  --batch_size 64 --num_workers 2 \
  --save_format png --metrics
```
### Validation set
```bash
python -u predict.py \
  --data_root data_hipmri_2d \
  --input_dir keras_slices_validate \
  --ckpt workdir/hipmri_vqvae/best_vq.pt \
  --out_dir workdir/hipmri_vqvae/predict_val_nii \
  --img_size 128 128 \
  --batch_size 64 \
  --save_format nii --metrics --dump_code_usage
```
### **What**
Run the trained model on a folder; save reconstructions (PNG or NIfTI), and optionally compute metrics.

### Outputs 














