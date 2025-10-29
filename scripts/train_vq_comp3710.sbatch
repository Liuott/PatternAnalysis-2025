#!/bin/bash
#SBATCH -J hipmri_vq
#SBATCH -p comp3710
#SBATCH --account=comp3710
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

source ~/.bashrc
conda activate hipmri_vq
nvidia-smi || true

python -u recognition/hipmri-VAQVAE-48302304/train.py \
  --data_root recognition/hipmri-VAQVAE-48302304/data_hipmri_2d \
  --train_dir keras_slices_train \
  --val_dir   keras_slices_validate \
  --test_dir  keras_slices_test \
  --work_dir  recognition/hipmri-VAQVAE-48302304/workdir/hipmri_vqvae \
  --img_size 128 128 --batch_size 64 --num_workers 4 --epochs 80 --lr 2e-4 \
  --codebook_size 512 --commit_beta 0.25 --ema_decay 0.99
