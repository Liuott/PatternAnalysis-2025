# train.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, argparse

# 让本地模块可被 import（modules.py / utils.py / dataset.py 与本文件同目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import build_loaders_from_dirs
from modules import VQVAE
from utils import batch_ssim, recon_loss_fn

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HipMRI 2D VQ-VAE training (images only)")


    ap.add_argument("--data_root", type=str, required=True, help="root with keras_slices_* dirs")
    ap.add_argument("--train_dir", type=str, default="keras_slices_train")
    ap.add_argument("--val_dir",   type=str, default="keras_slices_validate")
    ap.add_argument("--test_dir",  type=str, default="keras_slices_test")


    ap.add_argument("--work_dir", type=str, default="workdir/hipmri_vqvae")
    ap.add_argument("--img_size", type=int, nargs=2, default=[128, 128])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    ap.add_argument("--recon_loss", type=str, default="l1", choices=["l1","mse"])
    ap.add_argument("--seed", type=int, default=2025)


    ap.add_argument("--mixed_precision", action="store_true", help="enable AMP (default off)")
    ap.add_argument("--clip_norm", type=float, default=1.0, help="grad clip max-norm (0=off)")
    ap.add_argument("--vq_warmup", type=int, default=500, help="steps without vq_loss at start")


    ap.add_argument("--in_channels", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--z_channels", type=int, default=64)
    ap.add_argument("--n_res_blocks", type=int, default=2)
    ap.add_argument("--codebook_size", type=int, default=512)
    ap.add_argument("--commit_beta", type=float, default=0.25)
    ap.add_argument("--ema_decay", type=float, default=0.99)

    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)

    # Data
    trL, vaL, teL = build_loaders_from_dirs(
        args.data_root, args.train_dir, args.val_dir, args.test_dir,
        tuple(args.img_size), args.batch_size, args.seed, args.num_workers
    )

    # Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(
        in_channels=args.in_channels,
        hidden=args.hidden,
        z_channels=args.z_channels,
        n_res_blocks=args.n_res_blocks,
        codebook_size=args.codebook_size,
        commit_beta=args.commit_beta,
        ema_decay=args.ema_decay,
    ).to(device)

    # Optimizer & loss
    opt = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer=="adam" else \
          optim.AdamW(model.parameters(), lr=args.lr)
    rec_criterion = recon_loss_fn(args.recon_loss)

    # AMP scaler
    scaler = GradScaler(enabled=args.mixed_precision)

    # Train
    best_val = -1.0
    global_step = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        tl, n_tr = 0.0, 0
        pbar = tqdm(trL, desc=f"E{epoch}/{args.epochs}")

        for img, _ in pbar:
            img = img.to(device)
            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=args.mixed_precision):
                out, vq_loss, perp, _ = model(img)
                rec = rec_criterion(out, img)
                loss = rec if global_step < args.vq_warmup else (rec + vq_loss)

      
            if torch.isnan(loss) or torch.isinf(loss):
                print("[warn] NaN/Inf loss → skip step")
                global_step += 1
                continue

            scaler.scale(loss).backward()

        
            if args.clip_norm and args.clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            scaler.step(opt)
            scaler.update()

            tl += float(loss.detach())
            n_tr += 1
           
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", perp=f"{float(perp):.1f}")
            global_step += 1

        tl = tl / max(1, n_tr)

  
        model.eval()
        ssim_sum, n_va = 0.0, 0
        with torch.no_grad():
            for img, _ in vaL:
                img = img.to(device)
                out, _, _, _ = model(img)
                ssim_sum += float(batch_ssim(out, img))
                n_va += 1
        val_ssim = ssim_sum / max(1, n_va)
        print(f"Epoch {epoch}: train={tl:.4f} val_ssim={val_ssim:.4f}")

        # Save best by SSIM
        if val_ssim > best_val and torch.isfinite(torch.tensor(val_ssim)):
            best_val = val_ssim
            ckpt_path = os.path.join(args.work_dir, "best_vq.pt")
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_ssim": val_ssim},
                ckpt_path
            )

 
    ckpt_path = os.path.join(args.work_dir, "best_vq.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[info] Loaded best ckpt from {ckpt_path} @ epoch {ckpt.get('epoch','?')}, val_ssim={ckpt.get('val_ssim','?')}")
    else:
        print("[warn] best_vq.pt not found, evaluating with last epoch weights.")

    model.eval()
    ssim_sum, n_te = 0.0, 0
    with torch.no_grad():
        for img, _ in teL:
            img = img.to(device)
            out, _, _, _ = model(img)
            ssim_sum += float(batch_ssim(out, img))
            n_te += 1
    test_ssim = ssim_sum / max(1, n_te)

    print("=== Summary ===")
    print(f"Best Val SSIM: {best_val if best_val>=0 else float('nan'):.4f}")
    print(f"Test  SSIM:   {test_ssim:.4f}")
    print(f"Checkpoint:   {ckpt_path if os.path.exists(ckpt_path) else '(last epoch)'}")

if __name__ == "__main__":
    main()
