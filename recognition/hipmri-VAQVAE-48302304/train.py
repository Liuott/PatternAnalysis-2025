# train.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse, yaml, math
from typing import Tuple

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# 本地模块
from dataset import build_loaders_from_dirs   
from modules import VQVAE, VQVAE2
from utils import batch_ssim, recon_loss_fn

MODELS = {"VQVAE": VQVAE, "VQVAE2": VQVAE2}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HipMRI 2D VQ-VAE training")
    
    ap.add_argument("--data_root", type=str, required=True,
                    help="root dir containing keras_slices_* splits")
    ap.add_argument("--train_dir", type=str, default="keras_slices_train")
    ap.add_argument("--val_dir",   type=str, default="keras_slices_validate")
    ap.add_argument("--test_dir",  type=str, default="keras_slices_test")
    ap.add_argument("--work_dir",  type=str, default="workdir/hipmri_vqvae")
    
    ap.add_argument("--model", type=str, default="VQVAE", choices=["VQVAE", "VQVAE2"])
    ap.add_argument("--img_size", type=int, nargs=2, default=[128, 128])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    ap.add_argument("--recon_loss", type=str, default="l1", choices=["l1", "mse"])
    ap.add_argument("--mixed_precision", type=lambda x: str(x).lower() in ["1","true","yes","y"], default=True)
    ap.add_argument("--seed", type=int, default=2025)

    
    ap.add_argument("--in_channels", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_res_blocks", type=int, default=2)
    ap.add_argument("--z_channels", type=int, default=64)
    ap.add_argument("--codebook_size", type=int, default=512)
    ap.add_argument("--commit_beta", type=float, default=0.25)
    ap.add_argument("--ema_decay", type=float, default=0.99)

    
    ap.add_argument("--z_top_channels", type=int, default=64)
    ap.add_argument("--z_bot_channels", type=int, default=64)
    ap.add_argument("--codebook_top", type=int, default=256)
    ap.add_argument("--codebook_bot", type=int, default=512)
    ap.add_argument("--commit_beta_top", type=float, default=0.25)
    ap.add_argument("--commit_beta_bot", type=float, default=0.25)

    
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--ssim_target", type=float, default=0.60)

    return ap.parse_args()

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    Model = MODELS[args.model]
    
    ctor_keys = {
        "in_channels","hidden","n_res_blocks","z_channels",
        "codebook_size","commit_beta","ema_decay",
        "z_top_channels","z_bot_channels","codebook_top","codebook_bot",
        "commit_beta_top","commit_beta_bot"
    }
    ctor_kwargs = {k:getattr(args,k) for k in ctor_keys if hasattr(args,k)}
    model = Model(**ctor_kwargs).to(device)
    return model

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)

    
    trL, vaL, teL = build_loaders_from_dirs(
        root=args.data_root,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)

    if args.optimizer == "adamw":
        opt = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr)

    scaler = GradScaler('cuda', enabled=args.mixed_precision)
    rec_criterion = recon_loss_fn(args.recon_loss)

    best_val = -1.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(trL, desc=f"E{epoch}/{args.epochs}", ncols=100)
        for img, _ in pbar:
            img = img.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=args.mixed_precision):
                out, vq_loss, perp, _ = model(img)
                rec = rec_criterion(out, img)
                loss = rec + vq_loss

            
            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().cpu())
            pbar.set_postfix(loss=f"{float(loss):.4f}", perp=f"{float(perp):.1f}")

        running /= max(1, len(trL))

        
        val_ssim = float('nan')
        if epoch % max(1, args.val_every) == 0:
            model.eval()
            ssim_sum, n = 0.0, 0
            with torch.no_grad():
                for img, _ in vaL:
                    img = img.to(device, non_blocking=True)
                    out, _, _, _ = model(img)
                    ssim = batch_ssim(out, img)  # expects [-1,1]
                    if torch.isfinite(ssim):
                        ssim_sum += float(ssim)
                        n += 1
            if n > 0:
                val_ssim = ssim_sum / n
            print(f"Epoch {epoch}: train={running:.4f} val_ssim={val_ssim:.4f}" if n>0
                  else f"Epoch {epoch}: train={running:.4f} val_ssim=nan")

            
            improved = (n > 0) and (val_ssim > best_val or math.isnan(best_val))
            if improved:
                best_val = val_ssim
                bad_epochs = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_ssim": val_ssim},
                    os.path.join(args.work_dir, "best_vq.pt")
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.early_stop_patience:
                    print(f"Early stop at epoch {epoch} (no improvement {bad_epochs} epochs).")
                    break

    
    
    ckpt_path = os.path.join(args.work_dir, "best_vq.pt")
    if not os.path.exists(ckpt_path):
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "val_ssim": float(best_val)},
            ckpt_path
        )

    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ssim_sum, n = 0.0, 0
    with torch.no_grad():
        for img, _ in teL:
            img = img.to(device, non_blocking=True)
            out, _, _, _ = model(img)
            ssim = batch_ssim(out, img)
            if torch.isfinite(ssim):
                ssim_sum += float(ssim)
                n += 1
    test_ssim = ssim_sum / max(1, n)

    with open(os.path.join(args.work_dir, "test_ssim.txt"), "w") as f:
        f.write(f"test_ssim: {test_ssim:.4f}\n")

    print("=== Summary ===")
    print(f"Best Val SSIM: {best_val:.4f}")
    print(f"Test  SSIM:   {test_ssim:.4f}")
    print(f"Checkpoint:   {ckpt_path}")

if __name__ == "__main__":
    main()
