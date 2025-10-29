# train.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, math, argparse


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import build_loaders_from_dirs  
from modules import VQVAE                     #
from utils import batch_ssim, recon_loss_fn

MODELS = {"VQVAE": VQVAE}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HipMRI 2D VQ-VAE training (images only)")
 
    ap.add_argument("--data_root", type=str, required=True,
                    help="root dir containing keras_slices_train/validate/test")
    ap.add_argument("--train_dir", type=str, default="keras_slices_train")
    ap.add_argument("--val_dir",   type=str, default="keras_slices_validate")
    ap.add_argument("--test_dir",  type=str, default="keras_slices_test")
    ap.add_argument("--work_dir",  type=str, default="workdir/hipmri_vqvae")


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


    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--ssim_target", type=float, default=0.60)

    return ap.parse_args()

def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    ctor_kwargs = dict(
        in_channels=args.in_channels,
        hidden=args.hidden,
        n_res_blocks=args.n_res_blocks,
        z_channels=args.z_channels,
        codebook_size=args.codebook_size,
        commit_beta=args.commit_beta,
        ema_decay=args.ema_decay,
    )
    model = MODELS["VQVAE"](**ctor_kwargs).to(device)
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

    opt = optim.AdamW(model.parameters(), lr=args.lr) if args.optimizer == "adamw" \
          else optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda', enabled=args.mixed_precision)
    rec_criterion = recon_loss_fn(args.recon_loss)

    best_val = float('nan')
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(trL, desc=f"E{epoch}/{args.epochs}", ncols=100)

        for img, _ in pbar:
            img = img.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=args.mixed_precision):
                out, vq_loss, perp, _ = model(img)     
                rec = rec_criterion(out, img)
                loss = rec + vq_loss


            if not torch.isfinite(loss):
                pbar.set_postfix(loss="nan-skip", perp=f"{float(perp):.1f}")
                continue

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            pbar.set_postfix(loss=f"{float(loss):.4f}", perp=f"{float(perp):.1f}")

        epoch_loss /= max(1, len(trL))

 
        val_ssim = float('nan')
        if epoch % max(1, args.val_every) == 0:
            model.eval()
            ssim_sum, n = 0.0, 0
            with torch.no_grad():
                for img, _ in vaL:
                    img = img.to(device, non_blocking=True)
                    out, _, _, _ = model(img)
                    ssim = batch_ssim(out, img)  
                    if torch.isfinite(ssim):
                        ssim_sum += float(ssim)
                        n += 1
            if n > 0:
                val_ssim = ssim_sum / n
                print(f"Epoch {epoch}: train={epoch_loss:.4f} val_ssim={val_ssim:.4f}")
            else:
                print(f"Epoch {epoch}: train={epoch_loss:.4f} val_ssim=nan")

            improved = (n > 0) and (math.isnan(best_val) or val_ssim > best_val)
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
