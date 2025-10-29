# train.py
from __future__ import annotations
import os, argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import build_loaders_from_dirs
from utils import batch_ssim
from modules import VQVAE  

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--val_dir",   type=str, required=True)
    p.add_argument("--test_dir",  type=str, required=True)
    p.add_argument("--work_dir",  type=str, required=True)


    p.add_argument("--img_size", nargs=2, type=int, default=[128,128])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--commit_beta", type=float, default=0.25)
    p.add_argument("--codebook_size", type=int, default=512)
    p.add_argument("--ema_decay", type=float, default=0.99)


    p.add_argument("--mixed_precision", action="store_true", help="opem amp mixed precision")
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--vq_warmup", type=int, default=0, help="pre-weight steps for vq loss")
    return p.parse_args()

def set_seed(seed: int):
    import numpy as np, random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = build_loaders_from_dirs(
        args.data_root, args.train_dir, args.val_dir, args.test_dir,
        tuple(args.img_size), args.batch_size, args.seed, args.num_workers
    )

    # Model
    model = VQVAE(
        img_channels=1,
        codebook_size=args.codebook_size,
        commit_beta=args.commit_beta,
        ema_decay=args.ema_decay
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-8)
    scaler = GradScaler(enabled=args.mixed_precision)

    best_ssim = -1.0
    best_path = os.path.join(args.work_dir, "best_vq.pt")

    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        pbar = tqdm(train_loader, desc=f"E{epoch}/{args.epochs}", ncols=100)
        for xb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            
            xb = torch.nan_to_num(xb)
            xb = torch.clamp(xb, -1.0, 1.0)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=args.mixed_precision):
                recon, vq_loss, _, _ = model(xb)  # recon ∈ [-1,1]
                recon = torch.nan_to_num(recon)
                # L1 
                recon_loss = torch.mean(torch.abs(recon - xb))

                # vq warmup
                if args.vq_warmup > 0 and global_step < args.vq_warmup:
                    w = (global_step / max(1, args.vq_warmup))
                    loss = recon_loss + (w * args.commit_beta) * vq_loss
                else:
                    loss = recon_loss + args.commit_beta * vq_loss

         
            if not torch.isfinite(loss):
                print("[warn] NaN/Inf loss → skip step")
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

            if args.mixed_precision:
                scaler.scale(loss).backward()
                if args.clip_norm and args.clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.clip_norm and args.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                opt.step()

            total += float(loss.detach())
            global_step += 1
            pbar.set_postfix(loss=f"{total/ (pbar.n or 1):.4f}")

      
        val_ssim = evaluate_ssim(model, val_loader, device)
        print(f"Epoch {epoch}: train={total/max(1,len(train_loader)):.4f} val_ssim={val_ssim:.4f}")

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "best_ssim": best_ssim},
                best_path
            )

   
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[info] Loaded best ckpt from {best_path} @ epoch {ckpt.get('epoch','?')}, val_ssim={ckpt.get('best_ssim','?')}")
    test_ssim = evaluate_ssim(model, test_loader, device)
    print("=== Summary ===")
    print(f"Best Val SSIM: {best_ssim:.4f}")
    print(f"Test  SSIM:   {test_ssim:.4f}")
    print(f"Checkpoint:   {best_path}")

@torch.inference_mode()
def evaluate_ssim(model: nn.Module, loader, device) -> float:
    model.eval()
    ssim_sum, n = 0.0, 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xb = torch.nan_to_num(xb)
        xb = torch.clamp(xb, -1.0, 1.0)
        recon, _, _, _ = model(xb)
        recon = torch.nan_to_num(recon)
        ssim_sum += batch_ssim(recon, xb)
        n += 1
    return ssim_sum / max(1, n)

if __name__ == "__main__":
    main()
