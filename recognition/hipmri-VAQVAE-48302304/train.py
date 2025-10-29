# -*- coding: utf-8 -*-
import os, math, argparse, time, glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit("Please `pip install nibabel`") from e


from modules import VQVAE


from contextlib import nullcontext

def make_autocast(mixed: bool):
    """
    返回一个上下文管理器：
      - torch>=2.0: torch.amp.autocast('cuda', dtype=torch.bfloat16)
      - torch<2.0 : torch.cuda.amp.autocast(enabled=mixed)
      - CPU/未启用: nullcontext()
    """
    if not mixed or not torch.cuda.is_available():
        return nullcontext()
    # PyTorch 2.x
    try:
        import torch.amp
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    except Exception:
        pass
    
    try:
        from torch.cuda.amp import autocast as autocast_old
        return autocast_old(enabled=True)
    except Exception:
        return nullcontext()


# Utils

def set_seed(seed: int):
    if seed is None: return
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def count_params(m):
    return sum(p.numel() for p in m.parameters())/1e6

def is_finite(x: torch.Tensor):
    return torch.isfinite(x).all().item()



# Simple SSIM (expects input in [0,1])

def _gaussian_kernel(channels, kernel_size=11, sigma=1.5, device='cpu', dtype=torch.float32):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-(coords**2)/(2*sigma*sigma))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, -1)
    kernel_2d = (kernel_1d.transpose(2, 1) @ kernel_1d).view(1, 1, kernel_size, kernel_size)
    return kernel_2d.repeat(channels, 1, 1, 1)

def ssim_torch(x, y, data_range=1.0, K1=0.01, K2=0.03, kernel_size=11, sigma=1.5):
    # x,y: [B,1,H,W] in [0,1]
    device = x.device
    dtype  = x.dtype
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    channel = x.size(1)
    kernel = _gaussian_kernel(channel, kernel_size, sigma, device=device, dtype=dtype)
    mu_x = F.conv2d(x, kernel, padding=kernel_size//2, groups=channel)
    mu_y = F.conv2d(y, kernel, padding=kernel_size//2, groups=channel)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sigma_x  = F.conv2d(x*x, kernel, padding=kernel_size//2, groups=channel) - mu_x2
    sigma_y  = F.conv2d(y*y, kernel, padding=kernel_size//2, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x*y, kernel, padding=kernel_size//2, groups=channel) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x + sigma_y + C2))
    return ssim_map.mean()


# -----------------------------
# Dataset
# -----------------------------
class Nifti2DSliceDataset(Dataset):
    """
    读取目录下 *.nii 或 *.nii.gz，取第 0 通道为单通道图，缩放到 img_size，并标准化到 [-1,1]
    """
    def __init__(self, root_dir, img_size=(128,128)):
        self.files = sorted([p for ext in ("*.nii", "*.nii.gz") for p in glob.glob(os.path.join(root_dir, ext))])
        if not self.files:
            raise FileNotFoundError(f"No NIfTI found in {root_dir}")
        self.size = tuple(img_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.asanyarray(nib.load(path).get_fdata()).astype(np.float32)
        if arr.ndim == 3:  # H W C -> take first channel
            arr = arr[..., 0]
        # normalize to [-1,1]
        minv, maxv = float(arr.min()), float(arr.max())
        if math.isclose(maxv - minv, 0.0, rel_tol=0, abs_tol=1e-12):
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = (arr - minv) / (maxv - minv) * 2.0 - 1.0

        t = torch.from_numpy(arr)[None, None]  # [1,1,H,W]
        t = F.interpolate(t, size=self.size, mode="bilinear", align_corners=False)[0]  # [1,H,W]
        return t  



# Train / Val loops

def do_epoch(model, loader, optimizer, device, args, step0, scaler=None):
    model.train()
    pbar = tqdm(loader, desc=f"E{args.cur_epoch}/{args.epochs}", dynamic_ncols=True)
    loss_meter = 0.0
    count = 0
    global_step = step0

    recon_crit = F.l1_loss if args.recon_loss == "l1" else F.mse_loss

    for img in pbar:
        img = img.to(device, non_blocking=True)

        # warmup vq weight
        vq_w = 0.0
        if args.vq_warmup > 0:
            vq_w = min(1.0, global_step / float(args.vq_warmup))
        else:
            vq_w = 1.0

        with make_autocast(args.mixed_precision):

            recon, vq_loss, vq_stats = model(img)  
            recon_loss = F.l1_loss(recon, img, reduction='mean') 
            loss = recon_loss + vq_w * vq_loss


        if not torch.isfinite(loss):
            pbar.set_postfix_str("[warn] NaN/Inf loss → skip step")
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        if args.clip_norm is not None and args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

        optimizer.step()

        loss_meter += loss.detach().float().item()
        count += 1
        global_step += 1


        perp = float(vq_stats.get('perplexity', torch.tensor(0.0)).detach().float().item())
        pbar.set_postfix(loss=f"{loss.detach().float().item():.4f}", perp=f"{perp:.1f}", vq_w=f"{vq_w:.2f}")

    return (loss_meter / max(count, 1)), global_step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ssim_sum = 0.0
    n = 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        # forward
        recon, _, _ = model(x)

        x_01     = torch.clamp((x + 1) * 0.5, 0.0, 1.0)
        recon_01 = torch.clamp((recon + 1) * 0.5, 0.0, 1.0)
        ssim_val = ssim_torch(recon_01, x_01, data_range=1.0)
        if torch.isnan(ssim_val):
            continue
        ssim_sum += ssim_val.item()
        n += 1
    return ssim_sum / max(n, 1)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--train_dir", default="keras_slices_train", type=str)
    parser.add_argument("--val_dir",   default="keras_slices_validate", type=str)
    parser.add_argument("--test_dir",  default="keras_slices_test", type=str)
    parser.add_argument("--work_dir",  default="workdir/hipmri_vqvae", type=str)

    parser.add_argument("--img_size", nargs=2, type=int, default=[128,128])
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam","adamw"], default="adamw")
    parser.add_argument("--recon_loss", choices=["l1","mse"], default="l1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--vq_warmup", type=int, default=1000)


    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--z_channels", type=int, default=64)
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--commit_beta", type=float, default=0.15)
    parser.add_argument("--ema_decay", type=float, default=0.99)

    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.work_dir)

    # Data
    tr_ds = Nifti2DSliceDataset(os.path.join(args.data_root, args.train_dir), img_size=args.img_size)
    va_ds = Nifti2DSliceDataset(os.path.join(args.data_root, args.val_dir),   img_size=args.img_size)
    te_ds = Nifti2DSliceDataset(os.path.join(args.data_root, args.test_dir),  img_size=args.img_size)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    # Model
    model = VQVAE(
        in_channels=args.in_channels,
        hidden=args.hidden,
        z_channels=args.z_channels,
        n_res_blocks=args.n_res_blocks,
        codebook_size=args.codebook_size,
        commit_beta=args.commit_beta,
        ema_decay=args.ema_decay,
    ).to(device)

    n_params = count_params(model)
    print(f"[info] Model params: {n_params:.2f} M")
    print(f"[info] Mixed Precision (BF16): {args.mixed_precision}")

    # Optim
    if args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # (Optional) resume
    best_ckpt = os.path.join(args.work_dir, "best_vq.pt")
    last_ckpt = os.path.join(args.work_dir, "last_vq.pt")
    best_val = -1.0
    global_step = 0

    if args.resume and os.path.isfile(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        global_step = ckpt.get("global_step", 0)
        best_val = ckpt.get("best_val", -1.0)
        print(f"[info] Resume from {last_ckpt} @ step {global_step}, best_val={best_val:.4f}")

    # Train
    for epoch in range(1, args.epochs+1):
        args.cur_epoch = epoch
        train_loss, global_step = do_epoch(model, tr_loader, optim, device, args, global_step)

        # Val
        val_ssim = evaluate(model, va_loader, device)
        print(f"Epoch {epoch}: train={train_loss:.4f} val_ssim={val_ssim if not math.isnan(val_ssim) else float('nan'):.4f}")

        # Save last
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "global_step": global_step,
            "best_val": best_val,
            "epoch": epoch,
        }, last_ckpt)

        # Save best
        if not math.isnan(val_ssim) and val_ssim > best_val:
            best_val = val_ssim
            torch.save(model.state_dict(), best_ckpt)
            print(f"[info] Saved BEST to {best_ckpt} (val_ssim={best_val:.4f})")

    # Test on best
    if os.path.isfile(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        test_ssim = evaluate(model, te_loader, device)
        print("=== Summary ===")
        print(f"Best Val SSIM: {best_val:.4f}")
        print(f"Test  SSIM:   {test_ssim:.4f}")
        print(f"Checkpoint:   {best_ckpt}")
    else:
        print("[warn] No best checkpoint found; skipping test.")


if __name__ == "__main__":
    main()
