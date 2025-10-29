import os, json, math, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import VQVAE            
from dataset import build_loaders_from_dirs


def _gaussian_kernel(channels, kernel_size=11, sigma=1.5, device="cpu"):
    # 1D gaussian
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    # 2D separable
    g2d = torch.outer(g, g)
    kernel = g2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel

@torch.no_grad()
def ssim_torch(x, y, data_range=2.0, kernel_size=11, sigma=1.5):
    """
    x,y: [N,1,H,W] in [-1,1] or [0,1]; data_range 根据输入调整。
    返回 batch SSIM 的 mean。
    """
    device = x.device
    C = x.size(1)
    w = _gaussian_kernel(C, kernel_size, sigma, device=device)
    padding = kernel_size // 2

    mu_x = F.conv2d(x, w, groups=C, padding=padding)
    mu_y = F.conv2d(y, w, groups=C, padding=padding)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, groups=C, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, groups=C, padding=padding) - mu_y2
    sigma_xy = F.conv2d(x * y, w, groups=C, padding=padding) - mu_xy

    # constants per Image Quality Assessment literature
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean().item()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_dir", type=str, default="keras_slices_train")
    p.add_argument("--val_dir",   type=str, default="keras_slices_validate")
    p.add_argument("--test_dir",  type=str, default="keras_slices_test")
    p.add_argument("--work_dir",  type=str, default="workdir/hipmri_vqvae")

    p.add_argument("--img_size",  type=int, nargs=2, default=[128, 128])
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--optimizer",   type=str, choices=["adam", "adamw"], default="adamw")
    p.add_argument("--recon_loss",  type=str, choices=["l1", "mse"], default="l1")
    p.add_argument("--seed",        type=int, default=2025)

    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--clip_norm",   type=float, default=0.0)
    p.add_argument("--vq_warmup",   type=int, default=0)  # in steps; 0=off

    # VQVAE hyperparams (match modules.VQVAE signature)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--hidden",      type=int, default=128)
    p.add_argument("--z_channels",  type=int, default=64)
    p.add_argument("--n_res_blocks",type=int, default=2)
    p.add_argument("--codebook_size", type=int, default=512)
    p.add_argument("--commit_beta",   type=float, default=0.25)
    p.add_argument("--ema_decay",     type=float, default=0.99)

    return p

# ---------------------------
# helpers
# ---------------------------
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def is_finite_tensor(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

# ---------------------------
# training
# ---------------------------
def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ensure_dir(args.work_dir)
    log_file = os.path.join(args.work_dir, "train_log.txt")

    # data
    train_loader, val_loader, test_loader = build_loaders_from_dirs(
        args.data_root,
        args.train_dir,
        args.val_dir,
        args.test_dir,
        tuple(args.img_size),
        args.batch_size,
        args.seed,
        args.num_workers
    )

    # model
    model = VQVAE(
        in_channels=args.in_channels,
        hidden=args.hidden,
        z_channels=args.z_channels,
        n_res_blocks=args.n_res_blocks,
        codebook_size=args.codebook_size,
        commit_beta=args.commit_beta,
        ema_decay=args.ema_decay
    ).to(device)

    # loss
    if args.recon_loss == "l1":
        recon_criterion = nn.L1Loss()
        data_range = 2.0  # inputs normalized to [-1,1]
    else:
        recon_criterion = nn.MSELoss()
        data_range = 2.0

    # opt
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)

    # bookkeeping
    global_step = 0
    best_ssim = -1.0
    best_epoch = -1
    best_ckpt = os.path.join(args.work_dir, "best_vq.pt")
    last_ckpt = os.path.join(args.work_dir, "last_vq.pt")

    # ------------- train epochs -------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"E{epoch}/{args.epochs}", ncols=100)

        for img, _ in pbar:
            img = img.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                
                recon, vq_loss, vq_stats = model(img)
                recon_loss = recon_criterion(recon, img)
                # vq warmup
                if args.vq_warmup and args.vq_warmup > 0:
                    vq_w = min(1.0, global_step / float(args.vq_warmup))
                else:
                    vq_w = 1.0
                loss = recon_loss + vq_w * vq_loss

            if not is_finite_tensor(loss):
                print("[warn] NaN/Inf loss → skip step")
                global_step += 1
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.clip_norm and args.clip_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.detach().item()
            global_step += 1

            perp = None
            if isinstance(vq_stats, dict):
                p = vq_stats.get("perplexity", None)
                if isinstance(p, torch.Tensor):
                    try:
                        perp = float(p.detach().item())
                    except Exception:
                        perp = None
                elif isinstance(p, (int, float)):
                    perp = float(p)

           
            post = {"loss": f"{loss.detach().item():.4f}"}
            if perp is not None:
                post["perp"] = f"{perp:.1f}"
            if args.vq_warmup and args.vq_warmup > 0:
                post["vq_w"] = f"{vq_w:.2f}"
            pbar.set_postfix(**post)

        train_loss = epoch_loss / max(1, len(train_loader))
        # ------------- validation -------------
        model.eval()
        val_ssim_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for img, _ in val_loader:
                img = img.to(device, non_blocking=True)
                recon, _, _ = model(img)
                
                ssim_val = ssim_torch(torch.clamp(recon, -1, 1), torch.clamp(img, -1, 1), data_range=data_range)
                val_ssim_sum += ssim_val
                val_cnt += 1
        val_ssim = val_ssim_sum / max(1, val_cnt)

        print(f"Epoch {epoch}: train={train_loss:.4f} val_ssim={val_ssim:.4f}")

        # save last
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}, last_ckpt)

        # save best
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_ckpt)

        # append log
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_ssim": val_ssim,
                "best_ssim": best_ssim,
                "best_epoch": best_epoch
            }) + "\n")

    # ------------- load best, test -------------
    info = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(info["model"])
    print(f"[info] Loaded best ckpt from {best_ckpt} @ epoch {info.get('epoch','?')}, val_ssim={best_ssim}")

    model.eval()
    test_ssim_sum, test_cnt = 0.0, 0
    with torch.no_grad():
        for img, _ in test_loader:
            img = img.to(device, non_blocking=True)
            recon, _, _ = model(img)
            ssim_val = ssim_torch(torch.clamp(recon, -1, 1), torch.clamp(img, -1, 1), data_range=data_range)
            test_ssim_sum += ssim_val
            test_cnt += 1
    test_ssim = test_ssim_sum / max(1, test_cnt)

    print("=== Summary ===")
    print(f"Best Val SSIM: {best_ssim:.4f}")
    print(f"Test  SSIM:   {test_ssim:.4f}")
    print(f"Checkpoint:   {best_ckpt}")


if __name__ == "__main__":
    main()