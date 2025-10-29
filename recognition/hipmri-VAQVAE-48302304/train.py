import os, sys, math, argparse, time, glob, random
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


from modules import VQVAE

def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_files(dir_path):
    exts = (".nii", ".nii.gz", ".npy", ".png", ".jpg", ".jpeg")
    return sorted([p for p in glob.glob(os.path.join(dir_path, "*")) if p.lower().endswith(exts)])


def load_2d_array(path):
    pl = path.lower()
    if pl.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
 
            if arr.shape[2] == 1:
                arr = arr[..., 0]
            else:
                arr = arr.mean(axis=2)
    elif pl.endswith(".png") or pl.endswith(".jpg") or pl.endswith(".jpeg"):
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
    else:
        # NIfTI
        arr = np.asanyarray(nib.load(path).get_fdata()).astype(np.float32)

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[..., 0]
            else:
                arr = arr.mean(axis=2)
        elif arr.ndim > 3:

            arr = arr.squeeze()
            if arr.ndim > 2:
                arr = arr[..., 0]
    return arr


def to_tensor_1xHxW(arr, out_size):

    arr = arr - np.nanmin(arr)
    denom = (np.nanmax(arr) + 1e-8)
    if denom == 0 or not np.isfinite(denom):
        denom = 1.0
    arr = arr / denom * 2.0 - 1.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    t = torch.from_numpy(arr)[None, None]  # [1,1,H,W]
    t = F.interpolate(t, size=out_size, mode="bilinear", align_corners=False)
    return t[0]  # [1,H,W]


# -----------------------
# Dataset
# -----------------------
class MRI2DDataset(Dataset):
    def __init__(self, root, subdir, out_size=(128, 128)):
        self.paths = list_files(os.path.join(root, subdir))
        self.out_size = out_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            arr = load_2d_array(p)
        except Exception:
 
            arr = np.zeros(self.out_size, dtype=np.float32)
        x = to_tensor_1xHxW(arr, self.out_size)  # [1,H,W] in [-1,1]

        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return x, 0


# -----------------------
# Metrics: SSIM
# -----------------------
def _gaussian_window(channels, window_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    window_1d = g
    window_2d = (window_1d.t() @ window_1d).unsqueeze(0).unsqueeze(0)  # [1,1,ks,ks]
    window_2d = window_2d.repeat(channels, 1, 1, 1)  # [C,1,ks,ks]
    return window_2d


def ssim(x, y, window=None, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    # x,y: [B,1,H,W] in [-1,1], dtype float
 
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    device = x.device
    dtype = x.dtype
    if window is None:
        window = _gaussian_window(1, window_size, sigma=1.5, device=device, dtype=dtype)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=1)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=1) - mu_xy

    # 数值兜底
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


# -----------------------
# Train/Eval
# -----------------------
def get_vq_weight(step, warmup):
    if warmup <= 0:
        return 1.0
    return min(1.0, float(step) / float(warmup))


def evaluate(model, loader, device):
    model.eval()
    window = None
    tot = 0.0
    n = 0
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device, non_blocking=True)
     
            img = img.clamp(-1, 1)
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)

            recon, _, _ = model(img)
            # recon
            recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=-1.0)
            s = ssim(img, recon, window=window)
            if window is None:

                window = _gaussian_window(1, device=img.device, dtype=img.dtype)
            tot += float(s)
            n += 1
    return tot / max(1, n)


def save_ckpt(state, path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default="keras_slices_train")
    parser.add_argument("--val_dir",   type=str, default="keras_slices_validate")
    parser.add_argument("--test_dir",  type=str, default="keras_slices_test")
    parser.add_argument("--work_dir",  type=str, default="workdir/hipmri_vqvae")
    parser.add_argument("--img_size",  type=int, nargs=2, default=[128, 128])
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--recon_loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--vq_warmup", type=int, default=1000)

    # model hparams
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
    out_h, out_w = args.img_size

    # Datasets / Loaders
    train_set = MRI2DDataset(args.data_root, args.train_dir, (out_h, out_w))
    val_set   = MRI2DDataset(args.data_root, args.val_dir,   (out_h, out_w))
    test_set  = MRI2DDataset(args.data_root, args.test_dir,  (out_h, out_w))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=max(1, args.num_workers//2), pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=max(1, args.num_workers//2), pin_memory=True)

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

    # Optimizer
    if args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    # Resume
    best_ckpt = os.path.join(args.work_dir, "best_vq.pt")
    best_val = -1.0
    start_epoch = 1
    if args.resume and os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        try:
            optim.load_state_dict(ckpt["optim"])
        except Exception:
            pass
        start_epoch = ckpt.get("epoch", 1)
        best_val = ckpt.get("best_val", -1.0)
        print(f"[info] Loaded best ckpt from {best_ckpt} @ epoch {start_epoch}, val_ssim={best_val}")

    # Loss
    def recon_crit(pred, tgt):
        if args.recon_loss == "l1":
            return F.l1_loss(pred, tgt)
        else:
            return F.mse_loss(pred, tgt)

    global_step = 0
    os.makedirs(args.work_dir, exist_ok=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            pbar = tqdm(train_loader, ncols=100, desc=f"E{epoch}/{args.epochs}")
            epoch_loss = 0.0
            n_step = 0

            for img, _ in pbar:
                img = img.to(device, non_blocking=True)


                img = img.clamp(-1, 1)
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)

                optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    recon, vq_loss, vq_stats = model(img)


                    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=-1.0)

                    rloss = recon_crit(recon, img)
                    vq_w = get_vq_weight(global_step, args.vq_warmup)
                    total = rloss + vq_w * vq_loss


                total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=-1e4)

                scaler.scale(total).backward()
                if args.clip_norm and args.clip_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optim)
                scaler.update()

                global_step += 1
                n_step += 1
                epoch_loss += float(rloss.detach().cpu())

                perp = float(torch.as_tensor(vq_stats.get("perplexity", 0.0)).detach().cpu())
                pbar.set_postfix(loss=f"{epoch_loss/n_step:.4f}", perp=f"{perp:.1f}", vq_w=f"{vq_w:.2f}")

            val_ssim = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: train={epoch_loss/max(1,n_step):.4f} val_ssim={val_ssim:.4f}")


            if val_ssim > best_val:
                best_val = val_ssim
                save_ckpt(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "best_val": best_val,
                        "args": vars(args),
                    },
                    best_ckpt,
                )

        if os.path.isfile(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            print(f"[info] Loaded best ckpt from {best_ckpt} @ epoch {ckpt.get('epoch','?')}, val_ssim={ckpt.get('best_val','?'):.6f}")

        test_ssim = evaluate(model, test_loader, device)
        print("=== Summary ===")
        print(f"Best Val SSIM: {best_val:.4f}")
        print(f"Test  SSIM:   {test_ssim:.4f}")
        print(f"Checkpoint:   {best_ckpt}")

    except KeyboardInterrupt:
        print("\n[warn] Interrupted by user. Attempting to save last checkpoint...")
        last_ckpt = os.path.join(args.work_dir, "last_vq.pt")
        save_ckpt(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            },
            last_ckpt,
        )
        print(f"[info] Last checkpoint saved to {last_ckpt}")


if __name__ == "__main__":
    main()