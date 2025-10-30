# -*- coding: utf-8 -*-
import os, argparse, math, csv
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib  
    HAVE_NIB = True
except Exception:
    HAVE_NIB = False

from modules import VQVAE  

def to_tensor_2d(arr):

    x = arr.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8) * 2.0 - 1.0
    t = torch.from_numpy(x)[None, None]  # [1,1,H,W]
    return t

def to_numpy_01(t):

    x = t.detach().cpu().float()
    x = (x + 1.0) * 0.5
    return x.clamp(0, 1).numpy()

def psnr(x, y, eps=1e-8):
    # x,y in [0,1], NCHW
    mse = torch.mean((x - y) ** 2, dim=[1,2,3]) + eps
    return 10.0 * torch.log10(1.0 / mse)

def ssim_2d(x, y, C1=0.01**2, C2=0.03**2):

    kernel = torch.ones(1,1,3,3, device=x.device) / 9.0
    mu_x = F.conv2d(x, kernel, padding=1)
    mu_y = F.conv2d(y, kernel, padding=1)
    mu_x2, mu_y2 = mu_x**2, mu_y**2
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x*x, kernel, padding=1) - mu_x2
    sigma_y2 = F.conv2d(y*y, kernel, padding=1) - mu_y2
    sigma_xy = F.conv2d(x*y, kernel, padding=1) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-8)
    return ssim_map.mean(dim=[1,2,3])


class Nii2DDataset(Dataset):
    def __init__(self, root, subdir, out_size):
        self.root = Path(root)
        self.dir = self.root / subdir
        self.paths = sorted([p for p in self.dir.glob("*.nii*")])
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No NIfTI files under {self.dir}")
        self.out_h, self.out_w = out_size

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = np.asanyarray(nib.load(str(p)).get_fdata()).astype(np.float32)
        if img.ndim == 3:  # [H,W,C] -> 2D slice
            img = img[..., 0]
        t = to_tensor_2d(img)                      # [1,1,H,W], [-1,1]
        t = F.interpolate(t, size=(self.out_h,self.out_w), mode="bilinear", align_corners=False)[0]
        return t, str(p.name)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--input_dir", type=str, required=True, help="e.g., keras_slices_test / keras_slices_validate")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--img_size", type=int, nargs=2, default=[128,128])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_format", type=str, default="png", choices=["png","nii","none"])
    ap.add_argument("--metrics", action="store_true", help="compute SSIM & PSNR")
    ap.add_argument("--dump_code_usage", action="store_true", help="frequency of codebook usage")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)


    ds = Nii2DDataset(args.data_root, args.input_dir, tuple(args.img_size))
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)


    model = VQVAE(
        in_channels=1, hidden=128, z_channels=64, n_res_blocks=2,
        codebook_size=256, commit_beta=0.15, ema_decay=0.99
    ).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()


    all_ssim, all_psnr = [], []
    code_hist = None

    pbar = tqdm(dl, desc="Predict")
    with torch.no_grad():
        for x, names in pbar:
            x = x.to(device)                      # [-1,1], NCHW

            recon, vq_loss, vq_stats = model(x)

            x01    = (x + 1.0) * 0.5
            recon01= (recon + 1.0) * 0.5
            recon01 = recon01.clamp(0,1)

            if args.metrics:
                s = ssim_2d(x01, recon01)
                p = psnr(x01, recon01)
                all_ssim.extend(s.detach().cpu().tolist())
                all_psnr.extend(p.detach().cpu().tolist())
                pbar.set_postfix(ssim=np.mean(all_ssim[-len(s):]), psnr=np.mean(all_psnr[-len(p):]))

            if args.dump_code_usage:
                z_e = model.encoder(x)                               # [N,D,h,w]
                B, D, H, W = z_e.shape
                flat = z_e.permute(0,2,3,1).contiguous().view(-1, D) # [N*H*W, D]
                emb  = model.quantizer.embedding                     # [D,K]
 
                bs = 65536
                idx_list = []
                for i0 in range(0, flat.size(0), bs):
                    f = flat[i0:i0+bs]
                    dist = (f.pow(2).sum(1, keepdim=True)
                            + emb.pow(2).sum(0, keepdim=True)
                            - 2 * (f @ emb))
                    idx_list.append(torch.argmin(dist, dim=1))
                idx = torch.cat(idx_list, dim=0).detach().cpu().numpy()  # [N*H*W]
                K = emb.shape[1]
                hist = np.bincount(idx, minlength=K)
                code_hist = hist if code_hist is None else (code_hist + hist)

    
            if args.save_format != "none":
                for i in range(recon01.size(0)):
                    name = os.path.splitext(names[i])[0]
                    arr  = (recon01[i,0].detach().cpu().numpy())  # [H,W], 0~1
                    if args.save_format == "png":
                     
                        try:
                            import imageio.v2 as imageio
                        except Exception:
                            import imageio
                        imageio.imwrite(os.path.join(args.out_dir, f"{name}_recon.png"), (arr*255).astype(np.uint8))
                    elif args.save_format == "nii":
                        if not HAVE_NIB:
                            raise RuntimeError("nibabel not installed; pip install nibabel")
                        nii = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4, dtype=np.float32))
                        nib.save(nii, os.path.join(args.out_dir, f"{name}_recon.nii.gz"))


    if args.metrics:
        ssim_mean = float(np.mean(all_ssim)) if all_ssim else float("nan")
        psnr_mean = float(np.mean(all_psnr)) if all_psnr else float("nan")
        print(f"=== Metrics on {args.input_dir} ===")
        print(f"SSIM: {ssim_mean:.4f} | PSNR: {psnr_mean:.2f} dB")
        with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split","ssim_mean","psnr_mean","count"])
            w.writerow([args.input_dir, ssim_mean, psnr_mean, len(all_ssim)])

    if args.dump_code_usage and (code_hist is not None):
        np.save(os.path.join(args.out_dir, "code_usage.npy"), code_hist)

        with open(os.path.join(args.out_dir, "code_usage.txt"), "w") as f:
            tot = int(code_hist.sum())
            for k, c in enumerate(code_hist.tolist()):
                f.write(f"{k}\t{c}\t{c/tot:.6f}\n")
        print(f"[info] code usage saved: top={int(code_hist.max())}, zero_bins={(code_hist==0).sum()} / {len(code_hist)}")

if __name__ == "__main__":
    main()
