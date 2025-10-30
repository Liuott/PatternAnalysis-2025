# dataset.py
from __future__ import annotations
import os, glob, random
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nibabel as nib

def _safe_norm01_then_m11(img: np.ndarray) -> np.ndarray:

    img = np.asarray(img, dtype=np.float32)
   
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

   
    m = float(np.mean(img))
    s = float(np.std(img))
    if not np.isfinite(m):
        m = 0.0
    if (not np.isfinite(s)) or s < 1e-6:
        s = 1.0
    img = (img - m) / s

   
    img = np.clip(img, -5.0, 5.0)
    
    img = img / 5.0
    img = np.clip(img, -1.0, 1.0)
    return img

class NiftiSliceDataset(Dataset):
    def __init__(self, files: List[str], img_size: Tuple[int,int]=(128,128)):
        self.files = files
        self.H, self.W = int(img_size[0]), int(img_size[1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        f = self.files[idx]
        data = nib.load(f).get_fdata()  
        if data.ndim == 3:
           
            k = data.shape[-1] // 2
            data = data[..., k]
        img = _safe_norm01_then_m11(data)  

        
        t = torch.from_numpy(img)[None, None, ...]  # [1,1,H,W]
        t = torch.nan_to_num(t)
        t = F.interpolate(t, size=(self.H, self.W), mode="bilinear", align_corners=False)
        t = t[0]  # [1,H,W]
        return t, 0 

def _collect_files(root: str, subdir: str) -> List[str]:
    pat1 = os.path.join(root, subdir, "*.nii")
    pat2 = os.path.join(root, subdir, "*.nii.gz")
    files = sorted(glob.glob(pat1) + glob.glob(pat2))
    return files

def build_loaders_from_dirs(
    data_root: str,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    img_size: Tuple[int,int],
    batch_size: int,
    seed: int = 2025,
    num_workers: int = 4,
):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_files = _collect_files(data_root, train_dir)
    va_files = _collect_files(data_root, val_dir)
    te_files = _collect_files(data_root, test_dir)

    tr_ds = NiftiSliceDataset(tr_files, img_size)
    va_ds = NiftiSliceDataset(va_files, img_size)
    te_ds = NiftiSliceDataset(te_files, img_size)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, drop_last=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, drop_last=False)
    return tr_loader, va_loader, te_loader
