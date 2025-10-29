from __future__ import annotations
import os, glob, numpy as np, nibabel as nib, cv2, torch
from torch.utils.data import Dataset, DataLoader


class HipMRI2DImageOnly(Dataset):
    """Image-only loader for 2D NIfTI slices.
    - If files list is provided, it is used directly (absolute paths to .nii/.nii.gz).
    - Otherwise, (root/img_dir) is scanned for *.nii*.
    """
    def __init__(self, root, img_dir='imagesTr', img_size=(128,128), files=None, aug=False, norm=True):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.files = sorted(glob.glob(os.path.join(self.img_dir, '*.nii*'))) if files is None else files
        self.img_size = tuple(img_size)
        self.aug = aug; self.norm = norm
        
    def __len__(self): 
        return len(self.files)

    def _read_nii2d(self, p):
        nii = nib.load(p); arr = nii.get_fdata(caching='unchanged')
        if arr.ndim==3: arr = arr[:,:,0]
        return arr.astype(np.float32)

    def __getitem__(self, i):
        ip = self.files[i]
        img = self._read_nii2d(ip)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        if self.aug and np.random.rand() < 0.5: img = np.flip(img, 1).copy()
        if self.norm:
            m, s = img.mean(), img.std()+1e-5
            img = (img - m) / s
        # map to [-1,1]
        img = (img - img.min()) / (img.max()-img.min()+1e-8)
        img = img * 2 - 1
        img = torch.from_numpy(img[None,...]).float()
        return img, os.path.basename(ip)


# --- Official split support: use three folders directly ---


def build_loaders_from_dirs(root: str,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    img_size: tuple[int,int],
    batch_size: int,
    seed: int,
    num_workers: int):
    """Build DataLoaders using explicit train/val/test directories.
    Expect each directory to contain *.nii or *.nii.gz files.
    """
    def list_files(d):
        return sorted(glob.glob(os.path.join(root, d, '*.nii*')))


    tr_files = list_files(train_dir)
    va_files = list_files(val_dir)
    te_files = list_files(test_dir)


    get = lambda files, aug: HipMRI2DImageOnly(root, '', img_size, files=files, aug=aug)
    ds_tr, ds_va, ds_te = get(tr_files, True), get(va_files, False), get(te_files, False)


    L = lambda ds, s: DataLoader(ds, batch_size=batch_size, shuffle=(s=='tr'),
    num_workers=num_workers, pin_memory=True)
    return L(ds_tr,'tr'), L(ds_va,'va'), L(ds_te,'te')

def _split(files, val_ratio=0.15, test_ratio=0.15, seed=2025):
    import random
    rng = random.Random(seed); idx = list(range(len(files))); rng.shuffle(idx)
    n = len(idx); n_te = int(n*test_ratio); n_va = int(n*val_ratio)
    te, va, tr = idx[:n_te], idx[n_te:n_te+n_va], idx[n_te+n_va:]
    return tr, va, te


def build_loaders(root, img_dir, img_size, batch_size, val_ratio, test_ratio, seed, num_workers):
    files = sorted(glob.glob(os.path.join(root, img_dir, '*.nii*')))
    tr, va, te = _split(files, val_ratio, test_ratio, seed)
    get = lambda ids, aug: HipMRI2DImageOnly(root, img_dir, img_size,
    files=[files[i] for i in ids], aug=aug)
    ds_tr, ds_va, ds_te = get(tr, True), get(va, False), get(te, False)
    L = lambda ds, s: DataLoader(ds, batch_size=batch_size, shuffle=(s=='tr'), num_workers=num_workers, pin_memory=True)
    return L(ds_tr,'tr'), L(ds_va,'va'), L(ds_te,'te')