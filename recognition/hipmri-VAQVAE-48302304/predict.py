from __future__ import annotations
import os, argparse, torch, yaml, numpy as np, matplotlib.pyplot as plt
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_tm
from dataset import build_loaders_from_dirs
from modules import VQVAE

img_to_01 = lambda x: (x + 1) / 2

def batch_ssim(x, y):
    return ssim_tm(img_to_01(x), img_to_01(y), data_range=1.0)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data_hipmri_2d')
    ap.add_argument('--train_dir', default='keras_slices_train')
    ap.add_argument('--val_dir', default='keras_slices_validate')
    ap.add_argument('--test_dir', default='keras_slices_test')
    ap.add_argument('--work_dir', default='workdir/hipmri_vqvae')
    ap.add_argument('--img_size', type=int, nargs=2, default=[128,128])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--num_samples', type=int, default=8)
    return ap.parse_args()

def main():
    args = get_args()
    os.makedirs(os.path.join(args.work_dir,'samples'), exist_ok=True)

    # loaders for test/recon preview
    _, _, teL = build_loaders_from_dirs(
    root=args.data_root,
    train_dir=args.train_dir,
    val_dir=args.val_dir,
    test_dir=args.test_dir,
    img_size=tuple(args.img_size),
    batch_size=args.batch_size,
    seed=2025,
    num_workers=args.num_workers,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQVAE(1, 128, 64, 2, 512, 0.25, 0.99).to(device)
    ckpt = torch.load(os.path.join(args.work_dir,'best_vq.pt'), map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()

    # 1) Test SSIM
    ssim_sum=0; n=0
    with torch.no_grad():
        for img, _ in teL:
            img = img.to(device)
            rec, _, _, _ = model(img)
            ssim_sum += float(batch_ssim(rec, img)); n += 1
    test_ssim = ssim_sum/max(n,1)
    print('Test SSIM:', test_ssim)
    with open(os.path.join(args.work_dir,'test_ssim.txt'), 'w') as f:
        f.write(f"test_ssim: {test_ssim:.4f}\n")

    # 2) Save a few recon pairs & naive unconditional samples
    imgs, _ = next(iter(teL)); imgs = imgs[:args.num_samples].to(device)
    with torch.no_grad():
        rec, _, _, _ = model(imgs)
    for i in range(min(args.num_samples, imgs.size(0))):
        a = img_to_01(imgs[i]).cpu().numpy().squeeze()
        b = img_to_01(rec[i]).cpu().numpy().squeeze()
        plt.imsave(os.path.join(args.work_dir,'samples', f'recon_{i}_in.png'), a, cmap='gray')
        plt.imsave(os.path.join(args.work_dir,'samples', f'recon_{i}_out.png'), b, cmap='gray')

    # naive unconditional (toy): decode random latent (no prior)
    H = args.img_size[0] // 4; W = args.img_size[1] // 4
    z = torch.randn(args.num_samples, 64, H, W, device=device)
    if hasattr(model, 'dec'):
        with torch.no_grad():
            g = model.dec(z)
        for i in range(g.size(0)):
            gi = img_to_01(g[i]).cpu().numpy().squeeze()
            plt.imsave(os.path.join(args.work_dir,'samples', f'gen_{i}.png'), gi, cmap='gray')
    print('Saved to', os.path.join(args.work_dir,'samples'))

if __name__ == '__main__':
    main()