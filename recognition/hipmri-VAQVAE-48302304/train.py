from __future__ import annotations
import os, argparse, torch, torch.optim as optim, matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_tm
from tqdm import tqdm
from dataset import build_loaders
from modules import VQVAE

img_to_01 = lambda x: (x + 1) / 2 # [-1,1] -> [0,1]

def batch_ssim(x, y):
    return ssim_tm(img_to_01(x), img_to_01(y), data_range=1.0)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data_hipmri_2d')
    ap.add_argument('--images_dir', default='imagesTr')
    ap.add_argument('--work_dir', default='workdir/hipmri_vqvae')
    ap.add_argument('--img_size', type=int, nargs=2, default=[128,128])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--z_channels', type=int, default=64)
    ap.add_argument('--n_res_blocks', type=int, default=2)
    ap.add_argument('--codebook_size', type=int, default=512)
    ap.add_argument('--commit_beta', type=float, default=0.25)
    ap.add_argument('--ema_decay', type=float, default=0.99)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--early_stop_patience', type=int, default=15)
    ap.add_argument('--mixed_precision', action='store_true', default=True)
    return ap.parse_args()


def save_curves(hist, out_png):
    plt.figure()
    for k,v in hist.items(): plt.plot(v, label=k)
    plt.xlabel('epoch'); plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def main():
    args = get_args(); torch.manual_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)


    trL, vaL, teL = build_loaders(args.data_root, args.images_dir, args.img_size, args.batch_size, 0.15, 0.15, args.seed, args.num_workers)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQVAE(1, args.hidden, args.z_channels, args.n_res_blocks, args.codebook_size, args.commit_beta, args.ema_decay).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.mixed_precision)
    l1 = torch.nn.L1Loss()


    best_val = -1; bad = 0; hist = {'train_loss':[], 'val_ssim':[]}


    for ep in range(1, args.epochs+1):
        model.train(); tl=0
        pbar = tqdm(trL, desc=f'E{ep}/{args.epochs}')
        for img, _ in pbar:
            img = img.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.mixed_precision):
                rec, vq_loss, perp, _ = model(img)
                loss = l1(rec, img) + vq_loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item(); pbar.set_postfix(loss=f"{loss.item():.4f}", perp=f"{float(perp):.1f}")
        tl /= max(1,len(trL))


        # validation SSIM
        model.eval(); ssim_sum = 0; n=0
        with torch.no_grad():
            for img, _ in vaL:
                img = img.to(device)
                rec, _, _, _ = model(img)
                ssim_sum += float(batch_ssim(rec, img)); n += 1
        val_ssim = ssim_sum/max(n,1)
        hist['train_loss'].append(tl); hist['val_ssim'].append(val_ssim)
        print(f"Epoch {ep}: train={tl:.4f} val_ssim={val_ssim:.4f}")


        # save best by SSIM
        if val_ssim > best_val:
            best_val = val_ssim; bad = 0
            torch.save({'model': model.state_dict(), 'epoch': ep, 'val_ssim': val_ssim}, os.path.join(args.work_dir,'best_vq.pt'))
        else:
            bad += 1
            if bad >= args.early_stop_patience: break


        save_curves(hist, os.path.join(args.work_dir,'curves.png'))


    # final test
    ckpt = torch.load(os.path.join(args.work_dir,'best_vq.pt'), map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()
    ssim_sum=0; n=0
    with torch.no_grad():
        for img, _ in teL:
            img = img.to(device)
            rec, _, _, _ = model(img)
            ssim_sum += float(batch_ssim(rec, img)); n += 1
    test_ssim = ssim_sum/max(n,1)
    with open(os.path.join(args.work_dir,'test_ssim.txt'), 'w') as f:
        f.write(f"test_ssim: {test_ssim:.4f}\n")
    print('Test SSIM:', test_ssim)

if __name__ == '__main__':
    main()