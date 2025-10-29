import os, argparse, math, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import VQVAE            
from dataset import build_loaders_from_dirs

# --------- args ---------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--train_dir', default='train')
    ap.add_argument('--val_dir',   default='val')
    ap.add_argument('--test_dir',  default='test')
    ap.add_argument('--work_dir',  default='workdir/hipmri_vqvae')

    ap.add_argument('--img_size', nargs=2, type=int, default=[128,128])
    ap.add_argument('--batch_size', type=int, default=48)
    ap.add_argument('--num_workers', type=int, default=0)          
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--optimizer', choices=['adam','adamw'], default='adamw')
    ap.add_argument('--recon_loss', choices=['l1','mse'], default='l1')
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--mixed_precision', action='store_true')
    ap.add_argument('--clip_norm', type=float, default=0.0)

    # VQ 
    ap.add_argument('--codebook_size', type=int, default=256)
    ap.add_argument('--commit_beta', type=float, default=0.15)
    ap.add_argument('--ema_decay', type=float, default=0.99)
    ap.add_argument('--in_channels', type=int, default=1)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--z_channels', type=int, default=64)
    ap.add_argument('--n_res_blocks', type=int, default=2)

   
    ap.add_argument('--vq_warmup', type=int, default=5000,
                    help='steps before VQ loss starts (> total steps means off)')
    ap.add_argument('--vq_ramp', type=int, default=3000,
                    help='linear ramp steps to grow VQ weight from 0→1 after warmup')

    return ap.parse_args()

# --------- util ---------
def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_optimizer(name, params, lr):
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=(0.9,0.95), eps=1e-8, weight_decay=1e-4)
    else:
        return torch.optim.Adam(params, lr=lr, betas=(0.9,0.95), eps=1e-8)

def create_recon_loss(name):
    return (nn.L1Loss() if name=='l1' else nn.MSELoss())

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    
    import torch.nn.functional as F
    total_ssim_like, n = 0.0, 0
    for img, _meta in loader:
        img = img.to(device, non_blocking=True)
        rec, _ = model(img)
       
        mse = F.mse_loss(rec, img)
        ssim_like = float(1.0 - mse.clamp(0,1).item())
        total_ssim_like += ssim_like; n += 1
    return total_ssim_like / max(1,n)

# --------- train ---------
def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = build_loaders_from_dirs(
        args.data_root, args.train_dir, args.val_dir, args.test_dir,
        tuple(args.img_size), args.batch_size, args.seed, args.num_workers
    )

    train_loader.timeout = 0           
    train_loader.pin_memory = True
    val_loader.pin_memory = True
    test_loader.pin_memory = True

    model = VQVAE(
        in_channels=args.in_channels,
        hidden=args.hidden,
        z_channels=args.z_channels,
        n_res_blocks=args.n_res_blocks,
        codebook_size=args.codebook_size,
        commit_beta=args.commit_beta,
        ema_decay=args.ema_decay
    ).to(device)

    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)
    recon_loss_fn = create_recon_loss(args.recon_loss)

    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)

    best_val = -1.0
    best_path = os.path.join(args.work_dir, 'best_vq.pt')

    global_step = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"E{epoch}/{args.epochs}", ncols=120)
        running = 0.0; nstep = 0

        for img, meta in pbar:
            img = img.to(device, non_blocking=True)

       
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                rec, vq_stats = model(img)    # vq_stats
                loss_rec = recon_loss_fn(rec, img)
             
                if args.vq_warmup < 0:
                    vq_weight = 0.0
                else:
                    if global_step < args.vq_warmup:
                        vq_weight = 0.0
                    else:
                        if args.vq_ramp <= 0:
                            vq_weight = 1.0
                        else:
                            vq_weight = min(1.0, (global_step - args.vq_warmup) / float(args.vq_ramp))

                loss_vq = vq_stats.get('loss_vq', torch.tensor(0.0, device=device))
                loss = loss_rec + vq_weight * loss_vq


            if not torch.isfinite(loss):
                
                print(f"[warn] NaN/Inf loss -> skip step @ step={global_step} vq_w={vq_weight:.3f} "
                      f"loss_rec={float(loss_rec):.4g} loss_vq={float(loss_vq):.4g}")
                global_step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            if args.mixed_precision:
                scaler.scale(loss).backward()
                if args.clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if args.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()

            running += float(loss.detach().cpu())
            nstep += 1; global_step += 1
            perp = float(vq_stats.get('perplexity', torch.tensor(0.0)).detach().cpu())
            pbar.set_postfix(loss=f"{running/max(1,nstep):.4f}", perp=f"{perp:.1f}", vq_w=f"{vq_weight:.2f}")

        train_loss = running / max(1,nstep)

   
        val_ssim = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train={train_loss:.4f} val_ssim={val_ssim:.4f}")

     
        if val_ssim > best_val:
            best_val = val_ssim
            save_ckpt({'model': model.state_dict(),
                       'epoch': epoch,
                       'best_val': best_val}, best_path)


    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck['model'])
        print(f"[info] Loaded best ckpt from {best_path} @ epoch {ck.get('epoch','?')}, val_ssim={ck.get('best_val','?')}")
    test_ssim = evaluate(model, test_loader, device)
    print("=== Summary ===")
    print(f"Best Val SSIM: {best_val:.4f}")
    print(f"Test  SSIM:   {test_ssim:.4f}")
    print(f"Checkpoint:   {best_path}")

if __name__ == '__main__':
    main()
