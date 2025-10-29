import torch
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_tm

def img_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2

@torch.no_grad()
def batch_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x,y in [-1,1], shape (B,1,H,W)
    x01 = img_to_01(x).clamp(0,1)
    y01 = img_to_01(y).clamp(0,1)

    eps = 1e-4
    x01 = x01 + eps * (torch.rand_like(x01) - 0.5)
    y01 = y01 + eps * (torch.rand_like(y01) - 0.5)
    return ssim_tm(x01, y01, data_range=1.0)

def recon_loss_fn(kind='l1'):
    return torch.nn.L1Loss() if kind.lower()=='l1' else torch.nn.MSELoss()