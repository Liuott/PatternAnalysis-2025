import torch
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_tm

def img_to_01(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1]
    return (x + 1) / 2

@torch.no_grad()
def batch_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x,y in [-1,1], shape (B,1,H,W)
    x01, y01 = img_to_01(x), img_to_01(y)
    return ssim_tm(x01, y01, data_range=1.0)

def recon_loss_fn(kind='l1'):
    return torch.nn.L1Loss() if kind.lower()=='l1' else torch.nn.MSELoss()