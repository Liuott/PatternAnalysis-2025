# utils.py
from __future__ import annotations
import torch
import torch.nn.functional as F

def to_01(x: torch.Tensor) -> torch.Tensor:
 
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

def gaussian_window(window_size: int, sigma: float, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1)/2
    g = torch.exp(-(coords**2)/(2*sigma*sigma))
    g = g / (g.sum() + 1e-8)
    w = g[:, None] @ g[None, :]
    w = w / (w.sum() + 1e-8)
    return w[None, None, :, :]

def _ssim_single(x, y, window, C1=0.01**2, C2=0.03**2, eps=1e-8):
    mu_x = F.conv2d(x, window, padding='same')
    mu_y = F.conv2d(y, window, padding='same')

    sigma_x = F.conv2d(x * x, window, padding='same') - mu_x * mu_x
    sigma_y = F.conv2d(y * y, window, padding='same') - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, window, padding='same') - mu_x * mu_y


    sigma_x = torch.clamp(sigma_x, min=0.0)
    sigma_y = torch.clamp(sigma_y, min=0.0)

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + eps
    ssim_map = num / den
    return torch.clamp(ssim_map, 0.0, 1.0)

@torch.inference_mode()
def batch_ssim(x: torch.Tensor, y: torch.Tensor, win_size: int = 11, sigma: float = 1.5) -> float:

    x = to_01(x)
    y = to_01(y)
    x = torch.nan_to_num(x); y = torch.nan_to_num(y)

    device = x.device
    window = gaussian_window(win_size, sigma, device=device).to(x.dtype)


    ssim_map = _ssim_single(x, y, window)
    ssim_val = ssim_map.mean().item()
    if not (ssim_val == ssim_val):  
        return 0.0
    return float(ssim_val)
