# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple



# Basic Blocks
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 128, z_channels: int = 64, n_res_blocks: int = 2):
        super().__init__()
        # 128x128 -> 64x64 -> 32x32
        layers = [
            nn.Conv2d(in_channels, hidden, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),       # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, z_channels, kernel_size=3, padding=1),
        ]
        self.conv = nn.Sequential(*layers)
        self.res = nn.Sequential(*[ResidualBlock(z_channels) for _ in range(n_res_blocks)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.res(h)
        return h


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 1, hidden: int = 128, z_channels: int = 64, n_res_blocks: int = 2):
        super().__init__()
        self.res = nn.Sequential(*[ResidualBlock(z_channels) for _ in range(n_res_blocks)])
        self.deconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(z_channels, hidden, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride=2, padding=1),     # 128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.res(z)
        x = self.deconv(h)
        return x



# Vector Quantizer (EMA)

class VectorQuantizerEMA(nn.Module):

    def __init__(self, num_embeddings: int = 256, embedding_dim: int = 64, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        # [D, K]
        embed = torch.randn(embedding_dim, num_embeddings) * 0.02
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", embed.clone())

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, codes_onehot: torch.Tensor):
        # z_e: [B, D, H, W]  -> [BHW, D]
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # [N, D]

        cluster_size = codes_onehot.sum(0)  # [K]
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)

        embed_sum = flat.t() @ codes_onehot  # [D, K]
        self.ema_weight.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)


        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
        normalized_weight = self.ema_weight / cluster_size.unsqueeze(0)  # [D, K]
        self.embedding.copy_(normalized_weight)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, D, H, W = z_e.shape
        assert D == self.embedding_dim, f"embedding_dim={self.embedding_dim}, but got {D}"
        device = z_e.device
        dtype = z_e.dtype

        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # [N, D]

        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)                           # [N, 1]
            + self.embedding.pow(2).sum(dim=0, keepdim=True).to(dtype)     # [1, K]
            - 2 * flat @ self.embedding.to(dtype)                          # [N, K]
        )  # [N, K]

        codes = torch.argmin(dist, dim=1)  # [N]
        codes_onehot = F.one_hot(codes, self.num_embeddings).to(device=device, dtype=flat.dtype)  # [N, K]

        # z_q: [N, D] -> [B, H, W, D] -> [B, D, H, W]
        z_q = (codes_onehot @ self.embedding.t().to(dtype)).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        if self.training:
            self._ema_update(z_e.detach(), codes_onehot.detach())

        # perplexity
        avg_probs = codes_onehot.float().mean(dim=0) + 1e-10  # [K]
        perplexity = torch.exp(-(avg_probs * avg_probs.log()).sum())

        return z_q_st, perplexity, codes_onehot



#  VQ-VAE (EMA)

class VQVAE(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        hidden: int = 128,
        z_channels: int = 64,
        n_res_blocks: int = 2,
        codebook_size: int = 256,
        commit_beta: float = 0.15,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden, z_channels, n_res_blocks)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=codebook_size, embedding_dim=z_channels, decay=ema_decay
        )
        self.decoder = Decoder(in_channels, hidden, z_channels, n_res_blocks)
        self.commit_beta = commit_beta

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        x = x.clamp(-1.0, 1.0)

        z_e = self.encoder(x)  # [B, D, H, W]

        z_q_st, perplexity, _ = self.quantizer(z_e)

        recon = self.decoder(z_q_st)

        commit_loss = self.commit_beta * F.mse_loss(z_e, z_q_st.detach(), reduction="mean")


        vq_stats = {
            "perplexity": perplexity.detach(),
            "commit_loss": commit_loss.detach(),
        }
        return recon, commit_loss, vq_stats
