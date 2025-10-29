from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_codes=512, emb_dim=64, decay=0.99, eps=1e-5, beta=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.emb_dim = emb_dim
        self.decay = decay
        self.eps = eps
        self.beta = beta
        self.register_buffer('embed', torch.randn(n_codes, emb_dim))
        self.register_buffer('cluster_size', torch.zeros(n_codes))
        self.register_buffer('embed_avg', torch.randn(n_codes, emb_dim))
        nn.init.normal_(self.embed, std=0.1)
        self.embed_avg.copy_(self.embed)


    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, codes: torch.Tensor):
        # flat: (BHW, D), codes: (BHW,)
        onehot = F.one_hot(codes, self.n_codes).type_as(flat) # (BHW, K)

        self.cluster_size.mul_(self.decay).add_(onehot.sum(0) * (1 - self.decay))

        embed_sum = onehot.t() @ flat # (K,D)
        self.embed_avg.mul_(self.decay).add_(embed_sum * (1 - self.decay))

        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
        self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))


    def forward(self, z_e: torch.Tensor):
        # z_e: (B, D, H, W)
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D) # (BHW, D)

         # Cosine similarity (normalized) — more stable than L2 distance early on
        flat_n = F.normalize(flat, dim=1, eps=1e-8)      # (BHW, D)
        emb_n  = F.normalize(self.embed, dim=1, eps=1e-8)  # (K, D)
        # Cosine sim: larger is better → use argmax directly
        sim = flat_n @ emb_n.t()                         # (BHW, K)
        codes = sim.argmax(dim=1)                        # (BHW,)
        
        z_q = self.embed[codes].view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        # losses
        commit = self.beta * F.mse_loss(z_e.detach(), z_q)
        codebk = F.mse_loss(z_e, z_q.detach())
        vq_loss = commit + codebk
        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        # EMA update & perplexity
        with torch.no_grad():
            self._ema_update(flat, codes)
        avg_probs = torch.mean(F.one_hot(codes, self.n_codes).float(), dim=0)
        perp = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return z_q_st, vq_loss, perp, codes.view(B, H, W)


# ---------- Encoder / Decoder ----------
class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.block = nn.Sequential(
        nn.ReLU(inplace=True), nn.Conv2d(c, c, 3, 1, 1),
        nn.ReLU(inplace=True), nn.Conv2d(c, c, 1)
        )
    def forward(self, x):
        return x + self.block(x)
        


class Encoder(nn.Module):
    def __init__(self, in_ch=1, hidden=128, z_channels=64, n_res_blocks=2):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(in_ch, hidden, 4, 2, 1), # 1/2
        nn.ReLU(True),
        nn.Conv2d(hidden, hidden, 4, 2, 1), # 1/4
        nn.ReLU(True),
        nn.Conv2d(hidden, z_channels, 3, 1, 1),
        )
        self.res = nn.Sequential(*[ResBlock(z_channels) for _ in range(n_res_blocks)])
    def forward(self, x):
        x = self.net(x)
        return self.res(x)


class Decoder(nn.Module):
    def __init__(self, out_ch=1, hidden=128, z_channels=64, n_res_blocks=2):
        super().__init__()
        self.pre = nn.Sequential(*[ResBlock(z_channels) for _ in range(n_res_blocks)])
        self.net = nn.Sequential(
        nn.ReLU(True), nn.ConvTranspose2d(z_channels, hidden, 4, 2, 1), # x2
        nn.ReLU(True), nn.ConvTranspose2d(hidden, hidden // 2, 4, 2, 1), # x2
        nn.ReLU(True), nn.Conv2d(hidden // 2, out_ch, 3, 1, 1), nn.Tanh(),
        )
    def forward(self, z):
        z = self.pre(z)
        return self.net(z)


# ---------- VQ‑VAE (single level) ----------
class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden=128, z_channels=64, n_res_blocks=2,
        codebook_size=512, commit_beta=0.25, ema_decay=0.99):
        super().__init__()
        self.enc = Encoder(in_channels, hidden, z_channels, n_res_blocks)
        self.vq = VectorQuantizerEMA(codebook_size, z_channels, decay=ema_decay, beta=commit_beta)
        self.dec = Decoder(in_channels, hidden, z_channels, n_res_blocks)
    def forward(self, x):
        z_e = self.enc(x)
        z_q, vq_loss, perp, codes = self.vq(z_e)
        x_rec = self.dec(z_q)
        return x_rec, vq_loss, perp, codes