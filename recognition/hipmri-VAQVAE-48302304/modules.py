import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
#  Basic Blocks
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden=128, z_channels=64, n_res_blocks=2):
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

    def forward(self, x):
        h = self.conv(x)
        h = self.res(h)
        return h


class Decoder(nn.Module):
    def __init__(self, out_channels=1, hidden=128, z_channels=64, n_res_blocks=2):
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

    def forward(self, z):
        h = self.res(z)
        x = self.deconv(h)
        return x



class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=64, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        embedding = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer('embedding', embedding)            # [D, K]
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_weight', embedding.clone())   # EMA copy

    @torch.no_grad()
    def _ema_update(self, z_e, codes_onehot):
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)   # [BHW, D]

        cluster_size = codes_onehot.sum(0)  # [K]
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

        embed_sum = flat.t() @ codes_onehot  # [D, K]
        self.ema_weight.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
        normalized_weight = self.ema_weight / cluster_size.unsqueeze(0)
        self.embedding.copy_(normalized_weight)

    def forward(self, z_e):

        with torch.cuda.amp.autocast(enabled=False):
            z_e_fp32 = z_e.float()                           # [B, D, H, W] in FP32
            B, D, H, W = z_e_fp32.shape
            flat = z_e_fp32.permute(0, 2, 3, 1).contiguous().view(-1, D)  # [BHW, D]
            embed = self.embedding.float()                   # [D, K] in FP32

            # dist = ||z||^2 + ||e||^2 - 2 z·e
            dist = (
                flat.pow(2).sum(dim=1, keepdim=True)
                + embed.pow(2).sum(dim=0, keepdim=True)
                - 2 * flat @ embed
            )  # [BHW, K]

            codes = torch.argmin(dist, dim=1)               # [BHW]
            codes_onehot = F.one_hot(codes, self.num_embeddings).to(flat.dtype)  # FP32

            # z_q (FP32)
            z_q = (codes_onehot @ embed.t()).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        
            z_q_st = z_e + (z_q.to(z_e.dtype) - z_e).detach()

            if self.training:
                self._ema_update(z_e_fp32.detach(), codes_onehot.detach())

          
            avg_probs = codes_onehot.float().mean(dim=0).clamp_min(1e-12)
            perplexity = torch.exp(-(avg_probs * avg_probs.log()).sum())
            perplexity = perplexity.to(z_e.dtype)

        return z_q_st, perplexity


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden=128,
        z_channels=64,
        n_res_blocks=2,
        codebook_size=256,
        commit_beta=0.15,
        ema_decay=0.99,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden, z_channels, n_res_blocks)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=codebook_size, embedding_dim=z_channels, decay=ema_decay
        )
        self.decoder = Decoder(in_channels, hidden, z_channels, n_res_blocks)
        self.commit_beta = commit_beta

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.clamp(-1.0, 1.0)

        z_e = self.encoder(x)  # [B, D, H, W]

        z_q, perplexity = self.quantizer(z_e)

        x_rec = self.decoder(z_q)

        vq_loss = self.commit_beta * F.mse_loss(z_e.detach(), z_q)

        vq_stats = {
            'perplexity': perplexity,   # tensor
            'loss_vq': vq_loss,         # tensor
        }
        return x_rec, vq_loss, vq_stats