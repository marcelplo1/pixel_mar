import numpy as np
from scipy import stats
import torch
import torch.nn as nn

from model.model_utils import Attention, RMSNorm, SwiGLUFFN, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed
from utils.utils import patchify


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    @torch.compile
    def forward(self, x, feat_rope=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope)
        x = x + self.mlp(self.norm2(x))
        return x


class MAE(nn.Module):
    """
    Decoder-only variant: no encoder, GT patches are projected directly
    into the decoder. Masked positions receive a learnable mask token.
    """
    def __init__(
            self,
            img_size,
            patch_size=16,
            channels=3,
            num_classes=10,
            ema_decay=0.9999,
            encoder_dim=768,
            decoder_dim=768,
            encoder_depth=12,
            decoder_depth=12,
            encoder_num_heads=12,
            decoder_num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            buffer_size=64,
            min_mask_rate = 0.7,
            lable_dropout = 0.1,
            mae_config = None,
            bottleneck_dim = None
        ):
        super().__init__()

        self.patch_size = patch_size
        self.min_mask_rate = min_mask_rate
        self.buffer_size = buffer_size
        self.decoder_dim = decoder_dim
        self.img_size = img_size
        self.channels = channels
        self.seq_len = (img_size // patch_size) ** 2
        self.embed_dim = channels * patch_size**2
        self.bottleneck_dim = bottleneck_dim

        # Project raw patches directly to decoder dim (no encoder)
        bottleneck_dim = None #TODO Test also bottleneck for the decoder
        if bottleneck_dim is not None:
            self.x_proj = nn.Sequential(
                nn.Linear(self.embed_dim, bottleneck_dim, bias=False),
                nn.Linear(bottleneck_dim, decoder_dim, bias=True),
            )
        else:
            self.x_proj = nn.Linear(self.embed_dim, decoder_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.class_emb = nn.Embedding(num_classes, decoder_dim)
        self.fake_latent = nn.Parameter(torch.zeros(1, decoder_dim))

        self.decoder_pos_emb = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_dim), requires_grad=True
        )

        self.decoder_block = nn.ModuleList([
            Block(decoder_dim, decoder_num_heads, mlp_ratio=mlp_ratio,
                  attn_drop=dropout, proj_drop=dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        self.label_drop_prob = lable_dropout
        self.ema_decay = ema_decay
        self.ema_params = None

        self.initialize_weights()

        # RoPE for decoder
        half_head_dim = decoder_dim // decoder_num_heads // 2
        hw_seq_len = img_size // patch_size
        self.decoder_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.buffer_size
        )

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        grid_size = int(self.seq_len ** 0.5)
        pos_embed_grid = get_2d_sincos_pos_embed(self.decoder_dim, grid_size)
        full_pos_embed = torch.zeros(self.seq_len + self.buffer_size, self.decoder_dim)
        full_pos_embed[self.buffer_size:, :] = torch.from_numpy(pos_embed_grid).float()
        self.decoder_pos_emb.data.copy_(full_pos_embed.unsqueeze(0))

        if self.bottleneck_dim is not None:
            nn.init.xavier_uniform_(self.x_proj[0].weight)
            nn.init.xavier_uniform_(self.x_proj[1].weight)
            nn.init.constant_(self.x_proj[1].bias, 0)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.fake_latent, std=0.02)

    def forward(self, x, mask_orders, labels, num_visible=None, force_unconditional=False):
        x = patchify(x, self.patch_size)  # (B, N, embed_dim)
        B, N, D = x.shape

        x = self.x_proj(x)  # (B, N, decoder_dim)

        # Class conditioning with label dropout for CFG
        if force_unconditional:
            class_embedding = self.fake_latent.expand(B, -1)
        else:
            class_embedding = self.class_emb(labels)
            if self.training:
                drop_mask = (torch.rand(B, device=x.device) < self.label_drop_prob).unsqueeze(-1).to(x.dtype)
                class_embedding = drop_mask * self.fake_latent + (1 - drop_mask) * class_embedding

        if self.training:
            mask = self.random_masking(x, mask_orders, self.min_mask_rate)
            num_masked = int(mask[0].sum().item())
            num_vis = N - num_masked
        else:
            mask = torch.zeros(B, N, device=x.device)
            mask.scatter_(1, mask_orders[:, num_visible:].long(), 1.0)
            num_vis = num_visible

        ids_restore = torch.argsort(mask_orders, dim=1)

        # Reorder by mask_orders, split visible / masked, replace masked with mask_token
        x = torch.gather(x, dim=1, index=mask_orders.unsqueeze(-1).expand(-1, -1, self.decoder_dim))
        x_visible = x[:, :num_vis, :]
        mask_tokens = self.mask_token.repeat(B, N - num_vis, 1).to(x.dtype)

        x_full = torch.cat([x_visible, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim))

        # Buffer tokens carrying class conditioning
        buffer = torch.zeros(B, self.buffer_size, self.decoder_dim, device=x.device, dtype=x.dtype)
        buffer[:] = class_embedding.unsqueeze(1)
        x = torch.cat([buffer, x_full], dim=1)  # (B, buffer_size + N, decoder_dim)

        x = x + self.decoder_pos_emb

        for block in self.decoder_block:
            x = block(x, self.decoder_rope)

        z = self.decoder_norm(x)
        z = z[:, self.buffer_size:]

        return z, mask

    def random_masking(self, x, orders, min_mask_rate=0.7):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = stats.truncnorm((min_mask_rate - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    @torch.no_grad()
    def update_ema(self):
        ema_decay = self.ema_decay
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)