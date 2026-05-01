import math
import numpy as np
from scipy import stats
import torch
import torch.nn as nn

from model.model_utils import Attention, AttentionRoPE, RMSNorm, SwiGLUFFN, TimestepEmbedder, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed
from utils.utils import patchify


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = AttentionRoPE(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    @torch.compile
    def forward(self, x, feat_rope=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope)
        x = x + self.mlp(self.norm2(x))
        return x

class ArDecoder(nn.Module):
    """
    Decoder-only variant: no encoder, GT patches are projected directly into the decoder. Masked positions receive a learnable mask token.
    """
    def __init__(
            self,
            img_size,
            patch_size=16,
            channels=3,
            num_classes=10,
            ema_decay=0.9999,
            decoder_dim=768,
            decoder_depth=12,
            decoder_num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            class_token_size=16,
            mask_rate_token_size=8,
            min_mask_rate = 0.7,
            min_s = 0.0,
            lable_dropout = 0.1,
            latent_dim = None,
            gt_noise_scale = 0.0,
            mask_condition = None,
            train_mask_schedule = 'truncnorm'

        ):
        super().__init__()

        self.patch_size = patch_size
        self.min_mask_rate = min_mask_rate
        self.min_s = min_s
        self.train_mask_schedule = train_mask_schedule
        self.gt_noise_scale = gt_noise_scale
        self.decoder_dim = decoder_dim
        self.img_size = img_size
        self.channels = channels
        self.seq_len = (img_size // patch_size) ** 2
        
        self.latent_dim = latent_dim
        self.embed_dim = latent_dim if latent_dim is not None else channels * patch_size**2

        self.class_token_size = class_token_size
        self.mask_rate_token_size = mask_rate_token_size if mask_condition else 0
        self.total_c_token_size = class_token_size + self.mask_rate_token_size
        
        self.x_proj = nn.Linear(self.embed_dim, decoder_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.class_emb = nn.Embedding(num_classes, decoder_dim)
        self.fake_latent = nn.Parameter(torch.zeros(1, decoder_dim))

        # Extra Embeddings
        self.mask_rate_emb = TimestepEmbedder(decoder_dim) if mask_condition else None

        self.decoder_pos_emb = nn.Parameter(
            torch.zeros(1, self.seq_len + self.total_c_token_size, decoder_dim), requires_grad=True
        )

        self.decoder_block = nn.ModuleList([
            Block(decoder_dim, decoder_num_heads, mlp_ratio=mlp_ratio,
                  attn_drop=dropout, proj_drop=dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        self.reconstruction_head = nn.Linear(decoder_dim, self.embed_dim)
        self.diffusion_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, decoder_dim))

        self.label_drop_prob = lable_dropout
        self.ema_decays = ema_decay if isinstance(ema_decay, list) else [ema_decay]
        self.ema_params_list = None

        self.initialize_weights()

        # RoPE for decoder
        half_head_dim = decoder_dim // decoder_num_heads // 2
        hw_seq_len = img_size // patch_size
        self.decoder_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.total_c_token_size
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
        full_pos_embed = torch.zeros(self.seq_len + self.total_c_token_size, self.decoder_dim)
        full_pos_embed[self.total_c_token_size:, :] = torch.from_numpy(pos_embed_grid).float()
        self.decoder_pos_emb.data.copy_(full_pos_embed.unsqueeze(0))

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.fake_latent, std=0.02)
        nn.init.normal_(self.diffusion_pos_emb, std=0.02)

        if self.mask_rate_emb is not None:
            nn.init.normal_(self.mask_rate_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(self.mask_rate_emb.mlp[2].weight, std=0.02)

    def forward(self, x, mask_orders=None, labels=None, num_visible=None, force_unconditional=False, return_intermediates=False):
        x = self.x_proj(x)  # (B, N, decoder_dim)
        x_emb = x.clone().detach()
        B, N, D = x.shape

        # Class conditioning
        if force_unconditional:
            class_embedding = self.fake_latent.expand(B, -1)
        else:
            class_embedding = self.class_emb(labels)
            if self.training and self.label_drop_prob > 0:
                drop_mask = (torch.rand(B, device=x.device) < self.label_drop_prob).unsqueeze(-1).to(x.dtype)
                class_embedding = drop_mask * self.fake_latent + (1 - drop_mask) * class_embedding

        if self.training:
            mask = self.random_masking(x, self.min_mask_rate, self.min_s)
            num_masked = int(mask[0].sum().item())
            num_vis = N - num_masked
            # Mask-ratio-dependent noise: more noise when fewer patches are masked
            if self.gt_noise_scale > 0:
                visible_ratio = num_vis / N
                x = x + self.gt_noise_scale * visible_ratio * torch.randn_like(x)
        else:
            num_vis = num_visible
            mask = torch.zeros(B, N, device=x.device)
            mask.scatter_(1, mask_orders[:, num_vis:].long(), 1.0)

        if self.mask_rate_emb is not None:
            mask_rate = torch.tensor([1.0 - num_vis / N], device=x.device)
            mask_rate_embedding = self.mask_rate_emb(mask_rate).expand(B, -1)

        # Replace masked positions with the learnable mask token
        mask_tokens = self.mask_token.to(x.dtype).expand(B, N, -1)
        x_full = torch.where(mask.unsqueeze(-1).bool(), mask_tokens, x)

        class_tokens = torch.zeros(B, self.class_token_size, self.decoder_dim, device=x.device, dtype=x.dtype)
        class_tokens[:, :self.class_token_size] = class_embedding.unsqueeze(1)
        if self.mask_rate_emb is not None:
            mask_rate_tokens = mask_rate_embedding.unsqueeze(1).expand(B, self.mask_rate_token_size, -1)
            x = torch.cat([class_tokens, mask_rate_tokens, x_full], dim=1)
        else:
            x = torch.cat([class_tokens, x_full], dim=1)

        x = x + self.decoder_pos_emb

        for i, block in enumerate(self.decoder_block):
            x = block(x, self.decoder_rope)

        z = self.decoder_norm(x)
        z = z[:, self.total_c_token_size:]
        z = z + self.diffusion_pos_emb

        x_recon = self.reconstruction_head(z)

        return z, mask, x_recon

    def random_masking(self, x, min_mask_rate=0.7, min_s=0.2):
        bsz, seq_len, embed_dim = x.shape
        if self.train_mask_schedule == 'exp':
            eps = 1e-3
            valid_mask_generated = False
            while not valid_mask_generated:
                t = torch.rand((), device=x.device)
                t = torch.clamp(t, min=min_s)
                p_mask = (1 - eps) * t + eps
                p_mask = p_mask.expand(seq_len)
                random_vals = torch.rand((seq_len,), device=x.device)
                mask_prob = 1 - torch.exp(-5 * p_mask)
                mask = (random_vals < mask_prob).float()
                if mask.sum() >= 2:
                    valid_mask_generated = True
        elif self.train_mask_schedule == 'exp_v2':
            eps = 1e-3
            valid_mask_generated = False
            while not valid_mask_generated:
                t = (1.0 - min_s) * torch.rand((), device=x.device) + min_s
                p_mask = (1 - eps) * t + eps
                p_mask = p_mask.expand(seq_len)
                random_vals = torch.rand((seq_len,), device=x.device)
                mask_prob = 1 - torch.exp(-5 * p_mask)
                mask = (random_vals < mask_prob).float()
                if mask.sum() >= 2:
                    valid_mask_generated = True
        else:  # truncnorm (default)
            mask_rate = stats.truncnorm((min_mask_rate - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
            num_masked_tokens = int(np.ceil(seq_len * mask_rate))
            mask = torch.zeros(seq_len, device=x.device)
            mask[:num_masked_tokens] = 1.0

        mask = mask.expand(bsz, -1).clone()
        for i in range(bsz):
            random_indices = torch.randperm(seq_len, device=x.device)
            mask[i] = mask[i][random_indices]
        return mask

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for ema_params, decay in zip(self.ema_params_list, self.ema_decays):
            for targ, src in zip(ema_params, source_params):
                targ.detach().mul_(decay).add_(src, alpha=1 - decay)