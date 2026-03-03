import math
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
    def __init__(
            self,
            img_size,
            patch_size=16,
            channels=3,
            num_classes=1000,
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
            min_mask_rate=0.7,
            lable_dropout=0.1,
            mae_config=None
        ):
        super().__init__()

        self.patch_size = patch_size
        self.min_mask_rate = min_mask_rate
        self.buffer_size = buffer_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.img_size = img_size
        self.channels = channels
        self.seq_len = (img_size // patch_size) ** 2
        self.embed_dim = channels * patch_size**2

        self.x_proj = nn.Linear(self.embed_dim, self.encoder_dim, bias=True)
        self.x_ln = nn.LayerNorm(encoder_dim, eps=1e-6)
        self.decoder_embed = nn.Linear(self.encoder_dim, self.decoder_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.class_emb = nn.Embedding(num_classes, encoder_dim)

        self.encoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_dim), requires_grad=True)
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_dim), requires_grad=True)

        self.encoder_block = nn.ModuleList([
            Block(encoder_dim, encoder_num_heads, mlp_ratio=mlp_ratio,
                  attn_drop=dropout, proj_drop=dropout)
            for i in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_dim, eps=1e-6)

        self.decoder_block = nn.ModuleList([
            Block(decoder_dim, decoder_num_heads, mlp_ratio=mlp_ratio,
                  attn_drop=dropout, proj_drop=dropout)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        self.label_drop_prob = lable_dropout
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_dim))

        self.ema_decay = ema_decay
        self.ema_params = None

        # RoPE for decoder (encoder uses variable-length masked sequences, no RoPE)
        half_head_dim = self.decoder_dim // decoder_num_heads // 2
        hw_seq_len = self.img_size // patch_size
        self.decoder_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.buffer_size
        )

        self.initialize_weights()

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
        pos_embed_grid = get_2d_sincos_pos_embed(self.encoder_dim, grid_size)
        full_pos_embed = torch.zeros(self.seq_len + self.buffer_size, self.encoder_dim)
        full_pos_embed[self.buffer_size:, :] = torch.from_numpy(pos_embed_grid).float()
        self.encoder_pos_emb.data.copy_(full_pos_embed.unsqueeze(0))

        pos_embed_grid = get_2d_sincos_pos_embed(self.decoder_dim, grid_size)
        full_pos_embed = torch.zeros(self.seq_len + self.buffer_size, self.decoder_dim)
        full_pos_embed[self.buffer_size:, :] = torch.from_numpy(pos_embed_grid).float()
        self.decoder_pos_emb.data.copy_(full_pos_embed.unsqueeze(0))

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.fake_latent, std=.02)

    def forward_encoder(self, x, mask, class_emb, force_unconditional=False):
        x = self.x_proj(x)
        bsz, seq_len, embed_dim = x.shape

        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        if force_unconditional:
            class_embedding = self.fake_latent.expand(bsz, -1)
        elif self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_emb
        else:
            class_embedding = class_emb

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_emb
        x = self.x_ln(x)

        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, self.encoder_dim)

        for block in self.encoder_block:
            x = block(x)

        encoded = self.encoder_norm(x)
        return encoded

    def forward_decoder(self, x, mask):
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1-mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        x = x_after_pad + self.decoder_pos_emb

        for block in self.decoder_block:
            x = block(x, self.decoder_rope)

        decoded = self.decoder_norm(x)
        decoded = decoded[:, self.buffer_size:]

        return decoded

    def forward(self, x, mask_orders, labels, num_visible=None, force_unconditional=False):
        """
        x: [B, C, H, W] image (both training and eval).
        mask_orders: [B, N] shuffle orders defining unmasking sequence.
        Training: generates random mask from orders. Returns (z, mask).
        Eval:     builds mask from orders + num_visible. Returns (z, mask).
        """
        x = patchify(x, self.patch_size)
        B, N, D = x.shape
        class_embedding = self.class_emb(labels)

        if self.training:
            mask = self.random_masking(x, orders=mask_orders, min_mask_rate=self.min_mask_rate)
        else:
            mask = torch.zeros(B, N, device=x.device)
            indices_to_mask = mask_orders[:, num_visible:]
            mask.scatter_(1, indices_to_mask.long(), 1.0)

        encoded = self.forward_encoder(x, mask, class_embedding, force_unconditional=force_unconditional)
        z = self.forward_decoder(encoded, mask)
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