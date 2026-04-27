import math
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from model.model_utils import Attention, AttentionRoPE, RMSNorm, SwiGLUFFN, TimestepEmbedder, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = AttentionRoPE(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x, feat_rope=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope)
        x = x + self.mlp(self.norm2(x))
        return x


class ArEncoderDecoder(nn.Module):
    """
    Encoder-decoder variant: encoder processes only visible patches (variable length),
    decoder receives encoded visible + mask tokens and produces full-sequence latents.
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
            decoder_num_heads=12,
            encoder_num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            class_token_size=16,
            mask_rate_token_size=8,
            min_mask_rate=0.7,
            min_s=0.0,
            lable_dropout=0.1,
            latent_dim=None,
            gt_noise_scale=0.0,
            mask_condition=None,
            train_mask_schedule='truncnorm'
        ):
        super().__init__()

        self.patch_size = patch_size
        self.min_mask_rate = min_mask_rate
        self.min_s = min_s
        self.train_mask_schedule = train_mask_schedule
        self.gt_noise_scale = gt_noise_scale
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.img_size = img_size
        self.channels = channels
        self.seq_len = (img_size // patch_size) ** 2

        self.latent_dim = latent_dim
        self.embed_dim = latent_dim if latent_dim is not None else channels * patch_size**2

        self.class_token_size = class_token_size
        self.mask_rate_token_size = mask_rate_token_size if mask_condition else 0
        self.total_c_token_size = class_token_size + self.mask_rate_token_size

        # Input projection
        self.x_proj = nn.Linear(self.embed_dim, self.encoder_dim, bias=True)
        self.x_ln = nn.LayerNorm(self.encoder_dim, eps=1e-6)

        self.decoder_embed = nn.Linear(self.encoder_dim, self.decoder_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        self.class_emb = nn.Embedding(num_classes, self.encoder_dim)
        self.fake_latent = nn.Parameter(torch.zeros(1, self.encoder_dim))

        # Extra Embeddings
        self.mask_rate_emb = TimestepEmbedder(encoder_dim) if mask_condition else None

        self.encoder_pos_emb = nn.Parameter(
            torch.zeros(1, self.seq_len + self.total_c_token_size, self.encoder_dim), requires_grad=True
        )
        self.decoder_pos_emb = nn.Parameter(
            torch.zeros(1, self.seq_len + self.total_c_token_size, self.decoder_dim), requires_grad=True
        )

        # Encoder blocks (no RoPE — variable-length visible sequences)
        self.encoder_block = nn.ModuleList([
            Block(self.encoder_dim, encoder_num_heads, mlp_ratio=mlp_ratio,
                  attn_drop=dropout, proj_drop=dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(self.encoder_dim, eps=1e-6)

        # Decoder blocks (with RoPE — full-length sequences)
        self.decoder_block = nn.ModuleList([
            Block(self.decoder_dim, decoder_num_heads, mlp_ratio=mlp_ratio,
                  attn_drop=dropout, proj_drop=dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(self.decoder_dim, eps=1e-6)

        self.reconstruction_head = nn.Linear(self.decoder_dim, self.embed_dim)
        self.diffusion_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.decoder_dim))

        self.label_drop_prob = lable_dropout
        self.ema_decays = ema_decay if isinstance(ema_decay, list) else [ema_decay]
        self.ema_params_list = None

        self.initialize_weights()

        # RoPE for decoder
        half_head_dim = self.decoder_dim // decoder_num_heads // 2
        hw_seq_len = img_size // patch_size
        self.decoder_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.total_c_token_size
        )

    def initialize_weights(self):
        # Match MAR: normal_(std=0.02) for all learnable embeddings/pos_embs.
        nn.init.normal_(self.class_emb.weight, std=0.02)
        nn.init.normal_(self.fake_latent, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.encoder_pos_emb, std=0.02)
        nn.init.normal_(self.decoder_pos_emb, std=0.02)
        nn.init.normal_(self.diffusion_pos_emb, std=0.02)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if self.mask_rate_emb is not None:
            nn.init.normal_(self.mask_rate_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(self.mask_rate_emb.mlp[2].weight, std=0.02)

    def forward_encoder(self, x, mask, class_emb, force_unconditional=False):
        """
        MAR-style encoder: prepend buffer, add pos_emb to full sequence in raster
        order, LayerNorm, then drop masked tokens via boolean indexing.
        """
        B, N, D = x.shape

        # concat buffer
        x = torch.cat(
            [torch.zeros(B, self.total_c_token_size, D, device=x.device, dtype=x.dtype), x],
            dim=1,
        )
        mask_with_buffer = torch.cat(
            [torch.zeros(B, self.total_c_token_size, device=x.device, dtype=mask.dtype), mask],
            dim=1,
        )

        # random drop class embedding during training
        if force_unconditional:
            class_embedding = self.fake_latent.expand(B, -1)
        else:
            if self.training and self.label_drop_prob > 0:
                drop_latent_mask = (torch.rand(B, device=x.device) < self.label_drop_prob).unsqueeze(-1).to(x.dtype)
                class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_emb
            else:
                class_embedding = class_emb

        x[:, :self.class_token_size] = class_embedding.unsqueeze(1)

        if self.mask_rate_emb is not None:
            num_vis = int(N - mask[0].sum().item())
            mask_rate = torch.tensor([1.0 - num_vis / N], device=x.device)
            mask_rate_embedding = self.mask_rate_emb(mask_rate).expand(B, -1)
            x[:, self.class_token_size:self.class_token_size + self.mask_rate_token_size] = mask_rate_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_emb
        x = self.x_ln(x)

        # dropping
        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(B, -1, D)

        for block in self.encoder_block:
            x = block(x)

        encoded = self.encoder_norm(x)
        return encoded

    def forward_decoder(self, x, mask):
        """
        MAR-style decoder: scatter encoded visible tokens back into their raster
        positions, fill the rest with `mask_token`, add decoder pos_emb, run blocks.
        """
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat(
            [torch.zeros(x.size(0), self.total_c_token_size, device=x.device, dtype=mask.dtype), mask],
            dim=1,
        )

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_emb

        for block in self.decoder_block:
            x = block(x, self.decoder_rope)

        z = self.decoder_norm(x)
        z = z[:, self.total_c_token_size:]
        z = z + self.diffusion_pos_emb

        return z

    def forward(self, x, mask_orders=None, labels=None, num_visible=None, force_unconditional=False):
        x = self.x_proj(x)  # (B, N, encoder_dim)
        B, N, D = x.shape
        class_embedding = self.class_emb(labels)

        if self.training:
            mask = self.random_masking(x, self.min_mask_rate, self.min_s)
            num_vis = N - int(mask[0].sum().item())

            # Mask-ratio-dependent noise: more noise when fewer patches are masked
            if self.gt_noise_scale > 0:
                visible_ratio = num_vis / N
                x = x + self.gt_noise_scale * visible_ratio * torch.randn_like(x)
        else:
            num_vis = num_visible
            mask = torch.zeros(B, N, device=x.device)
            mask.scatter_(1, mask_orders[:, num_vis:].long(), 1.0)

        x = self.forward_encoder(x, mask, class_embedding, force_unconditional=force_unconditional)
        z = self.forward_decoder(x, mask)
        x_recon = self.reconstruction_head(z)

        return z, mask, x_recon

    def random_masking(self, x, min_mask_rate=0.7, min_s=0.0):
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
                if mask.sum() >= 2:  # TODO explore the min number of tokens
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