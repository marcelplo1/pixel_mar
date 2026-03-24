import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from model.model_utils import Attention, RMSNorm, SwiGLUFFN, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
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
    Encoder and decoder share the same hidden dimension.
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
            mlp_ratio=4.0,
            dropout=0.1,
            buffer_size=64,
            min_mask_rate=0.7,
            lable_dropout=0.1,
            bottleneck_dim=None,
            latent_dim=None
        ):
        super().__init__()

        self.patch_size = patch_size
        self.min_mask_rate = min_mask_rate
        self.buffer_size = buffer_size
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.img_size = img_size
        self.channels = channels
        self.seq_len = (img_size // patch_size) ** 2

        self.latent_dim = latent_dim
        self.embed_dim = latent_dim if latent_dim is not None else channels * patch_size**2

        # Input projection
        self.x_proj = nn.Linear(self.embed_dim, self.encoder_dim, bias=True)
        self.x_ln = nn.LayerNorm(self.encoder_dim, eps=1e-6)

        self.decoder_embed = nn.Linear(self.encoder_dim, self.decoder_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        self.class_emb = nn.Embedding(num_classes, self.encoder_dim)
        self.fake_latent = nn.Parameter(torch.zeros(1, self.encoder_dim))

        self.encoder_pos_emb = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, self.encoder_dim), requires_grad=True
        )
        self.decoder_pos_emb = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, self.decoder_dim), requires_grad=True
        )

        # Encoder blocks (no RoPE — variable-length visible sequences)
        self.encoder_block = nn.ModuleList([
            Block(self.encoder_dim, decoder_num_heads, mlp_ratio=mlp_ratio,
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
        nn.init.trunc_normal_(self.fake_latent, std=0.02)

    def forward_encoder(self, x, orders, num_visible, class_emb, force_unconditional=False):
        """
        Encode only visible patches. Patches are reordered by orders so that
        visible ones come first.
        """
        x = self.x_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # Reorder patches by mask order (visible first)
        x = torch.gather(x, dim=1, index=orders.unsqueeze(-1).expand(-1, -1, embed_dim))
        buffer_tokens = torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device, dtype=x.dtype)
        x = torch.cat([buffer_tokens, x], dim=1)

        # Class conditioning with label dropout
        if force_unconditional:
            class_embedding = self.fake_latent.expand(bsz, -1)
        else:
            if self.training:
                drop_mask = (torch.rand(bsz, device=x.device) < self.label_drop_prob).unsqueeze(-1).to(x.dtype)
                class_embedding = drop_mask * self.fake_latent + (1 - drop_mask) * class_emb
            else:
                class_embedding = class_emb

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_emb
        x = self.x_ln(x)

        # Keep only buffer + visible tokens
        x = x[:, :self.buffer_size + num_visible, :]

        for block in self.encoder_block:
            x = block(x)

        encoded = self.encoder_norm(x)
        return encoded

    def forward_decoder(self, x, ids_restore):
        """
        Decode full sequence: unshuffle encoded visible tokens, insert mask tokens
        for masked positions, then process with decoder blocks + RoPE.
        """
        x = self.decoder_embed(x)

        x_buffer = x[:, :self.buffer_size, :]
        x_visible = x[:, self.buffer_size:, :]

        num_masked = ids_restore.shape[1] - x_visible.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1).to(x.dtype)

        x_full = torch.cat([x_visible, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]))
        x = torch.cat([x_buffer, x_full], dim=1)

        x = x + self.decoder_pos_emb

        for block in self.decoder_block:
            x = block(x, self.decoder_rope)

        z = self.decoder_norm(x)
        z = z[:, self.buffer_size:]

        return z

    def forward(self, x, mask_orders, labels, num_visible=None, force_unconditional=False):
        B, N, D = x.shape
        class_embedding = self.class_emb(labels)

        ids_restore = torch.argsort(mask_orders, dim=1)

        if self.training:
            mask = self.random_masking(x, mask_orders, self.min_mask_rate)
            num_masked = int(mask[0].sum().item())
            num_vis = N - num_masked
            x = self.forward_encoder(x, mask_orders, num_vis, class_embedding, force_unconditional=force_unconditional)
        else:
            mask = torch.zeros(B, N, device=x.device)
            mask.scatter_(1, mask_orders[:, num_visible:].long(), 1.0)
            x = self.forward_encoder(x, mask_orders, num_visible, class_embedding, force_unconditional=force_unconditional)

        z = self.forward_decoder(x, ids_restore)

        x_recon = self.reconstruction_head(z)

        return z, mask, x_recon

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
        source_params = list(self.parameters())
        for ema_params, decay in zip(self.ema_params_list, self.ema_decays):
            for targ, src in zip(ema_params, source_params):
                targ.detach().mul_(decay).add_(src, alpha=1 - decay)