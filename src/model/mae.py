import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint

from utils.utils import get_2d_sincos_pos_embed
    

class MAE(nn.Module):
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
            grad_ckpt = False,
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
        self.grad_ckpt = grad_ckpt

        # self.encoder_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, bottleneck_dim),
        #     nn.SiLU(),
        #     nn.Linear(bottleneck_dim, hidden_dim)
        # )

        self.x_proj = nn.Linear(self.embed_dim, encoder_dim, bias=True)
        self.x_ln = nn.LayerNorm(encoder_dim, eps=1e-6)
        self.decoder_embed = nn.Linear(self.decoder_dim, decoder_dim, bias=True)

        self.mask_token  = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.class_emb = nn.Embedding(num_classes, encoder_dim)
        
        self.encoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_dim), requires_grad=True)
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_dim),  requires_grad=True)

        self.encoder_block = nn.ModuleList([
            Block(encoder_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
                  proj_drop=dropout, attn_drop=dropout) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_dim, eps=1e-6)

        self.decoder_block = nn.ModuleList([
            Block(decoder_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
                  proj_drop=dropout, attn_drop=dropout) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        #self.label_drop_prob = 0.1
        #self.fake_latent = nn.Parameter(torch.zeros(1, hidden_dim))

        self.ema_decay=ema_decay
        self.ema_params = None

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier Uniform is standard for Transformers (often called Glorot)
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

    def forward_encoder(self, x, mask, class_emb):
        x = self.x_proj(x)
        bsz, seq_len, embed_dim = x.shape

        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        if self.training and False: #TODO apply label dropping together with denoiser
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_emb
        else:
            class_embedding = class_emb

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_emb
        x = self.x_ln(x)

        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, self.encoder_dim)

        if self.grad_ckpt and not torch.jit.is_scripting():
            for block in self.encoder_block:
                x = checkpoint(block, x)
        else:
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

        if self.grad_ckpt and not torch.jit.is_scripting():
            for block in self.decoder_block:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_block:
                x = block(x)

        decoded = self.decoder_norm(x)
        decoded = decoded[:, self.buffer_size:]

        return decoded
    
    def forward(self, x, mask, labels):
        class_embedding = self.class_emb(labels)

        x = self.forward_encoder(x, mask, class_embedding)
        z = self.forward_decoder(x, mask)

        return z
    
    @torch.no_grad()
    def update_ema(self):
        ema_decay = self.ema_decay
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)