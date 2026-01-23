import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from utils.utils import get_2d_sincos_pos_embed
    

class MAE(nn.Module):
    def __init__(
            self, 
            img_size, 
            patch_size=16, 
            channels=3, 
            hidden_dim=768, 
            depth=6, 
            mlp_ratio=4.0, 
            dropout=0.1, 
            num_classes=10, 
            buffer_size=32, 
            bottleneck_dim=16,
            ema_decay=0.9999
        ):
        super().__init__()

        self.image_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.num_classes = num_classes

        self.seq_len = (img_size // patch_size) ** 2
        self.embed_dim = channels * patch_size**2

        # self.encoder_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, bottleneck_dim),
        #     nn.SiLU(),
        #     nn.Linear(bottleneck_dim, hidden_dim)
        # )

        self.x_proj = nn.Linear(self.embed_dim, hidden_dim, bias=True)
        self.x_ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.decoder_embed = nn.Linear(self.hidden_dim, hidden_dim, bias=True)

        self.mask_token  = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.class_emb = nn.Embedding(num_classes, self.hidden_dim)
        
        self.encoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, self.hidden_dim), requires_grad=True)
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, self.hidden_dim),  requires_grad=True)

        self.encoder_block = nn.ModuleList([
            Block(hidden_dim, hidden_dim//64, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
                  proj_drop=dropout, attn_drop=dropout) for _ in range(depth)])
        self.encoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.decoder_block = nn.ModuleList([
            Block(hidden_dim, hidden_dim//64, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
                  proj_drop=dropout, attn_drop=dropout) for _ in range(depth)])
        self.decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

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
        pos_embed_grid = get_2d_sincos_pos_embed(self.hidden_dim, grid_size)
        full_pos_embed = torch.zeros(self.seq_len + self.buffer_size, self.hidden_dim)
        full_pos_embed[self.buffer_size:, :] = torch.from_numpy(pos_embed_grid).float()
        
        self.encoder_pos_emb.data.copy_(full_pos_embed.unsqueeze(0))
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

        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, self.hidden_dim)

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