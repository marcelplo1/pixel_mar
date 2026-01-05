import math
import torch
import torch.nn as nn

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
            dropout=0.0, 
            num_classes=10, 
            buffer_size=32, 
            bottleneck_dim=16
        ):
        super().__init__()

        self.image_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size

        self.seq_len = (img_size // patch_size) ** 2
        self.embed_dim = channels * patch_size**2

        bottleneck_dim = hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(self.embed_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim)
        )

        #self.output_proj = nn.Linear(hidden_dim, self.embed_dim)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.ema_params = None

        self.input_ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes, self.hidden_dim)
        
        self.encoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, self.hidden_dim), requires_grad=True)
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, self.hidden_dim),  requires_grad=True)
        self.diffusion_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.hidden_dim),  requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=hidden_dim // 16,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=hidden_dim // 16,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

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

        # encoder_pos_emb = get_2d_sincos_pos_embed(self.encoder_pos_emb.shape[-1], int(self.seq_len ** 0.5))
        # self.encoder_pos_emb.data.copy_(torch.from_numpy(encoder_pos_emb).float().unsqueeze(0))

        # decoder_pos_emb = get_2d_sincos_pos_embed(self.decoder_pos_emb.shape[-1], int(self.seq_len ** 0.5))
        # self.decoder_pos_emb.data.copy_(torch.from_numpy(decoder_pos_emb).float().unsqueeze(0))

        # diffusion_pos_emb = get_2d_sincos_pos_embed(self.diffusion_pos_emb.shape[-1], int(self.seq_len ** 0.5))
        # self.diffusion_pos_emb.data.copy_(torch.from_numpy(diffusion_pos_emb).float().unsqueeze(0))

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)

    def forward(self, x, mask, labels):
        x = self.input_proj(x)
        bsz, seq_len, embed_dim = x.shape

        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        class_embedding = self.class_emb(labels)
        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_emb.to(x.device, dtype=x.dtype)
        x = self.input_ln(x)

        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, self.hidden_dim)

        encoded = self.encoder(x)
        encoded = self.encoder_norm(encoded)

        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1-mask_with_buffer).nonzero(as_tuple=True)] = encoded.reshape(encoded.shape[0] * encoded.shape[1], encoded.shape[2])

        x_after_pad = x_after_pad + self.decoder_pos_emb.to(x.device, dtype=x.dtype)
        decoded = self.decoder(x_after_pad)
        decoded = self.decoder_norm(decoded)
        decoded = decoded[:, self.buffer_size:]

        #decoded = self.output_proj(decoded)

        decoded = decoded + self.diffusion_pos_emb.to(x.device, dtype=x.dtype)

        return decoded
    
    @torch.no_grad()
    def update_ema(self):
        ema_decay = 0.9996
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)