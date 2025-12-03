import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

class MAE(nn.Module):
    def __init__(self, img_size, patch_size=16, hidden_dim=768, mlp_ratio=4.0, depth=12, proj_dropout=0.0):
        super().__init__()

        self.image_size = img_size
        self.seq_len = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = patch_size * patch_size
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.hidden_dim), requires_grad=True)
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.hidden_dim),  requires_grad=True)
        self.diffusion_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.embed_dim),  requires_grad=True)

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, mask):
        x = self.input_proj(x)
        bsz, seq_len, embed_dim = x.shape
        x = x + self.encoder_pos_emb.to(x.device, dtype=x.dtype)

        x = x[(1-mask).nonzero(as_tuple=True)].reshape(bsz, -1, self.hidden_dim)

        encoded = self.encoder(x)

        mask_tokens = self.mask_token.repeat(mask.shape[0], mask.shape[1], 1).to(x.dtype, x.device)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1-mask).nonzero(as_tuple=True)] = encoded.reshape(encoded.shape[0] * encoded.shape[1], encoded.shape[2])

        x_after_pad = x_after_pad + self.decoder_pos_emb.to(x.device, dtype=x.dtype)
        decoded = self.decoder(x_after_pad)
        decoded = self.output_proj(decoded)

        decoded = decoded + self.diffusion_pos_emb.to(x.device, dtype=x.dtype)

        return decoded
    

class DenoisingMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(2*input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        out = self.net(torch.cat((x, t_emb), dim=-1))
        return out