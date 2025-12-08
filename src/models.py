import torch
import torch.nn as nn

class MAE(nn.Module):
    def __init__(self, img_size, patch_size=16, hidden_dim=768, channels=3, mlp_ratio=4.0, depth=12, proj_dropout=0.0):
        super().__init__()

        self.image_size = img_size
        self.seq_len = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = channels * patch_size**2
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encder_pos_emb = get_2d_sincos_pos_embed(hidden_dim, img_size // patch_size, img_size // patch_size)
        self.encoder_pos_emb = nn.Parameter(encder_pos_emb, requires_grad=True)
        decoder_pos_emb = get_2d_sincos_pos_embed(hidden_dim, img_size // patch_size, img_size // patch_size)
        self.decoder_pos_emb = nn.Parameter(decoder_pos_emb,  requires_grad=True)
        diffusion_pos_emb = get_2d_sincos_pos_embed(self.embed_dim, img_size // patch_size, img_size // patch_size)
        self.diffusion_pos_emb = nn.Parameter(diffusion_pos_emb,  requires_grad=True)

        # self.encoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.hidden_dim), requires_grad=True)
        # self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.hidden_dim),  requires_grad=True)
        # self.diffusion_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.embed_dim),  requires_grad=True)

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

        mask_tokens = self.mask_token.repeat(mask.shape[0], mask.shape[1], 1).to(x.dtype)
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
    
    
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")  # 2D grid

    grid = torch.stack(grid, dim=0).reshape(2, -1)  # [2, H*W]

    assert embed_dim % 2 == 0
    half_dim = embed_dim // 2

    emb_h = get_1d_sincos_pos_embed(half_dim, grid[0])
    emb_w = get_1d_sincos_pos_embed(half_dim, grid[1])

    pos_emb = torch.cat([emb_h, emb_w], dim=1)

    return pos_emb.unsqueeze(0)  # [1, H*W, D]


def get_1d_sincos_pos_embed(embed_dim, positions):
    assert embed_dim % 2 == 0
    half = embed_dim // 2

    frequencies = torch.arange(half, dtype=torch.float32)
    frequencies = 1.0 / (10000 ** (frequencies / half))

    angles = positions[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)