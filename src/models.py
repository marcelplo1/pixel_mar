import math
import torch
import torch.nn as nn

class MAE(nn.Module):
    def __init__(self, img_size, patch_size=16, hidden_dim=768, 
                 channels=3, mlp_ratio=4.0, depth=2, proj_dropout=0.0):
        super().__init__()

        self.image_size = img_size
        self.seq_len = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = channels * patch_size**2
        self.hidden_dim = hidden_dim

        self.input_proj  = nn.Linear(self.embed_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.embed_dim)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Positional embeddings
        self.encoder_pos_emb   = nn.Parameter(get_2d_sincos_pos_embed(hidden_dim, img_size // patch_size, img_size // patch_size), requires_grad=True)
        self.decoder_pos_emb   = nn.Parameter(get_2d_sincos_pos_embed(hidden_dim, img_size // patch_size, img_size // patch_size), requires_grad=True)
        self.diffusion_pos_emb = nn.Parameter(get_2d_sincos_pos_embed(self.embed_dim, img_size // patch_size, img_size // patch_size), requires_grad=True)
        
        # self.encoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.hidden_dim), requires_grad=True)
        # self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.hidden_dim),  requires_grad=True)
        # self.diffusion_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len , self.embed_dim),  requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=hidden_dim // 16,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=proj_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=hidden_dim // 16,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=proj_dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)

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
    
class Embedder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedder = nn.Embedding(input_size, hidden_size)

    def forward(self, x):
        embeddings = self.embedder(x)
        return embeddings
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DenoisingMLP(nn.Module):
    def __init__(self, input_dim, output_dim, number_classes, seq_len, batch_size, hidden_dim=512):
        super().__init__()
        self.t_embedder = TimestepEmbedder(input_dim)
        self.y_embedder = Embedder(number_classes, input_dim)

        self.net = nn.Sequential(
            nn.Linear(3*input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, z, t, y):
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        c = c.unsqueeze(1).expand(-1, x.shape[1], -1)

        out = self.net(torch.cat((x, c, z), dim=-1))
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