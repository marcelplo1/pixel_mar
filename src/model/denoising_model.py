import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils.utils import RMSNorm, get_2d_sincos_pos_embed


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    """
    The final layer with a possible bottleneck layer
    """
    def __init__(self, hidden_dim, patch_size, out_channels, bottleneck_dim=8):
        super().__init__()
        self.norm_final = RMSNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        bottleneck_dim = hidden_dim
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 2 * hidden_dim, bias=True)
        )

    @torch.compile
    def forward(self, x, c_token):
        shift, scale = self.adaLN_modulation(c_token).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.in_ln = RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        )

    @torch.compile
    def forward(self, x, c_token):        
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_token).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class DenoisingMLP(nn.Module):
    """
    The diffusion model
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_dim=768,
        mae_hidden_dim=768,
        depth=6,
        num_heads=16,
        num_classes=10,
        bottleneck_dim=64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.num_classes = num_classes

        self.embedding_dim = in_channels * patch_size**2
        self.num_patches = (input_size // patch_size) ** 2

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))  #TODO

        self.input_proj = nn.Linear(self.embedding_dim, hidden_dim)
        self.c_token_proj = nn.Linear(mae_hidden_dim, hidden_dim, bias=True)

        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, self.embedding_dim)
            for i in range(depth)
        ])

        # linear predict with bottleneck layer
        self.final_layer = FinalLayer(hidden_dim, patch_size, self.out_channels, bottleneck_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Init pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Init of time_embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, z, t):
        """
        x: (B*N, D)
        z: (B*N, D)
        t: (B*N,)
        """
        x = self.input_proj(x)  
        t = self.t_embedder(t)      
        z = self.c_token_proj(z)

        c = t + z

        for i, block in enumerate(self.blocks):
            x = block(x, c)

        x = self.final_layer(x, c)

        return x
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations
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

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into a vector representations
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

def denoisingMLP_S_04(**kwargs):
    return DenoisingMLP(depth=6, hidden_dim=256, num_heads=6, patch_size=4, **kwargs)

def denoisingMLP_B_16(**kwargs):
    return DenoisingMLP(depth=12, hidden_dim=768, num_heads=12, patch_size=16, **kwargs)

denoising_models = {
    'denoisingMLP-S/04' : denoisingMLP_S_04,
    'denoisingMLP-B/16': denoisingMLP_B_16,
}


