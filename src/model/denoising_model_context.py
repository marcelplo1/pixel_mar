import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from utils.utils import RMSNorm

class InContextBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_ratio = 4.0, 
        attn_drop=0.0,
        proj_drop=0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn_ln = RMSNorm(hidden_dim, eps=1e-6)
        self.attn = Attention(hidden_dim, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.mlp_ln = RMSNorm(hidden_dim, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_dim, int(hidden_dim * mlp_ratio), drop=proj_drop)

    #@torch.compile
    def forward(self, x):     
        x = self.attn_ln(x)

        x = x + self.attn(x)
        x = x + self.mlp(self.mlp_ln(x))

        return x

class DenoisingModel(nn.Module):
    """
    The diffusion model
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        channels=3,
        num_classes=1000,
        hidden_dim=1024,
        depth=6,
        dropout=0.0,
        z_hidden_dim=768,
    ):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.img_size = img_size

        self.embedding_dim = channels * patch_size**2
        self.num_patches = (img_size // patch_size) ** 2

        self.x_proj = nn.Linear(self.embedding_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_dim)
        self.z_proj = nn.Linear(z_hidden_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            InContextBlock(hidden_dim, hidden_dim // 64)
            for i in range(depth)
        ])

        self.final_layer = nn.Linear(hidden_dim, self.embedding_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Basic Xavier initialization for all Linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Specific normal initialization for projections and embedders
        nn.init.normal_(self.x_proj.weight, std=0.02)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.normal_(self.z_proj.weight, std=0.02)

        # Timestep MLP initialization (standard in diffusion models)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            # For Attention: Zero the output projection
            nn.init.constant_(block.attn.proj.weight, 0)
            if block.attn.proj.bias is not None:
                nn.init.constant_(block.attn.proj.bias, 0)
            
            # For SwiGLU: Zero the w3 layer (the output layer)
            nn.init.constant_(block.mlp.w3.weight, 0)
            if block.mlp.w3.bias is not None:
                nn.init.constant_(block.mlp.w3.bias, 0)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x, z, t, y):
        """
        x: (B*N, D)
        z: (B*N, D')
        y: (B*N, Cls_num)
        t: (B*N, 1)
        """
        x = self.x_proj(x)  
        t = self.t_embedder(t)      
        z = self.z_proj(z)
        y = self.y_embedder(y)

        c = t + z + y

        c = c.unsqueeze(1)
        x = x.unsqueeze(1)
        x = torch.cat([x, c], dim=1)

        for i, block in enumerate(self.blocks):
            x = block(x)

        x = x[:, 0, :]
        x = self.final_layer(x)

        return x
    

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))
    
def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
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
    


