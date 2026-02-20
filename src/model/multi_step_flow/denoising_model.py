import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.model_utils import Attention, RMSNorm, SwiGLUFFN, TimestepEmbedder, VisionRotaryEmbeddingFast

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

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
        #return x + gate_mlp * h
        return gate_mlp * h
    
class FinalLayer(nn.Module):
    """
    The final layer with a possible bottleneck layer
    """
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
 
    @torch.compile
    def forward(self, x, c_token):
        shift, scale = self.adaLN_modulation(c_token).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

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

    @torch.compile
    def forward(self, x, feat_rope=None):     
        x = self.attn_ln(x)

        x = x + self.attn(x, rope=feat_rope)
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
        grad_ckpt = False,
        denoiser_type = 'in_context'
    ):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_heads = hidden_dim // 64
        self.img_size = img_size
        self.grad_ckpt = grad_ckpt
        self.denoiser_type = denoiser_type

        self.embedding_dim = channels * patch_size**2
        self.num_patches = (img_size // patch_size) ** 2

        self.x_proj = nn.Linear(self.embedding_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        #self.y_embedder = nn.Embedding(num_classes + 1, hidden_dim)
        self.z_proj = nn.Linear(z_hidden_dim, hidden_dim)

        # rope
        half_head_dim = self.hidden_dim // self.num_heads  // 2
        hw_seq_len = img_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )

        if denoiser_type == 'in_context':
            self.blocks = nn.ModuleList([
                InContextBlock(hidden_dim, hidden_dim // 64)
                for i in range(depth)
            ])
            self.final_layer = nn.Linear(hidden_dim, self.embedding_dim, bias=True)
            self.initialize_weights_in_context()
        elif denoiser_type == 'ada_ln':
            self.blocks = nn.ModuleList([
                ResBlock(hidden_dim)
                for i in range(depth)
            ])
            self.final_layer = FinalLayer(hidden_dim, patch_size, channels)
            self.initialize_weights_ada_ln_wo_res()

    def initialize_weights_in_context(self):
        # Basic Xavier initialization for all Linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Specific normal initialization for projections and embedders
        nn.init.normal_(self.x_proj.weight, std=0.02)
        #nn.init.normal_(self.y_embedder.weight, std=0.02)
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

    def initialize_weights_ada_ln(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def initialize_weights_ada_ln_wo_res(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            # We use Xavier/Kaiming so the gates (gate_mlp) start with non-zero values.
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Initialize MLP weights using Kaiming Normal for SiLU activation
            for m in block.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward_in_context(self, x, z, t, y):
        """
        x: (B*N, D)
        z: (B*N, D')
        y: (B*N, Cls_num)
        t: (B*N, 1)
        """
        x = self.x_proj(x)  
        t = self.t_embedder(t)      
        z = self.z_proj(z)
        #y = self.y_embedder(y)

        #c = t + z + y
        c = t + z

        c = c.unsqueeze(1)
        x = x.unsqueeze(1)
        x = torch.cat([x, c], dim=1)

        if self.grad_ckpt and not torch.jit.is_scripting() and False:
            for block in self.blocks:
                x = checkpoint(block, x)
        else:
            for block in self.blocks:
                x = block(x, self.feat_rope)

        x = x[:, 0, :]
        x = self.final_layer(x)

        return x
    
    def forward_ada_ln(self, x, z, t, y):
        """
        x: (B*N, D)
        z: (B*N, D')
        y: (B*N, Cls_num)
        t: (B*N, 1)
        """
        x = self.x_proj(x)  
        t = self.t_embedder(t)      
        z = self.z_proj(z)
        #y = self.y_embedder(y)

        #c = t + z + y
        c = t + z

        if self.grad_ckpt and not torch.jit.is_scripting() and False:
            for block in self.blocks:
                x = checkpoint(block, x, c)
        else:
            for block in self.blocks:
                x = block(x, c)

        x = self.final_layer(x, c)

        return x
    
    def forward(self, x, z, t, y):
        if self.denoiser_type == 'in_context':
            x = self.forward_in_context(x, z, t, y)
        elif self.denoiser_type == 'ada_ln':
            x = self.forward_ada_ln(x, z, t, y)
        else:
            raise NotImplementedError
        return x



