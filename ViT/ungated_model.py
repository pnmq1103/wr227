import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=144, patch_size=12, in_chans=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class ChunkwiseRALAAttention(nn.Module):
    def __init__(self, d_model: int, head: int = 8, chunk_size: int = 16, gamma: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.chunk_size = chunk_size
        self.gamma = gamma
        self.d_k = d_model // head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, w_gating=None, use_dilution=False, use_gating=False):
        b, n, d = x.shape
        T = n // self.chunk_size
        C = self.chunk_size

        q = rearrange(self.w_q(x), 'b (T C) (h dk) -> b T h C dk', T=T, C=C, h=self.head) * (self.d_k ** -0.25)
        k = rearrange(self.w_k(x), 'b (T C) (h dk) -> b T h C dk', T=T, C=C, h=self.head) * (self.d_k ** -0.25)
        v = rearrange(self.w_v(x), 'b (T C) (h dk) -> b T h C dk', T=T, C=C, h=self.head)

        phi_q = F.elu(q) + 1.0 
        phi_k = F.elu(k) + 1.0 

        # Non-gated: all tokens active (w_bar = 1)
        k_f32 = phi_k.to(torch.float32)
        v_f32 = v.to(torch.float32)
        KV_chunks = torch.matmul(k_f32.transpose(-2, -1), v_f32)  
        Z_chunks = k_f32.sum(dim=-2)  

        outputs = []
        S = torch.zeros(b, self.head, self.d_k, self.d_k, device=x.device, dtype=torch.float32)
        Z = torch.zeros(b, self.head, self.d_k, device=x.device, dtype=torch.float32)

        for t in range(T):
            # Non-gated: No decay, no dilution scale needed (or w_bar=1, so decay=1)
            S = S + KV_chunks[:, t]
            Z = Z + Z_chunks[:, t]

            phi_q_t = phi_q[:, t].to(torch.float32)          
            nom = torch.matmul(phi_q_t, S)                   
            denom = (phi_q_t * Z.unsqueeze(-2)).sum(dim=-1, keepdim=True) + 1e-5

            out_t = nom / denom
            if self.training: out_t = self.dropout(out_t)
            outputs.append(out_t.to(q.dtype))

        out = torch.stack(outputs, dim=1)                    
        out = rearrange(out, 'b T h C dk -> b (T C) (h dk)')
        out = self.w_o_proj(out)
        return out, phi_k

class EncoderBlock(nn.Module):
    def __init__(self, d_model, head, chunk_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ChunkwiseRALAAttention(d_model, head=head, chunk_size=chunk_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4)

    def forward(self, x, w_gating=None, use_dilution=False, use_gating=False):
        res = x
        x_normed = self.norm1(x)
        out, phi_k = self.attn(x_normed, w_gating, use_dilution, use_gating)
        x = res + out
        x = x + self.mlp(self.norm2(x))
        return x, phi_k

class ViT(nn.Module):
    def __init__(self, image_size=144, patch_size=12, num_classes=100,
                 d_model=256, depth=16, head=8, chunk_size=16):
        super().__init__()
        self.depth = depth
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, d_model))
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, head, chunk_size) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, use_dilution=False, deterministic=False, use_gating=False):
        x = self.patch_embed(x) + self.pos_embed  

        for block in self.blocks:
            x, _ = block(x, use_dilution=use_dilution, use_gating=use_gating)

        x = self.norm(x)
        logits = self.head(x.mean(dim=1))  

        return {
            'logits': logits,
            'w_t': None # Non-gated has no routing weights
        }
