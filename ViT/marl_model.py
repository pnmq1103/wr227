import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision import transforms
from datasets import load_dataset
from einops import rearrange
from tqdm import tqdm
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
class SharedActorCritic(nn.Module):
    """
    Shared Actor-Critic for the swarm of N agents.
    
    - Shared across ALL layers (parameter sharing as described in the paper).
    - Each token is an independent agent; the AC processes them all in parallel.
    - The actor outputs a squashed Gaussian scalar weight w ∈ (0,1) per token.
    - The critic outputs a per-token value estimate.
    """
    def __init__(self, d_model: int):
        super().__init__()
        half = max(d_model // 2, 16)

        # Shared projection: s_{n,l} = Proj(x_{n,l})
        self.proj = nn.Linear(d_model, half)

        # Actor: produces μ and σ for the squashed Gaussian
        self.actor_mlp = nn.Sequential(nn.Linear(half, half), nn.ELU())
        self.mu_head = nn.Linear(half, 1)
        self.sigma_head = nn.Linear(half, 1)

        # Critic: per-token value estimate V_φ(s_{n,l})
        self.critic_mlp = nn.Sequential(
            nn.Linear(half, half), nn.ELU(), nn.Linear(half, 1)
        )

    def forward(self, local_features: torch.Tensor, deterministic: bool = False):
        """
        Args:
            local_features: (B, N, d_model) — token features at layer l
            deterministic: if True, use mean (no sampling)
        Returns:
            w:        (B, N)   — gating weights in (0, 1)
            log_prob: (B, N)   — Jacobian-corrected log probability (None if deterministic)
            value:    (B, N)   — per-token value estimate
            mu:       (B, N)   — actor mean (for entropy computation)
            sigma:    (B, N)   — actor std (for entropy computation)
        """
        # State observation for each agent
        s = self.proj(local_features)                       # (B, N, half)

        # Actor
        h_actor = self.actor_mlp(s)                         # (B, N, half)
        mu = self.mu_head(h_actor).squeeze(-1)              # (B, N)
        sigma = F.softplus(self.sigma_head(h_actor)).squeeze(-1) + 1e-5  # (B, N)

        # Critic — per-token value
        value = self.critic_mlp(s).squeeze(-1)              # (B, N)

        if deterministic:
            z = mu
            log_prob = None
        else:
            dist = Normal(mu, sigma)
            z = dist.rsample()                              # reparameterization trick
            log_prob = dist.log_prob(z)                     # (B, N)

        # Squash to (0, 1) via tanh
        # Apply temperature scaling to force sharpness and break the 0.5 saddle-point trap
        w_raw = torch.tanh(z * 5.0)
        w = (w_raw + 1.0) / 2.0                            # (B, N)

        # Jacobian correction for the log probability
        if not deterministic:
            # log|det(dw/dz)| = log((1 - tanh²(z))/2 + ε)
            # log|det(dw/dz)| with temperature factor
            jacobian = torch.log(5.0 * 0.5 * (1.0 - w_raw.pow(2)) + 1e-5)
            log_prob = log_prob - jacobian                  # (B, N)

        return w, log_prob, value, mu, sigma


class ChunkwiseRALAAttention(nn.Module):
    """
    Chunkwise Linear Attention with MARL gating.
    
    Based on RALA but replaces α_j reweighting with MARL agent weights:
    - κ(·) = ELU(·) + 1.0 kernel function
    - MARL gating: w_t from shared Actor-Critic replaces RALA's α_j
    - Output modulation: Y_i = φ(X_i) ⊙ (κ(Q_i) · S_t) — Hadamard product gate
    
    The router is NOT internal — weights are passed from outside.
    """
    def __init__(self, d_model: int, head: int = 8, chunk_size: int = 16,
                 gamma: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.chunk_size = chunk_size
        self.gamma = gamma
        self.d_k = d_model // head

        # QKV projection
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output gate: φ(X) for Hadamard product modulation
        self.w_o_gate = nn.Linear(d_model, d_model)

        # Output projection
        self.w_o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, w_gating, use_dilution=False):
        """
        Args:
            x:            (B, N, d_model)
            w_gating:     (B, N) — per-token gating weights from the shared AC
            use_dilution: bool — enable state dilution Γ_τ (Phase 2/3)
        Returns:
            out:  (B, N, d_model) — attention output
            phi_k: for Fisher info computation (gradients flow through here)
        """
        b, n, d = x.shape
        T = n // self.chunk_size
        C = self.chunk_size

        # --- Linear projections ---
        q = rearrange(self.w_q(x), 'b (T C) (h dk) -> b T h C dk',
                       T=T, C=C, h=self.head)
        k = rearrange(self.w_k(x), 'b (T C) (h dk) -> b T h C dk',
                       T=T, C=C, h=self.head)
        v = rearrange(self.w_v(x), 'b (T C) (h dk) -> b T h C dk',
                       T=T, C=C, h=self.head)

        # Scaling
        q = q * (self.d_k ** -0.25)
        k = k * (self.d_k ** -0.25)

        # --- Kernel function: κ(·) = ELU(·) + 1 ---
        phi_q = F.elu(q) + 1.0  # (B, T, h, C, dk)
        phi_k = F.elu(k) + 1.0  # (B, T, h, C, dk)

        # --- Apply MARL gating directly to φ(K) ---
        # MARL agents replace RALA's α_j: each agent's weight gates its token's key
        # w_gating: (B, N) -> reshape to (B, T, C) -> expand to (B, T, 1, C, 1)
        w_chunks = rearrange(w_gating, 'b (T C) -> b T C', T=T, C=C)
        w_expanded = w_chunks.unsqueeze(2).unsqueeze(-1)  # (B, T, 1, C, 1)
        k_gated = w_expanded * phi_k                      # (B, T, h, C, dk)

        # --- Chunk-level KV computation ---
        k_gated_f32 = k_gated.to(torch.float32)
        v_f32 = v.to(torch.float32)
        KV_chunks = torch.matmul(k_gated_f32.transpose(-2, -1), v_f32)  # (B,T,h,dk,dk)
        Z_chunks = k_gated_f32.sum(dim=-2)                               # (B,T,h,dk)

        # --- Chunk-level average gating weight w̄_t ---
        w_bar = w_chunks.mean(dim=-1)  # (B, T)

        # --- Sequential recurrence ---
        outputs = []
        S = torch.zeros(b, self.head, self.d_k, self.d_k,
                        device=x.device, dtype=torch.float32)
        Z = torch.zeros(b, self.head, self.d_k,
                        device=x.device, dtype=torch.float32)

        for t in range(T):
            # Decay: 1 - γ(1 - w̄_t)
            decay_factor = 1.0 - (self.gamma * (1.0 - w_bar[:, t]))
            decay_S = decay_factor.view(b, 1, 1, 1)
            decay_Z = decay_factor.view(b, 1, 1)

            # State dilution term Γ_τ (Eq. 11 from approach.tex)
            if use_dilution and t > 0:
                dilution_scale = self.gamma * (1.0 - w_bar[:, t])  # (B,)
                gamma_tau = dilution_scale.view(b, 1, 1, 1) * S / max(t, 1)
                S = (S * decay_S) + KV_chunks[:, t] + gamma_tau
            else:
                S = (S * decay_S) + KV_chunks[:, t]

            Z = (Z * decay_Z) + Z_chunks[:, t]

            # Attention output for chunk t
            phi_q_t = phi_q[:, t].to(torch.float32)          # (B, h, C, dk)
            nom = torch.matmul(phi_q_t, S)                   # (B, h, C, dk)
            denom = (phi_q_t * Z.unsqueeze(-2)).sum(dim=-1, keepdim=True) + 1e-5

            out_t = nom / denom

            if self.training:
                out_t = self.dropout(out_t)

            out_t = torch.clamp(out_t, min=-65000.0, max=65000.0)
            outputs.append(out_t.to(q.dtype))

        out = torch.stack(outputs, dim=1)                    # (B, T, h, C, dk)
        out = rearrange(out, 'b T h C dk -> b (T C) (h dk)')

        # --- Output modulation: Y = φ(X) ⊙ attn_output ---
        gate = torch.sigmoid(self.w_o_gate(x))                # (B, N, d)
        out = out * gate

        out = self.w_o_proj(out)

        return out, phi_k


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=144, patch_size=12, in_chans=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

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


class EncoderBlock(nn.Module):
    """
    A single encoder block. The attention module does NOT own a router.
    The router (SharedActorCritic) lives at the ViT level.
    """
    def __init__(self, d_model, head, chunk_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ChunkwiseRALAAttention(d_model, head=head, chunk_size=chunk_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4)

    def forward(self, x, w_gating, use_dilution=False):
        res = x
        x_normed = self.norm1(x)
        out, phi_k = self.attn(x_normed, w_gating, use_dilution)
        x = res + out
        x = x + self.mlp(self.norm2(x))
        return x, phi_k


class ViT(nn.Module):
    """
    MARL-Gated RALA ViT.
    
    Key difference from v1: a SINGLE SharedActorCritic module is shared
    across all layers. At each layer, every token acts as an independent
    agent making its own gating decision via this shared policy.
    """
    def __init__(self, image_size=144, patch_size=12, num_classes=100,
                 d_model=256, depth=6, head=8, chunk_size=16):
        super().__init__()
        self.depth = depth
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, d_model))
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, head, chunk_size) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # Single shared Actor-Critic for ALL layers (swarm with shared params)
        self.router = SharedActorCritic(d_model)

    def forward(self, x, deterministic=False, phase1=False, use_dilution=False):
        x = self.patch_embed(x) + self.pos_embed  # (B, N, d)

        w_list, log_prob_list, value_list, mu_list, sigma_list, kt_list = \
            [], [], [], [], [], []

        for block in self.blocks:
            # Each layer: the shared AC evaluates all N tokens (N agents)
            w, log_prob, value, mu, sigma = self.router(x, deterministic)

            if phase1:
                w_gating = torch.ones_like(w)
            else:
                w_gating = w

            x, phi_k = block(x, w_gating, use_dilution)

            w_list.append(w)
            log_prob_list.append(log_prob)
            value_list.append(value)
            mu_list.append(mu)
            sigma_list.append(sigma)
            kt_list.append(phi_k)

        x = self.norm(x)
        logits = self.head(x.mean(dim=1))  # (B, num_classes)

        has_log_prob = log_prob_list[0] is not None

        return {
            'logits': logits,
            'w_t': torch.stack(w_list, dim=1),           # (B, L, N)
            'log_probs': torch.stack(log_prob_list, dim=1) if has_log_prob else None,
            'values': torch.stack(value_list, dim=1),     # (B, L, N)
            'mu': torch.stack(mu_list, dim=1),            # (B, L, N)
            'sigma': torch.stack(sigma_list, dim=1),      # (B, L, N)
            'k_t': kt_list                                # list of (B,T,h,C,dk)
        }
import torch
