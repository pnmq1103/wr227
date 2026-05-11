"""
MARL-Gated RALA ViT modules for MMDetection.

This package provides:
- MARLRALABackbone: A flat ViT backbone with FPN projection layers
- FreezeRouterHook: Controls router freeze/unfreeze schedule
- PPORouterHook: Runs PPO updates on the router
"""

from .attention import (
    SharedActorCritic,
    ChunkwiseRALAAttention,
    EncoderBlock,
    PatchEmbedding,
    MLP,
    ConditionalPositionEncoding,
    DropPath,
)
from .backbone import MARLRALABackbone
from .reward import (
    compute_ppo_reward,
    compute_entropy_redundancy_penalty,
    compute_bimodal_sparsity,
    compute_variance_bonus,
    ppo_update,
)
from .hooks import FreezeRouterHook, PPORouterHook

__all__ = [
    'SharedActorCritic',
    'ChunkwiseRALAAttention',
    'EncoderBlock',
    'PatchEmbedding',
    'MLP',
    'ConditionalPositionEncoding',
    'DropPath',
    'MARLRALABackbone',
    'compute_ppo_reward',
    'compute_entropy_redundancy_penalty',
    'compute_bimodal_sparsity',
    'compute_variance_bonus',
    'ppo_update',
    'FreezeRouterHook',
    'PPORouterHook',
]
