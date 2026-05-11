"""
PPO Reward computation for MARL-gated ViT.

Design:
- reward = -detection_loss - lambda_budget * mean(w_t) - lambda_sparse * bimodal_sparsity(w_t)
- No fixed target budget.
- The model learns its own optimal retention rate by balancing task performance against compute cost.
- Bimodal sparsity sharpens the probabilities to be closer to 0 or 1.
"""

import torch
import torch.nn.functional as F


def compute_bimodal_sparsity(w_t):
    """
    Bimodal sparsity penalty: w * (1 - w)
    Equals 0 when w=0 or w=1, maximized at w=0.5.
    Pushes each individual weight toward binary values.
    """
    return (w_t * (1.0 - w_t)).mean()

def compute_ppo_reward(detection_loss, w_t, lambda_budget=2.0, lambda_sparse=1.0):
    """
    Compute the total PPO reward for the router Actor-Critic.

    reward = -detection_loss - lambda_budget * mean(w_t) - lambda_sparse * bimodal_sparsity(w_t)

    The task loss pulls retention UP (need tokens for accuracy).
    The budget penalty pushes retention DOWN (minimize compute).
    The bimodal sparsity sharpens decisions.

    Args:
        detection_loss: scalar, the combined detection loss from heads
        w_t: (B, L, N) routing weights from all layers
        lambda_budget: weight for direct budget minimization (default 2.0)
        lambda_sparse: weight for bimodal sparsity (default 1.0)

    Returns:
        reward: scalar
        reward_components: dict with individual terms for logging
    """
    task_reward = -detection_loss.detach()
    budget_cost = w_t.mean()
    sparsity_pen = compute_bimodal_sparsity(w_t)

    total_reward = task_reward - lambda_budget * budget_cost - lambda_sparse * sparsity_pen

    components = {
        'task_reward': task_reward.item(),
        'budget_cost': budget_cost.item(),
        'sparsity_pen': sparsity_pen.item(),
        'total_reward': total_reward.item(),
        'w_mean': w_t.mean().item(),
        'w_std': w_t.std().item(),
    }

    return total_reward, components


def ppo_update(router, routing_info, reward, optimizer,
               clip_eps=0.2, entropy_coeff=0.01, ppo_epochs=1,
               max_grad_norm=0.5):
    """
    Run PPO clipped surrogate update on the router parameters.
    """
    if routing_info['log_probs'] is None:
        return {'ppo_loss': 0.0, 'skipped': True}

    old_log_probs = routing_info['log_probs'].detach()  # (B, L, N)
    old_values = routing_info['values'].detach()         # (B, L, N)
    w_t = routing_info['w_t'].detach()                   # (B, L, N)
    old_mu = routing_info['mu'].detach()                 # (B, L, N)
    old_sigma = routing_info['sigma'].detach()           # (B, L, N)

    # Compute advantages
    advantage = reward - old_values.mean()
    # Normalize advantages
    if advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    for _ in range(ppo_epochs):
        z_old = torch.atanh((w_t * 2.0 - 1.0).clamp(-0.999, 0.999))

        from torch.distributions import Normal
        dist_new = Normal(old_mu, old_sigma)
        new_log_probs = dist_new.log_prob(z_old)
        w_raw = torch.tanh(z_old)
        jacobian = torch.log(0.5 * (1.0 - w_raw.pow(2)) + 1e-5)
        new_log_probs = new_log_probs - jacobian

        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = 0.5 * (reward - old_values).pow(2).mean()

        # Entropy bonus
        entropy = dist_new.entropy().mean()

        loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), max_grad_norm)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()

    return {
        'policy_loss': total_policy_loss / ppo_epochs,
        'value_loss': total_value_loss / ppo_epochs,
        'entropy': total_entropy / ppo_epochs,
    }
