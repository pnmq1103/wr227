"""
MMEngine Hooks for MARL-gated training.

FreezeRouterHook: Freezes/unfreezes the router at specific epochs
PPORouterHook: Runs PPO updates on the router after each training iteration
"""

import logging
import torch

try:
    from mmengine.hooks import Hook
    from mmdet.registry import HOOKS
    HAS_MMDET = True
except ImportError:
    # Standalone fallback
    class Hook:
        pass
    HAS_MMDET = False

from .reward import compute_ppo_reward, ppo_update

logger = logging.getLogger(__name__)


class FreezeRouterHookBase(Hook):
    """
    Controls the router freeze/unfreeze schedule.

    During the freeze phase (epochs 0 to unfreeze_epoch-1):
        - backbone.freeze_router = True (w = 1.0 for all tokens)
        - Router gradients are disabled
        - Detection heads converge on full-attention features

    After unfreeze_epoch:
        - backbone.freeze_router = False
        - Router participates in training via PPO
    """

    def __init__(self, unfreeze_epoch=4):
        self.unfreeze_epoch = unfreeze_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        # Handle DataParallel/DistributedDataParallel wrappers
        if hasattr(model, 'module'):
            model = model.module

        backbone = model.backbone

        if epoch < self.unfreeze_epoch:
            backbone.freeze_router = True
            # Disable router gradients during freeze phase
            for param in backbone.router.parameters():
                param.requires_grad = False
            logger.info(
                f"[FreezeRouterHook] Epoch {epoch}: Router FROZEN (w=1.0)")
        else:
            backbone.freeze_router = False
            for param in backbone.router.parameters():
                param.requires_grad = True
            logger.info(
                f"[FreezeRouterHook] Epoch {epoch}: Router ACTIVE (PPO)")


class PPORouterHookBase(Hook):
    """
    Runs PPO updates on the router after each training iteration.

    Only active when backbone.freeze_router is False.
    Extracts the detection loss from the runner's outputs and uses it
    as the terminal reward for the router's Actor-Critic.
    """

    def __init__(self,
                 lambda_budget=2.0,
                 lambda_sparse=1.0,
                 ppo_lr=1e-4,
                 clip_eps=0.2,
                 entropy_coeff=0.01,
                 ppo_epochs=1,
                 log_interval=50):
        self.lambda_budget = lambda_budget
        self.lambda_sparse = lambda_sparse
        self.ppo_lr = ppo_lr
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs
        self.log_interval = log_interval
        self._optimizer = None

    def _get_router_optimizer(self, runner):
        """Lazily create a separate optimizer for the router."""
        if self._optimizer is None:
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            router_params = list(model.backbone.router.parameters())
            self._optimizer = torch.optim.Adam(
                router_params, lr=self.ppo_lr)
        return self._optimizer

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Called after each training iteration.
        Extracts detection loss and runs PPO on the router.
        """
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        backbone = model.backbone

        # Skip if router is frozen
        if backbone.freeze_router:
            return

        # Get routing info from the backbone's last forward pass
        if not hasattr(backbone, '_routing_info'):
            return
        routing_info = backbone._routing_info

        # Skip if no log_probs (deterministic mode)
        if routing_info.get('log_probs') is None:
            return

        # Extract detection loss from runner outputs
        if outputs is None or 'loss' not in outputs:
            return
        det_loss = outputs['loss'].detach()

        # Compute PPO reward
        w_t = routing_info['w_t']
        reward, reward_components = compute_ppo_reward(
            det_loss, w_t,
            lambda_budget=self.lambda_budget,
            lambda_sparse=self.lambda_sparse,
        )

        # Run PPO update
        optimizer = self._get_router_optimizer(runner)
        ppo_stats = ppo_update(
            router=backbone.router,
            routing_info=routing_info,
            reward=reward,
            optimizer=optimizer,
            clip_eps=self.clip_eps,
            entropy_coeff=self.entropy_coeff,
            ppo_epochs=self.ppo_epochs,
        )

        # Log periodically
        if batch_idx % self.log_interval == 0:
            logger.info(
                f"[PPO] iter={batch_idx} | "
                f"task_r={reward_components['task_reward']:.4f} | "
                f"budget={reward_components['budget_cost']:.4f} | "
                f"sparse={reward_components['sparsity_pen']:.4f} | "
                f"w_mean={reward_components['w_mean']:.3f} | "
                f"w_std={reward_components['w_std']:.3f} | "
                f"ppo_loss={ppo_stats.get('policy_loss', 0):.4f}"
            )


# Register with MMDetection if available
if HAS_MMDET:
    @HOOKS.register_module()
    class FreezeRouterHook(FreezeRouterHookBase):
        pass

    @HOOKS.register_module()
    class PPORouterHook(PPORouterHookBase):
        pass
else:
    FreezeRouterHook = FreezeRouterHookBase
    PPORouterHook = PPORouterHookBase
