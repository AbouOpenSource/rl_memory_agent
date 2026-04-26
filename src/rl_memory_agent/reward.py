from __future__ import annotations

import math
from dataclasses import dataclass


REWARD_MODES = (
    "neg_step_time",
    "samples_per_second",
    "tokens_per_second",
    "optimizer_updates_per_second",
    "loss_delta_per_second",
)


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "neg_step_time"
    sequence_length: int = 1
    loss_delta_per_update: float = 1.0
    log_progress: bool = True
    scale: float = 1.0
    oom_penalty: float = 5.0
    restart_penalty: float = 0.0


def _positive_metric(value: float, *, log_progress: bool, scale: float) -> float:
    scaled = max(0.0, float(value)) / max(1e-12, float(scale))
    return float(math.log1p(scaled)) if log_progress else scaled


def compute_reward(
    *,
    config: RewardConfig,
    elapsed_s: float,
    world_size: int,
    micro_batch: int,
    grad_accum_steps: int,
    oom: bool = False,
    restart: bool = False,
) -> float:
    """Compute the scalar RL reward from useful progress and elapsed time."""

    if config.mode not in REWARD_MODES:
        raise ValueError(f"unsupported reward mode: {config.mode!r}")

    elapsed = max(1e-9, float(elapsed_s))
    effective_samples = (
        max(1.0, float(world_size))
        * max(1.0, float(micro_batch))
        * max(1.0, float(grad_accum_steps))
    )

    if config.mode == "neg_step_time":
        reward = -elapsed / max(1e-12, float(config.scale))
    elif config.mode == "samples_per_second":
        reward = _positive_metric(
            effective_samples / elapsed,
            log_progress=config.log_progress,
            scale=config.scale,
        )
    elif config.mode == "tokens_per_second":
        tokens = effective_samples * max(1.0, float(config.sequence_length))
        reward = _positive_metric(tokens / elapsed, log_progress=config.log_progress, scale=config.scale)
    elif config.mode == "optimizer_updates_per_second":
        reward = _positive_metric(1.0 / elapsed, log_progress=config.log_progress, scale=config.scale)
    else:
        reward = _positive_metric(
            max(0.0, float(config.loss_delta_per_update)) / elapsed,
            log_progress=config.log_progress,
            scale=config.scale,
        )

    if oom:
        reward -= float(config.oom_penalty)
    if restart:
        reward -= float(config.restart_penalty)
    return float(reward)
