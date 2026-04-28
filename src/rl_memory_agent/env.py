from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from rl_memory_agent.knobs import KnobActionSpace, KnobConfig, KnobConstraints
from rl_memory_agent.reward import RewardConfig, compute_reward
from rl_memory_agent.safety import SafetyShield
from rl_memory_agent.state import StateBuilder
from rl_memory_agent.telemetry import TelemetrySample, TelemetryWindow


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    cost: float
    done: bool
    info: Dict[str, Any]


@dataclass(frozen=True)
class ToyEnvConfig:
    seed: int = 0
    world_size: int = 4
    budget_mb: float = 16_000.0
    base_model_state_mb: float = 7_000.0
    activation_mb_per_sample_fp32: float = 180.0
    transient_mb: float = 1_000.0
    base_compute_time_s: float = 0.03
    compute_time_per_sample_s: float = 0.003
    comm_time_s: float = 0.01
    io_bandwidth_mb_s: float = 600.0
    ckpt_size_mb: float = 4_000.0
    oom_penalty: float = 5.0
    restart_penalty: float = 0.0
    noise_std_mb: float = 50.0
    reward_mode: str = "neg_step_time"
    sequence_length: int = 1
    loss_delta_per_update: float = 1.0
    reward_log_progress: bool = True
    reward_scale: float = 1.0


class ToyMemoryEnv:
    """
    A minimal environment that simulates memory usage and step time as a function of knobs.

    This is *not* a model of any specific framework; it is a convenient harness to develop and test
    the RL control stack without requiring a full distributed setup.
    """

    def __init__(
        self,
        *,
        env_config: ToyEnvConfig | None = None,
        knob_constraints: KnobConstraints | None = None,
        action_profile: str = "all",
        window_len: int = 20,
        headroom_margin: float = 0.05,
    ) -> None:
        self.cfg = env_config or ToyEnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.action_space = KnobActionSpace(knob_constraints or KnobConstraints(), action_profile=action_profile)
        self.state_builder = StateBuilder(budget_mb=self.cfg.budget_mb, action_space=self.action_space)
        self.telemetry = TelemetryWindow(maxlen=window_len)

        self.safety = SafetyShield(budget_mb=self.cfg.budget_mb, headroom_margin=headroom_margin)

        self.step_idx = 0
        self.config = KnobConfig().clipped(self.action_space.constraints)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_idx = 0
        self.telemetry = TelemetryWindow(maxlen=self.telemetry.maxlen)
        self.config = KnobConfig().clipped(self.action_space.constraints)
        self.safety.record_safe(self.config)

        initial = self._simulate_step(self.config, allow_oom=False)
        self.telemetry.append(initial)
        obs = self.state_builder.build(self.telemetry, self.config)
        return obs, {"reset": True}

    def step(self, action_id: int) -> StepResult:
        last = self.telemetry.last()
        if last is None:
            raise RuntimeError("env must be reset before stepping")

        proposed = self.action_space.apply(action_id, self.config)
        safety = self.safety.check(last=last, current=self.config, proposed=proposed)

        info: Dict[str, Any] = {"action_id": action_id, "action_name": self.action_space.names()[action_id]}
        if not safety.allowed:
            # Treat as no-op if blocked; allow the policy to learn that it is ineffective.
            info["shielded"] = True
            info["shield_reason"] = safety.reason
            proposed = self.config
        else:
            info["shielded"] = False

        sample = self._simulate_step(proposed, allow_oom=True)
        self.telemetry.append(sample)
        self.step_idx += 1

        reward_cfg = RewardConfig(
            mode=self.cfg.reward_mode,
            sequence_length=self.cfg.sequence_length,
            loss_delta_per_update=self.cfg.loss_delta_per_update,
            log_progress=self.cfg.reward_log_progress,
            scale=self.cfg.reward_scale,
            oom_penalty=self.cfg.oom_penalty,
            restart_penalty=self.cfg.restart_penalty,
        )
        reward = compute_reward(
            config=reward_cfg,
            elapsed_s=sample.step_time_s,
            world_size=self.cfg.world_size,
            micro_batch=proposed.micro_batch,
            grad_accum_steps=proposed.grad_accum_steps,
            oom=sample.oom,
            restart=sample.restart,
        )
        peak_ratio = float(sample.vram_peak_mb / self.cfg.budget_mb)
        cost = max(0.0, peak_ratio - 1.0)

        if sample.oom:
            # Roll back to last safe configuration if available.
            safe = self.safety.last_safe_config()
            if safe is not None:
                self.config = safe
                info["rollback"] = True
            else:
                info["rollback"] = False
        else:
            self.config = proposed
            self.safety.record_safe(self.config)
            info["rollback"] = False

        obs = self.state_builder.build(self.telemetry, self.config)
        done = False
        return StepResult(observation=obs, reward=reward, cost=cost, done=done, info=info)

    def _precision_factor(self, precision: str) -> float:
        return {"fp32": 1.0, "fp16": 0.70, "bf16": 0.70}.get(precision, 1.0)

    def _checkpoint_factor(self, enabled: bool) -> float:
        return 0.65 if enabled else 1.0

    def _shard_factor(self, sharding: str) -> float:
        # Roughly model that sharded state reduces persistent footprint with world size.
        if sharding == "fsdp_full_shard":
            return float(max(1, self.cfg.world_size))
        return 1.0

    def _simulate_step(self, config: KnobConfig, *, allow_oom: bool) -> TelemetrySample:
        precision_factor = self._precision_factor(config.precision)
        ckpt_factor = self._checkpoint_factor(config.activation_checkpointing)

        model_state_mb = float(self.cfg.base_model_state_mb) / self._shard_factor(config.sharding)
        activation_mb = float(self.cfg.activation_mb_per_sample_fp32) * float(config.micro_batch) * precision_factor * ckpt_factor
        transient_mb = float(self.cfg.transient_mb) + float(config.bucket_cap_mb)

        noise_mb = float(self.rng.normal(0.0, self.cfg.noise_std_mb))
        peak_mb = max(0.0, model_state_mb + activation_mb + transient_mb + noise_mb)
        oom = bool(peak_mb > self.cfg.budget_mb) if allow_oom else False

        compute_time = float(self.cfg.base_compute_time_s) + float(self.cfg.compute_time_per_sample_s) * float(
            config.micro_batch * config.grad_accum_steps
        )
        if config.activation_checkpointing:
            compute_time *= 1.20  # recomputation overhead (toy)

        comm_time = float(self.cfg.comm_time_s) * (1.0 + float(config.bucket_cap_mb) / 128.0)

        io_time = 0.0
        if config.ckpt_interval_steps > 0 and (self.step_idx % config.ckpt_interval_steps == 0) and self.step_idx > 0:
            io_time = float(self.cfg.ckpt_size_mb) / max(1.0, float(self.cfg.io_bandwidth_mb_s))

        step_time = compute_time + comm_time + io_time
        return TelemetrySample(
            step=int(self.step_idx),
            vram_allocated_mb=peak_mb * 0.85,
            vram_reserved_mb=peak_mb * 0.95,
            vram_peak_mb=peak_mb,
            step_time_s=step_time,
            compute_time_s=compute_time,
            comm_time_s=comm_time,
            io_time_s=io_time,
            oom=oom,
            restart=False,
        )
