from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rl_memory_agent.knobs import KnobActionSpace, KnobConfig
from rl_memory_agent.telemetry import TelemetrySample, TelemetryWindow


@dataclass(frozen=True)
class StateSpec:
    names: Sequence[str]


class StateBuilder:
    """
    Maps a telemetry window + current knob configuration into a compact numeric state vector.

    This is intentionally lightweight and can be replaced by:
    - richer feature extractors
    - learned encoders (e.g., RNN over telemetry)
    - model-based predictors
    """

    def __init__(self, budget_mb: float, action_space: KnobActionSpace) -> None:
        if budget_mb <= 0:
            raise ValueError("budget_mb must be > 0")
        self.budget_mb = float(budget_mb)
        self.action_space = action_space

        self.spec = StateSpec(
            names=[
                # config
                "log2_micro_batch",
                "log2_grad_accum",
                "checkpointing",
                "precision_id",
                "sharding_id",
                "bucket_cap_norm",
                "ckpt_interval_norm",
                # telemetry (last)
                "peak_ratio",
                "reserved_ratio",
                "allocated_ratio",
                "step_time_s",
                "comm_frac",
                "io_frac",
                "oom",
                # telemetry (window means)
                "mean_peak_ratio",
                "mean_step_time_s",
            ]
        )

    def _precision_id(self, config: KnobConfig) -> float:
        precs = list(self.action_space.constraints.precisions)
        return float(precs.index(config.precision)) if config.precision in precs else 0.0

    def _sharding_id(self, config: KnobConfig) -> float:
        modes = list(self.action_space.constraints.sharding_modes)
        return float(modes.index(config.sharding)) if config.sharding in modes else 0.0

    def build(self, window: TelemetryWindow, config: KnobConfig) -> np.ndarray:
        last = window.last() or TelemetrySample(
            step=0,
            vram_allocated_mb=0.0,
            vram_reserved_mb=0.0,
            vram_peak_mb=0.0,
            step_time_s=0.0,
            compute_time_s=0.0,
            comm_time_s=0.0,
            io_time_s=0.0,
            oom=False,
            restart=False,
        )

        peak_ratio = float(last.vram_peak_mb / self.budget_mb)
        reserved_ratio = float(last.vram_reserved_mb / self.budget_mb)
        allocated_ratio = float(last.vram_allocated_mb / self.budget_mb)

        step_time = max(float(last.step_time_s), 1e-9)
        comm_frac = float(last.comm_time_s / step_time)
        io_frac = float(last.io_time_s / step_time)

        mean_peak_ratio = float(window.mean_peak_mb() / self.budget_mb) if len(window) else peak_ratio
        mean_step_time = float(window.mean_step_time_s())

        bucket_cap_norm = float(config.bucket_cap_mb / self.action_space.constraints.bucket_cap_mb_max)
        ckpt_interval_norm = float(
            config.ckpt_interval_steps / max(1.0, float(self.action_space.constraints.ckpt_interval_max))
        )

        features = np.array(
            [
                math.log2(max(1.0, float(config.micro_batch))),
                math.log2(max(1.0, float(config.grad_accum_steps))),
                float(config.activation_checkpointing),
                self._precision_id(config),
                self._sharding_id(config),
                bucket_cap_norm,
                ckpt_interval_norm,
                peak_ratio,
                reserved_ratio,
                allocated_ratio,
                float(last.step_time_s),
                comm_frac,
                io_frac,
                float(last.oom),
                mean_peak_ratio,
                mean_step_time,
            ],
            dtype=np.float32,
        )
        return features

