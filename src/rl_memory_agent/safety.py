from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_memory_agent.knobs import KnobConfig
from rl_memory_agent.telemetry import TelemetrySample


class MemoryPredictor(Protocol):
    def predict_peak_mb(
        self, *, last: TelemetrySample, current: KnobConfig, proposed: KnobConfig, budget_mb: float
    ) -> float: ...


@dataclass(frozen=True)
class HeuristicMemoryPredictor:
    """
    A lightweight predictor used by the safety shield.

    It is deliberately conservative and intended as a placeholder for:
    - learned predictors from profiling data
    - analytical estimators based on activation shapes and optimizer state
    """

    activation_fraction: float = 0.6  # fraction of peak attributed to activations (rough proxy)

    def _precision_activation_factor(self, precision: str) -> float:
        return {"fp32": 1.0, "fp16": 0.70, "bf16": 0.70}.get(precision, 1.0)

    def _checkpoint_activation_factor(self, enabled: bool) -> float:
        return 0.65 if enabled else 1.0

    def _sharding_factor(self, sharding: str) -> float:
        # Very rough: sharding reduces persistent state but can increase transients.
        return {"ddp": 1.0, "fsdp_full_shard": 0.9}.get(sharding, 1.0)

    def predict_peak_mb(
        self, *, last: TelemetrySample, current: KnobConfig, proposed: KnobConfig, budget_mb: float
    ) -> float:
        last_peak = float(last.vram_peak_mb)
        if current.micro_batch <= 0:
            return last_peak

        micro_batch_scale = float(proposed.micro_batch) / float(current.micro_batch)
        precision_scale = self._precision_activation_factor(proposed.precision) / self._precision_activation_factor(
            current.precision
        )
        checkpoint_scale = self._checkpoint_activation_factor(proposed.activation_checkpointing) / self._checkpoint_activation_factor(
            current.activation_checkpointing
        )
        sharding_scale = self._sharding_factor(proposed.sharding) / self._sharding_factor(current.sharding)

        activation_mb = last_peak * float(self.activation_fraction)
        constant_mb = max(0.0, last_peak - activation_mb)

        predicted_activation = activation_mb * micro_batch_scale * precision_scale * checkpoint_scale
        predicted_peak = constant_mb + predicted_activation

        # Apply sharding adjustment on the constant component (proxy for persistent state).
        predicted_peak = (constant_mb * sharding_scale) + predicted_activation

        # Conservative floor/ceiling to avoid degenerate predictions.
        predicted_peak = max(0.0, min(predicted_peak, 2.0 * float(budget_mb)))
        return predicted_peak


@dataclass
class SafetyResult:
    allowed: bool
    reason: str


class SafetyShield:
    def __init__(
        self,
        *,
        budget_mb: float,
        predictor: MemoryPredictor | None = None,
        headroom_margin: float = 0.05,
    ) -> None:
        if budget_mb <= 0:
            raise ValueError("budget_mb must be > 0")
        if not (0.0 <= headroom_margin < 1.0):
            raise ValueError("headroom_margin must be in [0, 1)")

        self.budget_mb = float(budget_mb)
        self.predictor = predictor or HeuristicMemoryPredictor()
        self.headroom_margin = float(headroom_margin)
        self._last_safe_config: KnobConfig | None = None

    def last_safe_config(self) -> KnobConfig | None:
        return self._last_safe_config

    def record_safe(self, config: KnobConfig) -> None:
        self._last_safe_config = config

    def check(self, *, last: TelemetrySample, current: KnobConfig, proposed: KnobConfig) -> SafetyResult:
        predicted_peak = self.predictor.predict_peak_mb(
            last=last, current=current, proposed=proposed, budget_mb=self.budget_mb
        )
        limit = self.budget_mb * (1.0 - self.headroom_margin)

        if predicted_peak > limit:
            return SafetyResult(
                allowed=False,
                reason=f"blocked_by_headroom(pred_peak_mb={predicted_peak:.1f}, limit_mb={limit:.1f})",
            )
        return SafetyResult(allowed=True, reason="ok")

