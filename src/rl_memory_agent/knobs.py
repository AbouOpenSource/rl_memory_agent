from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence


ACTION_PROFILES = ("all", "fast_only")


@dataclass(frozen=True)
class KnobConstraints:
    micro_batch_min: int = 1
    micro_batch_max: int = 64
    grad_accum_min: int = 1
    grad_accum_max: int = 32
    bucket_cap_mb_min: int = 1
    bucket_cap_mb_max: int = 256
    ckpt_interval_min: int = 0
    ckpt_interval_max: int = 10_000
    precisions: Sequence[str] = ("fp32", "fp16", "bf16")
    sharding_modes: Sequence[str] = ("ddp", "fsdp_full_shard")


@dataclass(frozen=True)
class KnobConfig:
    micro_batch: int = 1
    grad_accum_steps: int = 1
    activation_checkpointing: bool = False
    precision: str = "fp32"
    sharding: str = "ddp"
    bucket_cap_mb: int = 25
    ckpt_interval_steps: int = 0  # 0 disables periodic checkpoints in this scaffold

    def clipped(self, constraints: KnobConstraints) -> "KnobConfig":
        micro_batch = int(min(max(self.micro_batch, constraints.micro_batch_min), constraints.micro_batch_max))
        grad_accum_steps = int(min(max(self.grad_accum_steps, constraints.grad_accum_min), constraints.grad_accum_max))
        bucket_cap_mb = int(min(max(self.bucket_cap_mb, constraints.bucket_cap_mb_min), constraints.bucket_cap_mb_max))
        ckpt_interval_steps = int(
            min(max(self.ckpt_interval_steps, constraints.ckpt_interval_min), constraints.ckpt_interval_max)
        )

        precision = self.precision if self.precision in constraints.precisions else str(constraints.precisions[0])
        sharding = self.sharding if self.sharding in constraints.sharding_modes else str(constraints.sharding_modes[0])

        return KnobConfig(
            micro_batch=micro_batch,
            grad_accum_steps=grad_accum_steps,
            activation_checkpointing=bool(self.activation_checkpointing),
            precision=precision,
            sharding=sharding,
            bucket_cap_mb=bucket_cap_mb,
            ckpt_interval_steps=ckpt_interval_steps,
        )


@dataclass(frozen=True)
class KnobAction:
    name: str
    fn: Callable[[KnobConfig, KnobConstraints], KnobConfig]

    def apply(self, config: KnobConfig, constraints: KnobConstraints) -> KnobConfig:
        return self.fn(config, constraints).clipped(constraints)


class KnobActionSpace:
    def __init__(self, constraints: KnobConstraints | None = None, *, action_profile: str = "all") -> None:
        if action_profile not in ACTION_PROFILES:
            raise ValueError(f"unsupported action_profile={action_profile!r}; supported={ACTION_PROFILES}")
        self.constraints = constraints or KnobConstraints()
        self.action_profile = action_profile
        self._actions: List[KnobAction] = self._build_default_actions()

    def _build_default_actions(self) -> List[KnobAction]:
        c = self.constraints

        def noop(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return config

        def micro_batch_up(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "micro_batch": int(config.micro_batch * 2)})

        def micro_batch_down(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "micro_batch": max(c.micro_batch_min, int(config.micro_batch // 2))})

        def grad_accum_up(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "grad_accum_steps": int(config.grad_accum_steps + 1)})

        def grad_accum_down(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "grad_accum_steps": max(c.grad_accum_min, int(config.grad_accum_steps - 1))})

        def toggle_checkpointing(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "activation_checkpointing": (not config.activation_checkpointing)})

        def set_precision(precision: str) -> Callable[[KnobConfig, KnobConstraints], KnobConfig]:
            def _fn(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
                return KnobConfig(**{**config.__dict__, "precision": precision})

            return _fn

        def set_sharding(sharding: str) -> Callable[[KnobConfig, KnobConstraints], KnobConfig]:
            def _fn(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
                return KnobConfig(**{**config.__dict__, "sharding": sharding})

            return _fn

        def bucket_up(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "bucket_cap_mb": int(config.bucket_cap_mb + 16)})

        def bucket_down(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "bucket_cap_mb": int(config.bucket_cap_mb - 16)})

        def ckpt_interval_up(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "ckpt_interval_steps": int(config.ckpt_interval_steps + 200)})

        def ckpt_interval_down(config: KnobConfig, _c: KnobConstraints) -> KnobConfig:
            return KnobConfig(**{**config.__dict__, "ckpt_interval_steps": int(config.ckpt_interval_steps - 200)})

        actions: List[KnobAction] = [
            KnobAction("noop", noop),
            KnobAction("micro_batch_x2", micro_batch_up),
            KnobAction("micro_batch_half", micro_batch_down),
            KnobAction("grad_accum_plus1", grad_accum_up),
            KnobAction("grad_accum_minus1", grad_accum_down),
            KnobAction("toggle_checkpointing", toggle_checkpointing),
            KnobAction("set_precision_fp32", set_precision("fp32")),
            KnobAction("set_precision_fp16", set_precision("fp16")),
            KnobAction("set_precision_bf16", set_precision("bf16")),
            KnobAction("set_sharding_ddp", set_sharding("ddp")),
            KnobAction("set_sharding_fsdp_full_shard", set_sharding("fsdp_full_shard")),
            KnobAction("bucket_plus16mb", bucket_up),
            KnobAction("bucket_minus16mb", bucket_down),
            KnobAction("ckpt_interval_plus200", ckpt_interval_up),
            KnobAction("ckpt_interval_minus200", ckpt_interval_down),
        ]

        if self.action_profile == "fast_only":
            fast_names = {
                "noop",
                "micro_batch_x2",
                "micro_batch_half",
                "grad_accum_plus1",
                "grad_accum_minus1",
                "bucket_plus16mb",
                "bucket_minus16mb",
            }
            actions = [a for a in actions if a.name in fast_names]

        return actions

    @property
    def n(self) -> int:
        return len(self._actions)

    def names(self) -> List[str]:
        return [a.name for a in self._actions]

    def index(self, name: str) -> int:
        for i, action in enumerate(self._actions):
            if action.name == name:
                return i
        raise ValueError(f"unknown action name for profile {self.action_profile!r}: {name!r}")

    def apply(self, action_id: int, config: KnobConfig) -> KnobConfig:
        if not (0 <= action_id < self.n):
            raise IndexError(f"action_id must be in [0, {self.n}), got {action_id}")
        action = self._actions[action_id]
        return action.apply(config, self.constraints)
