from __future__ import annotations

import argparse
import os
import shutil
import socket
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from rl_memory_agent.knobs import ACTION_PROFILES, KnobActionSpace, KnobConfig, KnobConstraints
from rl_memory_agent.ppo_lagrangian import LagrangianConfig, LagrangianPPO, PPOConfig, RolloutBuffer
from rl_memory_agent.reward import REWARD_MODES, RewardConfig, compute_reward
from rl_memory_agent.safety import SafetyShield
from rl_memory_agent.state import StateBuilder
from rl_memory_agent.telemetry import TelemetrySample, TelemetryWindow


@dataclass(frozen=True)
class SimGridHello:
    world_size: int
    steps: int
    control_interval: int
    budget_mb: float
    headroom_margin: float
    micro_batch_min: int
    micro_batch_max: int
    grad_accum_min: int
    grad_accum_max: int
    ckpt_interval_min: int
    ckpt_interval_max: int
    bucket_cap_mb_min: int
    bucket_cap_mb_max: int
    precisions: Tuple[str, ...]
    sharding_modes: Tuple[str, ...]


@dataclass(frozen=True)
class SimGridTelemetry:
    step_end: int
    apply_step: int
    interval_steps: int
    mean_step_s: float
    mean_compute_s: float
    mean_comm_s: float
    mean_io_s: float
    mean_peak_mb: float
    max_peak_mb: float
    any_oom: bool
    any_restart: bool
    knobs: KnobConfig


_SUPPORTED_COST_COMPONENTS = {
    "mem_overflow",
    "comm_frac",
    "io_frac",
    "oom",
    "restart",
    "switch_cost",
}

_DEFAULT_COST_LIMITS: Dict[str, float] = {
    "mem_overflow": 0.0,
    "comm_frac": 0.35,
    "io_frac": 0.10,
    "oom": 0.0,
    "restart": 0.0,
    "switch_cost": 0.02,
}


def _parse_kv_line(line: str) -> Tuple[str, Dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        return "", {}
    cmd = parts[0]
    kv: Dict[str, str] = {}
    for tok in parts[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        kv[k] = v
    return cmd, kv


def _recv_line(sock: socket.socket) -> Optional[str]:
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            return None
        if b == b"\n":
            return buf.decode("utf-8", errors="replace")
        buf.extend(b)


def _send_line(sock: socket.socket, line: str) -> None:
    sock.sendall((line.rstrip("\n") + "\n").encode("utf-8"))


def _parse_csv_list(value: str, *, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if not value:
        return default
    items = [x.strip() for x in value.split(",") if x.strip()]
    return tuple(items) if items else default


def _parse_csv_floats(value: str) -> Tuple[float, ...]:
    items: List[float] = []
    for tok in (x.strip() for x in value.split(",")):
        if not tok:
            continue
        try:
            items.append(float(tok))
        except ValueError as exc:
            raise ValueError(f"invalid float in list: {tok!r}") from exc
    return tuple(items)


def _hello_from_kv(kv: Dict[str, str]) -> SimGridHello:
    def i(name: str, default: int) -> int:
        try:
            return int(kv.get(name, str(default)))
        except ValueError:
            return default

    def f(name: str, default: float) -> float:
        try:
            return float(kv.get(name, str(default)))
        except ValueError:
            return default

    precisions = _parse_csv_list(kv.get("precisions", ""), default=("fp32", "fp16", "bf16"))
    sharding_modes = _parse_csv_list(kv.get("sharding_modes", ""), default=("ddp", "fsdp_full_shard"))

    world_size = max(1, i("world_size", 1))
    steps = max(1, i("steps", 1))
    control_interval = max(1, i("control_interval", 1))
    budget_mb = max(1.0, f("budget_mb", 16_000.0))
    headroom_margin = min(max(f("headroom_margin", 0.05), 0.0), 0.99)

    micro_batch_min = max(1, i("micro_batch_min", 1))
    micro_batch_max = max(micro_batch_min, i("micro_batch_max", 64))
    grad_accum_min = max(1, i("grad_accum_min", 1))
    grad_accum_max = max(grad_accum_min, i("grad_accum_max", 32))
    ckpt_interval_min = max(0, i("ckpt_interval_min", 0))
    ckpt_interval_max = max(ckpt_interval_min, i("ckpt_interval_max", 10_000))
    bucket_cap_mb_min = max(0, i("bucket_cap_mb_min", 1))
    bucket_cap_mb_max = max(bucket_cap_mb_min, i("bucket_cap_mb_max", 256))

    return SimGridHello(
        world_size=world_size,
        steps=steps,
        control_interval=control_interval,
        budget_mb=budget_mb,
        headroom_margin=headroom_margin,
        micro_batch_min=micro_batch_min,
        micro_batch_max=micro_batch_max,
        grad_accum_min=grad_accum_min,
        grad_accum_max=grad_accum_max,
        ckpt_interval_min=ckpt_interval_min,
        ckpt_interval_max=ckpt_interval_max,
        bucket_cap_mb_min=bucket_cap_mb_min,
        bucket_cap_mb_max=bucket_cap_mb_max,
        precisions=precisions,
        sharding_modes=sharding_modes,
    )


def _telemetry_from_kv(kv: Dict[str, str], *, action_space: KnobActionSpace) -> SimGridTelemetry:
    def i(name: str, default: int) -> int:
        try:
            return int(kv.get(name, str(default)))
        except ValueError:
            return default

    def f(name: str, default: float) -> float:
        try:
            return float(kv.get(name, str(default)))
        except ValueError:
            return default

    step_end = i("step_end", i("step", 0))
    apply_step = i("apply_step", 0)
    interval_steps = max(1, i("interval_steps", 1))

    mean_step_s = max(0.0, f("mean_step_s", 0.0))
    mean_compute_s = max(0.0, f("mean_compute_s", 0.0))
    mean_comm_s = max(0.0, f("mean_comm_s", 0.0))
    mean_io_s = max(0.0, f("mean_io_s", 0.0))

    mean_peak_mb = max(0.0, f("mean_peak_mb", 0.0))
    max_peak_mb = max(0.0, f("max_peak_mb", 0.0))

    any_oom = bool(i("any_oom", 0) != 0)
    any_restart = bool(i("any_restart", 0) != 0)

    knobs = KnobConfig(
        micro_batch=i("micro_batch", 1),
        grad_accum_steps=i("grad_accum_steps", 1),
        activation_checkpointing=bool(i("activation_checkpointing", 0) != 0),
        precision=str(kv.get("precision", "fp32")),
        sharding=str(kv.get("sharding", "ddp")),
        bucket_cap_mb=i("bucket_cap_mb", 25),
        ckpt_interval_steps=i("ckpt_interval_steps", 0),
    ).clipped(action_space.constraints)

    return SimGridTelemetry(
        step_end=step_end,
        apply_step=apply_step,
        interval_steps=interval_steps,
        mean_step_s=mean_step_s,
        mean_compute_s=mean_compute_s,
        mean_comm_s=mean_comm_s,
        mean_io_s=mean_io_s,
        mean_peak_mb=mean_peak_mb,
        max_peak_mb=max_peak_mb,
        any_oom=any_oom,
        any_restart=any_restart,
        knobs=knobs,
    )


def _config_line(config: KnobConfig) -> str:
    # Keep this line protocol intentionally simple (space-separated key=value pairs).
    return (
        "CONFIG"
        f" micro_batch={int(config.micro_batch)}"
        f" grad_accum_steps={int(config.grad_accum_steps)}"
        f" ckpt_interval_steps={int(config.ckpt_interval_steps)}"
        f" bucket_cap_mb={int(config.bucket_cap_mb)}"
        f" precision={config.precision}"
        f" activation_checkpointing={1 if config.activation_checkpointing else 0}"
        f" sharding={config.sharding}"
    )


def _knob_switch_cost(previous: KnobConfig | None, current: KnobConfig) -> float:
    if previous is None:
        return 0.0

    cost = 0.0
    if previous.micro_batch != current.micro_batch:
        cost += 0.02
    if previous.grad_accum_steps != current.grad_accum_steps:
        cost += 0.01
    if previous.bucket_cap_mb != current.bucket_cap_mb:
        cost += 0.005
    if previous.ckpt_interval_steps != current.ckpt_interval_steps:
        cost += 0.02
    if previous.activation_checkpointing != current.activation_checkpointing:
        cost += 0.10
    if previous.precision != current.precision:
        cost += 0.15
    if previous.sharding != current.sharding:
        cost += 0.20
    return float(min(cost, 1.0))


def _cost_vector(
    telemetry: SimGridTelemetry,
    hello: SimGridHello,
    *,
    components: Tuple[str, ...],
    switch_from: KnobConfig | None = None,
) -> np.ndarray:
    step_s = max(1e-6, float(telemetry.mean_step_s))
    out: List[float] = []
    for name in components:
        if name == "mem_overflow":
            out.append(max(0.0, float(telemetry.max_peak_mb / hello.budget_mb) - 1.0))
        elif name == "comm_frac":
            out.append(float(telemetry.mean_comm_s) / step_s)
        elif name == "io_frac":
            out.append(float(telemetry.mean_io_s) / step_s)
        elif name == "oom":
            out.append(float(telemetry.any_oom))
        elif name == "restart":
            out.append(float(telemetry.any_restart))
        elif name == "switch_cost":
            out.append(_knob_switch_cost(switch_from, telemetry.knobs))
        else:
            raise ValueError(f"unsupported cost component: {name}")
    return np.asarray(out, dtype=np.float32)


def _fmt_scalar_or_list(x: float | list[float]) -> str:
    if isinstance(x, list):
        return "[" + ",".join(f"{v:.3f}" for v in x) + "]"
    return f"{float(x):.3f}"


def run_simgrid_online_controller(
    *,
    host: str,
    port: int,
    rollout_steps: int,
    window_len: int,
    cost_components: Tuple[str, ...],
    cost_limits: Tuple[float, ...],
    reward_mode: str,
    sequence_length: int,
    loss_delta_per_update: float,
    reward_log_progress: bool,
    reward_scale: float,
    oom_penalty: float,
    restart_penalty: float,
    headroom_margin: Optional[float],
    action_profile: str,
    entropy_coef: float,
    device: str,
    log_every: int,
    checkpoint_dir: Optional[str],
    save_every_updates: int,
    resume_from: Optional[str],
) -> None:
    if not cost_components:
        raise ValueError("cost_components must be non-empty")
    for name in cost_components:
        if name not in _SUPPORTED_COST_COMPONENTS:
            raise ValueError(f"unsupported cost component: {name!r} (supported: {sorted(_SUPPORTED_COST_COMPONENTS)})")
    if len(cost_limits) != len(cost_components):
        raise ValueError("cost_limits must have the same length as cost_components")
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"unsupported reward_mode: {reward_mode!r} (supported: {REWARD_MODES})")
    if int(save_every_updates) < 0:
        raise ValueError("save_every_updates must be >= 0")

    with socket.create_connection((host, int(port))) as sock:
        hello_line = _recv_line(sock)
        if hello_line is None:
            raise RuntimeError("server closed before HELLO")

        cmd, kv = _parse_kv_line(hello_line)
        if cmd != "HELLO":
            raise RuntimeError(f"expected HELLO, got: {hello_line!r}")

        hello = _hello_from_kv(kv)
        constraints = KnobConstraints(
            micro_batch_min=hello.micro_batch_min,
            micro_batch_max=hello.micro_batch_max,
            grad_accum_min=hello.grad_accum_min,
            grad_accum_max=hello.grad_accum_max,
            ckpt_interval_min=hello.ckpt_interval_min,
            ckpt_interval_max=hello.ckpt_interval_max,
            bucket_cap_mb_min=hello.bucket_cap_mb_min,
            bucket_cap_mb_max=hello.bucket_cap_mb_max,
            precisions=hello.precisions,
            sharding_modes=hello.sharding_modes,
        )
        action_space = KnobActionSpace(constraints, action_profile=action_profile)
        noop_action_id = action_space.index("noop")

        window = TelemetryWindow(maxlen=int(window_len))
        state_builder = StateBuilder(budget_mb=hello.budget_mb, action_space=action_space)
        safety = SafetyShield(
            budget_mb=hello.budget_mb,
            headroom_margin=float(hello.headroom_margin if headroom_margin is None else headroom_margin),
        )
        reward_cfg = RewardConfig(
            mode=reward_mode,
            sequence_length=max(1, int(sequence_length)),
            loss_delta_per_update=float(loss_delta_per_update),
            log_progress=bool(reward_log_progress),
            scale=float(reward_scale),
            oom_penalty=float(oom_penalty),
            restart_penalty=float(restart_penalty),
        )

        # We will initialize algo lazily after receiving the first telemetry and building an obs.
        algo: Optional[LagrangianPPO] = None
        buffer: Optional[RolloutBuffer] = None

        pending: Optional[Tuple[np.ndarray, int, float, float, KnobConfig]] = None
        decisions = 0
        updates = 0
        recent_rewards: List[float] = []
        recent_costs: List[np.ndarray] = []
        resume_path = resume_from

        while True:
            line = _recv_line(sock)
            if line is None:
                break

            cmd, kv = _parse_kv_line(line)
            if cmd != "TELEMETRY":
                continue

            telemetry = _telemetry_from_kv(kv, action_space=action_space)

            # Build a telemetry sample compatible with the RL state builder.
            peak_mb = float(telemetry.max_peak_mb)
            sample = TelemetrySample(
                step=int(telemetry.step_end),
                vram_allocated_mb=peak_mb * 0.85,
                vram_reserved_mb=peak_mb * 0.95,
                vram_peak_mb=peak_mb,
                step_time_s=float(telemetry.mean_step_s),
                compute_time_s=float(telemetry.mean_compute_s),
                comm_time_s=float(telemetry.mean_comm_s),
                io_time_s=float(telemetry.mean_io_s),
                oom=bool(telemetry.any_oom),
                restart=bool(telemetry.any_restart),
            )
            window.append(sample)

            obs = state_builder.build(window, telemetry.knobs)
            if algo is None:
                algo = LagrangianPPO(
                    obs_dim=int(obs.shape[0]),
                    n_actions=action_space.n,
                    ppo=PPOConfig(entropy_coef=float(entropy_coef)),
                    lagrangian=LagrangianConfig(cost_limits=tuple(float(x) for x in cost_limits)),
                    device=device,
                )
                buffer = RolloutBuffer(obs_dim=int(obs.shape[0]), size=int(rollout_steps), n_costs=len(cost_components))
                print(f"cost_components={','.join(cost_components)} cost_limits={','.join(str(x) for x in cost_limits)}")
                print(
                    f"reward_mode={reward_cfg.mode} sequence_length={reward_cfg.sequence_length} "
                    f"reward_log_progress={int(reward_cfg.log_progress)} reward_scale={reward_cfg.scale}"
                )
                print(f"action_profile={action_space.action_profile} n_actions={action_space.n} entropy_coef={entropy_coef}")
                if resume_path:
                    if not os.path.isfile(resume_path):
                        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
                    extra = algo.load_checkpoint(resume_path)
                    restored_updates = int(extra.get("updates", 0)) if isinstance(extra, dict) else 0
                    restored_decisions = int(extra.get("decisions", 0)) if isinstance(extra, dict) else 0
                    updates = max(updates, restored_updates)
                    decisions = max(decisions, restored_decisions)
                    print(
                        f"resumed_from={resume_path} restored_updates={restored_updates} "
                        f"restored_decisions={restored_decisions}"
                    )
                    resume_path = None

            assert algo is not None
            assert buffer is not None

            # Reward/cost observed for the *interval that just ended* (config chosen previously).
            reward = compute_reward(
                config=reward_cfg,
                elapsed_s=telemetry.mean_step_s,
                world_size=hello.world_size,
                micro_batch=telemetry.knobs.micro_batch,
                grad_accum_steps=telemetry.knobs.grad_accum_steps,
                oom=telemetry.any_oom,
                restart=telemetry.any_restart,
            )

            if pending is not None:
                prev_obs, prev_action, prev_log_prob, prev_value, prev_knobs = pending
                cost = _cost_vector(
                    telemetry,
                    hello,
                    components=cost_components,
                    switch_from=prev_knobs,
                )
                buffer.add(
                    obs=prev_obs,
                    action=prev_action,
                    log_prob=prev_log_prob,
                    value=prev_value,
                    reward=reward,
                    cost=cost,
                    done=False,
                )
                recent_rewards.append(reward)
                recent_costs.append(cost)

                if buffer.full:
                    last_value = algo.predict_value(obs)
                    metrics = algo.update(buffer.get(), last_value=last_value)
                    buffer.reset()
                    updates += 1

                    if updates % max(1, int(log_every)) == 0:
                        mean_r = float(np.mean(recent_rewards[-rollout_steps:])) if recent_rewards else 0.0
                        if recent_costs:
                            mean_c_vec = np.mean(np.stack(recent_costs[-rollout_steps:]), axis=0)
                            mean_c = "[" + ",".join(f"{v:.3f}" for v in mean_c_vec.tolist()) + "]"
                        else:
                            mean_c = "[]"
                        print(
                            f"update={updates:04d} decisions={decisions:07d} "
                            f"mean_reward={mean_r:+.4f} mean_cost={mean_c} "
                            f"lambda={_fmt_scalar_or_list(metrics['lambda'])}"
                        )

                    if checkpoint_dir and int(save_every_updates) > 0 and (updates % int(save_every_updates) == 0):
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        ckpt_path = os.path.join(checkpoint_dir, f"update_{updates:06d}.pt")
                        extra = {
                            "updates": int(updates),
                            "decisions": int(decisions),
                            "cost_components": list(cost_components),
                            "cost_limits": [float(x) for x in cost_limits],
                            "reward_mode": reward_cfg.mode,
                            "sequence_length": int(reward_cfg.sequence_length),
                            "action_profile": action_space.action_profile,
                            "entropy_coef": float(entropy_coef),
                        }
                        algo.save_checkpoint(ckpt_path, extra=extra)
                        latest_path = os.path.join(checkpoint_dir, "latest.pt")
                        shutil.copyfile(ckpt_path, latest_path)
                        print(f"checkpoint_saved={ckpt_path}")

            # Record last safe configuration from observed outcomes.
            limit_mb = hello.budget_mb * (1.0 - safety.headroom_margin)
            if (not telemetry.any_oom) and (telemetry.max_peak_mb <= limit_mb):
                safety.record_safe(telemetry.knobs)

            # Choose next action (config to apply at the next control boundary).
            action_id, log_prob, value = algo.select_action(obs)
            proposed = action_space.apply(action_id, telemetry.knobs)
            stored_action_id = int(action_id)
            stored_log_prob = float(log_prob)
            stored_value = float(value)

            # Safety shield: block risky actions.
            safety_res = safety.check(last=sample, current=telemetry.knobs, proposed=proposed)
            if not safety_res.allowed:
                proposed = telemetry.knobs
                stored_action_id = noop_action_id
                stored_log_prob, stored_value = algo.log_prob_value(obs, stored_action_id)

            # Hard rollback if we just observed an OOM/restart.
            if telemetry.any_oom or telemetry.any_restart:
                safe = safety.last_safe_config()
                if safe is not None:
                    proposed = safe

            _send_line(sock, _config_line(proposed))

            pending = (obs, stored_action_id, stored_log_prob, stored_value, telemetry.knobs)
            decisions += 1

        if checkpoint_dir and algo is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            final_path = os.path.join(checkpoint_dir, "final.pt")
            extra = {
                "updates": int(updates),
                "decisions": int(decisions),
                "cost_components": list(cost_components),
                "cost_limits": [float(x) for x in cost_limits],
                "reward_mode": reward_cfg.mode,
                "sequence_length": int(reward_cfg.sequence_length),
                "action_profile": action_space.action_profile,
                "entropy_coef": float(entropy_coef),
            }
            algo.save_checkpoint(final_path, extra=extra)
            latest_path = os.path.join(checkpoint_dir, "latest.pt")
            shutil.copyfile(final_path, latest_path)
            print(f"checkpoint_saved={final_path}")


def add_simgrid_subcommand(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("simgrid", help="Train online by controlling a SimGrid cluster env (external controller).")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--window-len", type=int, default=20)
    p.add_argument("--action-profile", choices=ACTION_PROFILES, default="fast_only")
    p.add_argument("--cost-limit", type=float, default=0.0)
    p.add_argument(
        "--cost-components",
        type=str,
        default="mem_overflow,switch_cost",
        help="Comma-separated list: mem_overflow,comm_frac,io_frac,oom,restart,switch_cost",
    )
    p.add_argument(
        "--cost-limits",
        type=str,
        default=None,
        help="Comma-separated floats aligned with --cost-components (defaults are used when omitted).",
    )
    p.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="neg_step_time",
        help="Scalar reward objective: legacy negative step time or useful progress per second.",
    )
    p.add_argument("--sequence-length", type=int, default=1, help="Tokens per sample for tokens_per_second reward.")
    p.add_argument(
        "--loss-delta-per-update",
        type=float,
        default=1.0,
        help="Loss-progress proxy for loss_delta_per_second reward; real integrations should pass measured loss delta.",
    )
    p.add_argument("--reward-scale", type=float, default=1.0, help="Divisor applied before reward transform.")
    p.add_argument(
        "--no-reward-log-progress",
        dest="reward_log_progress",
        action="store_false",
        help="Use raw progress/sec rewards instead of log1p(progress/sec).",
    )
    p.add_argument("--oom-penalty", type=float, default=5.0)
    p.add_argument("--restart-penalty", type=float, default=0.0)
    p.add_argument("--headroom-margin", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=0.001)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to periodically save agent checkpoints.")
    p.add_argument(
        "--save-every-updates",
        type=int,
        default=0,
        help="Save checkpoint every N PPO updates (0 disables periodic saves).",
    )
    p.add_argument("--resume-from", type=str, default=None, help="Path to a checkpoint file to resume from.")

    def _run(args: argparse.Namespace) -> None:
        comps = _parse_csv_list(str(args.cost_components), default=("mem_overflow",))
        if args.cost_limits is not None:
            limits = _parse_csv_floats(str(args.cost_limits))
            if len(limits) != len(comps):
                raise ValueError("--cost-limits must have the same length as --cost-components")
        else:
            limits_list: List[float] = []
            for name in comps:
                base = float(_DEFAULT_COST_LIMITS.get(name, 0.0))
                if name == "mem_overflow":
                    base = float(args.cost_limit)
                limits_list.append(base)
            limits = tuple(limits_list)
        run_simgrid_online_controller(
            host=args.host,
            port=int(args.port),
            rollout_steps=int(args.rollout_steps),
            window_len=int(args.window_len),
            cost_components=comps,
            cost_limits=limits,
            reward_mode=str(args.reward_mode),
            sequence_length=int(args.sequence_length),
            loss_delta_per_update=float(args.loss_delta_per_update),
            reward_log_progress=bool(args.reward_log_progress),
            reward_scale=float(args.reward_scale),
            oom_penalty=float(args.oom_penalty),
            restart_penalty=float(args.restart_penalty),
            headroom_margin=None if args.headroom_margin is None else float(args.headroom_margin),
            action_profile=str(args.action_profile),
            entropy_coef=float(args.entropy_coef),
            device=str(args.device),
            log_every=int(args.log_every),
            checkpoint_dir=None if args.checkpoint_dir is None else str(args.checkpoint_dir),
            save_every_updates=int(args.save_every_updates),
            resume_from=None if args.resume_from is None else str(args.resume_from),
        )

    p.set_defaults(func=_run, reward_log_progress=True)
