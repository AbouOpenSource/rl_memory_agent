from __future__ import annotations

import argparse

from rl_memory_agent.agent import AgentRunner, TrainConfig
from rl_memory_agent.env import ToyEnvConfig, ToyMemoryEnv
from rl_memory_agent.ppo_lagrangian import LagrangianConfig, LagrangianPPO, PPOConfig
from rl_memory_agent.reward import REWARD_MODES
from rl_memory_agent.simgrid_online import add_simgrid_subcommand


def _cmd_toy(args: argparse.Namespace) -> None:
    env = ToyMemoryEnv(
        env_config=ToyEnvConfig(
            seed=args.seed,
            budget_mb=args.budget_mb,
            reward_mode=args.reward_mode,
            sequence_length=args.sequence_length,
            loss_delta_per_update=args.loss_delta_per_update,
            reward_log_progress=args.reward_log_progress,
            reward_scale=args.reward_scale,
            oom_penalty=args.oom_penalty,
            restart_penalty=args.restart_penalty,
        )
    )
    obs, _ = env.reset()

    algo = LagrangianPPO(
        obs_dim=int(obs.shape[0]),
        n_actions=env.action_space.n,
        ppo=PPOConfig(),
        lagrangian=LagrangianConfig(cost_limit=args.cost_limit),
        device=args.device,
    )
    runner = AgentRunner(env=env, algo=algo, train=TrainConfig(total_steps=args.total_steps))
    runner.train_loop()


def _cmd_actions(args: argparse.Namespace) -> None:
    env = ToyMemoryEnv()
    for i, name in enumerate(env.action_space.names()):
        print(f"{i:02d}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rl-memory-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_toy = sub.add_parser("toy", help="Run the agent on the toy environment.")
    p_toy.add_argument("--total-steps", type=int, default=50_000)
    p_toy.add_argument("--seed", type=int, default=0)
    p_toy.add_argument("--budget-mb", type=float, default=16_000.0)
    p_toy.add_argument("--cost-limit", type=float, default=0.0)
    p_toy.add_argument("--reward-mode", choices=REWARD_MODES, default="neg_step_time")
    p_toy.add_argument("--sequence-length", type=int, default=1)
    p_toy.add_argument("--loss-delta-per-update", type=float, default=1.0)
    p_toy.add_argument("--reward-scale", type=float, default=1.0)
    p_toy.add_argument("--no-reward-log-progress", dest="reward_log_progress", action="store_false")
    p_toy.add_argument("--oom-penalty", type=float, default=5.0)
    p_toy.add_argument("--restart-penalty", type=float, default=0.0)
    p_toy.add_argument("--device", type=str, default="cpu")
    p_toy.set_defaults(reward_log_progress=True)
    p_toy.set_defaults(func=_cmd_toy)

    p_actions = sub.add_parser("actions", help="List available knob actions.")
    p_actions.set_defaults(func=_cmd_actions)

    add_simgrid_subcommand(sub)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
