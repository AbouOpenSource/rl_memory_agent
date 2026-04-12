from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np

from rl_memory_agent.env import StepResult, ToyMemoryEnv
from rl_memory_agent.ppo_lagrangian import LagrangianPPO, RolloutBuffer


@dataclass(frozen=True)
class TrainConfig:
    rollout_steps: int = 1024
    total_steps: int = 50_000
    log_every: int = 10


class AgentRunner:
    def __init__(self, *, env: ToyMemoryEnv, algo: LagrangianPPO, train: TrainConfig) -> None:
        self.env = env
        self.algo = algo
        self.train = train

        obs, _info = self.env.reset()
        self.obs = obs
        self.buffer = RolloutBuffer(obs_dim=int(obs.shape[0]), size=self.train.rollout_steps)

        self.step = 0
        self.updates = 0

    def _collect_step(self) -> Tuple[StepResult, Dict[str, float]]:
        action, log_prob, value = self.algo.select_action(self.obs)
        result = self.env.step(action)

        self.buffer.add(
            obs=self.obs,
            action=action,
            log_prob=log_prob,
            value=value,
            reward=result.reward,
            cost=result.cost,
            done=result.done,
        )
        self.obs = result.observation

        metrics = {
            "reward": float(result.reward),
            "cost": float(result.cost),
            "shielded": float(result.info.get("shielded", False)),
            "rollback": float(result.info.get("rollback", False)),
        }
        return result, metrics

    def train_loop(self) -> None:
        recent_rewards = []
        recent_costs = []

        while self.step < self.train.total_steps:
            _result, metrics = self._collect_step()
            recent_rewards.append(metrics["reward"])
            recent_costs.append(metrics["cost"])
            self.step += 1

            if self.buffer.full:
                batch = self.buffer.get()
                last_value = self.algo.predict_value(self.obs)
                update_metrics = self.algo.update(batch, last_value=last_value)
                self.buffer.reset()
                self.updates += 1

                mean_r = float(np.mean(recent_rewards[-self.train.rollout_steps :]))
                mean_c = float(np.mean(recent_costs[-self.train.rollout_steps :]))
                if self.updates % self.train.log_every == 0:
                    lam: Union[float, list[float]] = update_metrics["lambda"]  # type: ignore[assignment]
                    if isinstance(lam, list):
                        lam_s = "[" + ",".join(f"{x:.3f}" for x in lam) + "]"
                    else:
                        lam_s = f"{float(lam):.3f}"
                    print(
                        f"update={self.updates:04d} step={self.step:07d} "
                        f"mean_reward={mean_r:+.4f} mean_cost={mean_c:.4f} "
                        f"lambda={lam_s}"
                    )
