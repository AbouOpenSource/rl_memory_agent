from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class LagrangianConfig:
    cost_limit: float = 0.0
    lambda_init: float = 0.0
    lambda_lr: float = 0.05
    lambda_max: float = 100.0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        layers = []
        last = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.Tanh())
            last = size
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, n_actions)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, obs_dim: int, size: int) -> None:
        self.size = int(size)
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.size,), dtype=np.int64)
        self.log_probs = np.zeros((self.size,), dtype=np.float32)
        self.values = np.zeros((self.size,), dtype=np.float32)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.costs = np.zeros((self.size,), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, *, obs: np.ndarray, action: int, log_prob: float, value: float, reward: float, cost: float, done: bool) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = int(action)
        self.log_probs[self.pos] = float(log_prob)
        self.values[self.pos] = float(value)
        self.rewards[self.pos] = float(reward)
        self.costs[self.pos] = float(cost)
        self.dones[self.pos] = float(done)
        self.pos += 1
        if self.pos >= self.size:
            self.full = True
            self.pos = 0

    def get(self) -> Dict[str, np.ndarray]:
        if not self.full:
            raise RuntimeError("buffer is not full")
        return {
            "obs": self.obs.copy(),
            "actions": self.actions.copy(),
            "log_probs": self.log_probs.copy(),
            "values": self.values.copy(),
            "rewards": self.rewards.copy(),
            "costs": self.costs.copy(),
            "dones": self.dones.copy(),
        }


def _gae(returns_rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(returns_rewards, dtype=np.float32)
    last_adv = 0.0
    last_value = 0.0
    for t in reversed(range(len(returns_rewards))):
        nonterminal = 1.0 - dones[t]
        delta = returns_rewards[t] + gamma * last_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
        last_value = values[t]
    returns = advantages + values
    return advantages, returns


class LagrangianPPO:
    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        ppo: PPOConfig | None = None,
        lagrangian: LagrangianConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.ppo = ppo or PPOConfig()
        self.lagrangian = lagrangian or LagrangianConfig()

        self.model = ActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.ppo.lr)

        self.lambda_dual = float(self.lagrangian.lambda_init)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, log_prob_t, value_t = self.model.act(obs_t)
        return int(action_t.item()), float(log_prob_t.item()), float(value_t.item())

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        rewards = batch["rewards"]
        costs = batch["costs"]
        dones = batch["dones"]

        penalized = rewards - float(self.lambda_dual) * costs
        advantages, returns = _gae(penalized, batch["values"], dones, self.ppo.gamma, self.ppo.gae_lambda)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        n = obs_t.shape[0]
        idx = np.arange(n)

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0

        for _epoch in range(self.ppo.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.ppo.minibatch_size):
                mb = idx[start : start + self.ppo.minibatch_size]
                mb_obs = obs_t[mb]
                mb_actions = actions_t[mb]
                mb_old_log = old_log_probs_t[mb]
                mb_returns = returns_t[mb]
                mb_adv = adv_t[mb]

                new_log, entropy, value = self.model.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_log - mb_old_log)

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - self.ppo.clip_range, 1.0 + self.ppo.clip_range) * mb_adv
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                value_loss = torch.mean((mb_returns - value) ** 2)
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ppo.value_coef * value_loss + self.ppo.entropy_coef * entropy_loss

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo.max_grad_norm)
                self.optim.step()

                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                entropy_acc += float(torch.mean(entropy).item())

        # Dual update (projected ascent).
        mean_cost = float(np.mean(costs))
        self.lambda_dual = float(
            np.clip(
                self.lambda_dual + self.lagrangian.lambda_lr * (mean_cost - self.lagrangian.cost_limit),
                0.0,
                self.lagrangian.lambda_max,
            )
        )

        updates = max(1, self.ppo.update_epochs * max(1, n // self.ppo.minibatch_size))
        return {
            "lambda": float(self.lambda_dual),
            "mean_cost": mean_cost,
            "mean_reward": float(np.mean(rewards)),
            "policy_loss": policy_loss_acc / updates,
            "value_loss": value_loss_acc / updates,
            "entropy": entropy_acc / updates,
        }

