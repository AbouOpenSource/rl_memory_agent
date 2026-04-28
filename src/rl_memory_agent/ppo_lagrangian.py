from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

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
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class LagrangianConfig:
    cost_limit: float = 0.0
    # Optional multi-constraint limits. When provided, it overrides `cost_limit`.
    cost_limits: Tuple[float, ...] | None = None
    lambda_init: float = 0.0
    lambda_lr: float = 0.05
    lambda_max: float = 100.0

    def limits_array(self) -> np.ndarray:
        limits = self.cost_limits if self.cost_limits is not None else (self.cost_limit,)
        arr = np.asarray(limits, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("cost limits must be a non-empty 1D sequence")
        return arr


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
    def __init__(self, obs_dim: int, size: int, *, n_costs: int = 1) -> None:
        self.size = int(size)
        self.n_costs = int(n_costs)
        if self.n_costs <= 0:
            raise ValueError("n_costs must be >= 1")
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.size,), dtype=np.int64)
        self.log_probs = np.zeros((self.size,), dtype=np.float32)
        self.values = np.zeros((self.size,), dtype=np.float32)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.costs = np.zeros((self.size, self.n_costs), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(
        self,
        *,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        cost: float | Sequence[float] | np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = int(action)
        self.log_probs[self.pos] = float(log_prob)
        self.values[self.pos] = float(value)
        self.rewards[self.pos] = float(reward)
        cost_arr = np.asarray(cost, dtype=np.float32)
        if cost_arr.ndim == 0:
            cost_arr = np.full((self.n_costs,), float(cost_arr), dtype=np.float32)
        elif cost_arr.ndim != 1:
            raise ValueError("cost must be a scalar or a 1D vector")
        if int(cost_arr.shape[0]) != int(self.n_costs):
            raise ValueError(f"cost has {int(cost_arr.shape[0])} dims, expected n_costs={int(self.n_costs)}")
        self.costs[self.pos] = cost_arr
        self.dones[self.pos] = float(done)
        self.pos += 1
        if self.pos >= self.size:
            self.full = True
            self.pos = 0

    def reset(self) -> None:
        self.pos = 0
        self.full = False

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


def _gae(
    returns_rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    *,
    last_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(returns_rewards, dtype=np.float32)
    last_adv = 0.0
    last_value_t = float(last_value)
    for t in reversed(range(len(returns_rewards))):
        nonterminal = 1.0 - dones[t]
        delta = returns_rewards[t] + gamma * last_value_t * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
        last_value_t = float(values[t])
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

        self.cost_limits = self.lagrangian.limits_array()
        self.lambda_dual = np.full((int(self.cost_limits.shape[0]),), float(self.lagrangian.lambda_init), dtype=np.float32)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, log_prob_t, value_t = self.model.act(obs_t)
        return int(action_t.item()), float(log_prob_t.item()), float(value_t.item())

    @torch.no_grad()
    def predict_value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _logits, value = self.model.forward(obs_t)
        return float(value.item())

    @torch.no_grad()
    def log_prob_value(self, obs: np.ndarray, action: int) -> Tuple[float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor([int(action)], dtype=torch.int64, device=self.device)
        logits, value = self.model.forward(obs_t)
        dist = Categorical(logits=logits)
        return float(dist.log_prob(action_t).item()), float(value.item())

    def update(self, batch: Dict[str, np.ndarray], *, last_value: float = 0.0) -> Dict[str, float | list[float]]:
        rewards = batch["rewards"]
        costs = batch["costs"]
        dones = batch["dones"]

        costs_arr = np.asarray(costs, dtype=np.float32)
        if costs_arr.ndim == 1:
            costs_arr = costs_arr.reshape(-1, 1)
        if costs_arr.ndim != 2:
            raise ValueError("batch['costs'] must be 1D or 2D array")

        k = int(costs_arr.shape[1])
        if int(self.lambda_dual.shape[0]) != k:
            # Broadcast scalar config to a vector cost setting (useful for backward compatibility).
            if int(self.lambda_dual.shape[0]) == 1 and k > 1:
                self.lambda_dual = np.full((k,), float(self.lambda_dual[0]), dtype=np.float32)
                self.cost_limits = np.full((k,), float(self.cost_limits[0]), dtype=np.float32)
            else:
                raise ValueError(f"lambda_dual has dim {int(self.lambda_dual.shape[0])}, but costs have dim {k}")

        penalized = rewards - costs_arr @ self.lambda_dual
        advantages, returns = _gae(
            penalized,
            batch["values"],
            dones,
            self.ppo.gamma,
            self.ppo.gae_lambda,
            last_value=float(last_value),
        )

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
        mean_cost = np.mean(costs_arr, axis=0)
        self.lambda_dual = np.clip(
            self.lambda_dual + float(self.lagrangian.lambda_lr) * (mean_cost - self.cost_limits),
            0.0,
            float(self.lagrangian.lambda_max),
        ).astype(np.float32)

        num_minibatches = max(1, int(math.ceil(n / self.ppo.minibatch_size)))
        updates = self.ppo.update_epochs * num_minibatches
        lambda_out: float | list[float]
        mean_cost_out: float | list[float]
        if int(self.lambda_dual.shape[0]) == 1:
            lambda_out = float(self.lambda_dual[0])
            mean_cost_out = float(mean_cost[0])
        else:
            lambda_out = [float(x) for x in self.lambda_dual.tolist()]
            mean_cost_out = [float(x) for x in mean_cost.tolist()]

        return {
            "lambda": lambda_out,
            "mean_cost": mean_cost_out,
            "mean_reward": float(np.mean(rewards)),
            "policy_loss": policy_loss_acc / updates,
            "value_loss": value_loss_acc / updates,
            "entropy": entropy_acc / updates,
        }

    def checkpoint_payload(self, *, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "checkpoint_version": 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "lambda_dual": self.lambda_dual.tolist(),
            "cost_limits": self.cost_limits.tolist(),
            "ppo_config": self.ppo.__dict__.copy(),
            "lagrangian_config": self.lagrangian.__dict__.copy(),
        }
        if extra is not None:
            payload["extra"] = dict(extra)
        return payload

    def save_checkpoint(self, path: str, *, extra: Dict[str, Any] | None = None) -> None:
        torch.save(self.checkpoint_payload(extra=extra), path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])

        optim_state = payload.get("optimizer_state_dict")
        if optim_state is not None:
            self.optim.load_state_dict(optim_state)

        lambda_dual = payload.get("lambda_dual")
        if lambda_dual is not None:
            self.lambda_dual = np.asarray(lambda_dual, dtype=np.float32)

        cost_limits = payload.get("cost_limits")
        if cost_limits is not None:
            self.cost_limits = np.asarray(cost_limits, dtype=np.float32)

        extra = payload.get("extra")
        return dict(extra) if isinstance(extra, dict) else {}
