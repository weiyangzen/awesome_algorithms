"""Policy Gradient (REINFORCE) minimal runnable MVP.

This script trains a stochastic policy on a tiny custom episodic MDP
without any interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class PGConfig:
    n_states: int = 7
    max_steps: int = 12
    hidden_dim: int = 16
    gamma: float = 0.99
    lr: float = 0.02
    train_episodes: int = 600
    eval_episodes: int = 200
    baseline_momentum: float = 0.90
    grad_clip: float = 5.0
    seed: int = 2026


class LineWorldEnv:
    """1D chain environment with terminal rewards.

    States: 0 .. n_states-1
    Start: center
    Actions: 0=left, 1=right
    Terminal:
      - reach state 0      => reward -1
      - reach state n-1    => reward +1
      - timeout            => reward 0
    """

    def __init__(self, n_states: int, max_steps: int):
        if n_states < 3 or n_states % 2 == 0:
            raise ValueError("n_states must be an odd integer >= 3.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self.n_states = n_states
        self.max_steps = max_steps
        self.start_state = n_states // 2
        self.left_terminal = 0
        self.right_terminal = n_states - 1

        self._state = self.start_state
        self._steps = 0

    def reset(self) -> int:
        self._state = self.start_state
        self._steps = 0
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool]:
        if action not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action}.")

        self._steps += 1
        delta = -1 if action == 0 else 1
        self._state = int(np.clip(self._state + delta, self.left_terminal, self.right_terminal))

        if self._state == self.right_terminal:
            return self._state, 1.0, True
        if self._state == self.left_terminal:
            return self._state, -1.0, True
        if self._steps >= self.max_steps:
            return self._state, 0.0, True

        return self._state, 0.0, False


class PolicyNet(nn.Module):
    """State-index -> action logits."""

    def __init__(self, n_states: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(n_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        return self.net(state_index)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return torch.from_numpy(returns)


def sample_action(policy: PolicyNet, state: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
    state_tensor = torch.tensor([state], dtype=torch.long)
    logits = policy(state_tensor)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    probs = torch.softmax(logits, dim=-1)
    return int(action.item()), log_prob.squeeze(0), probs.squeeze(0)


def rollout_episode(
    env: LineWorldEnv,
    policy: PolicyNet,
) -> Tuple[List[torch.Tensor], List[float], float, int, float]:
    state = env.reset()
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []

    right_prob_at_start = float("nan")

    while True:
        action, log_prob, probs = sample_action(policy, state)
        if np.isnan(right_prob_at_start):
            right_prob_at_start = float(probs[1].item())

        next_state, reward, done = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state
        if done:
            break

    total_return = float(np.sum(rewards))
    return log_probs, rewards, total_return, len(rewards), right_prob_at_start


def train_reinforce(cfg: PGConfig) -> Tuple[PolicyNet, pd.DataFrame]:
    env = LineWorldEnv(n_states=cfg.n_states, max_steps=cfg.max_steps)
    policy = PolicyNet(n_states=cfg.n_states, hidden_dim=cfg.hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    moving_baseline = 0.0
    history_rows: List[Dict[str, float]] = []

    for episode in range(1, cfg.train_episodes + 1):
        log_probs, rewards, episode_return, steps, start_right_prob = rollout_episode(
            env=env,
            policy=policy,
        )

        returns = discounted_returns(rewards, gamma=cfg.gamma)
        advantage = returns - moving_baseline

        log_prob_tensor = torch.stack(log_probs)
        loss = -(log_prob_tensor * advantage.detach()).sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        moving_baseline = cfg.baseline_momentum * moving_baseline + (1.0 - cfg.baseline_momentum) * episode_return

        history_rows.append(
            {
                "episode": float(episode),
                "return": episode_return,
                "steps": float(steps),
                "loss": float(loss.item()),
                "baseline": float(moving_baseline),
                "start_right_prob": start_right_prob,
            }
        )

    history = pd.DataFrame(history_rows)
    history["return_ma20"] = history["return"].rolling(20, min_periods=1).mean()
    history["success"] = (history["return"] > 0.0).astype(float)
    history["success_ma20"] = history["success"].rolling(20, min_periods=1).mean()
    return policy, history


def evaluate_policy(
    policy: PolicyNet,
    n_states: int,
    max_steps: int,
    episodes: int,
    greedy: bool,
) -> Dict[str, float]:
    env = LineWorldEnv(n_states=n_states, max_steps=max_steps)

    returns: List[float] = []
    steps: List[int] = []
    successes = 0

    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            episode_rewards: List[float] = []
            t = 0
            while True:
                t += 1
                state_tensor = torch.tensor([state], dtype=torch.long)
                logits = policy(state_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

                if greedy:
                    action = int(torch.argmax(probs).item())
                else:
                    dist = torch.distributions.Categorical(probs=probs)
                    action = int(dist.sample().item())

                next_state, reward, done = env.step(action)
                episode_rewards.append(reward)
                state = next_state
                if done:
                    break

            total_return = float(np.sum(episode_rewards))
            returns.append(total_return)
            steps.append(t)
            if total_return > 0.0:
                successes += 1

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(successes / episodes),
        "avg_steps": float(np.mean(steps)),
    }


def print_training_summary(history: pd.DataFrame) -> None:
    print("\nTraining trajectory (head):")
    print(history.head(6).to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    print("\nTraining trajectory (tail):")
    print(history.tail(6).to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    best_ma20 = float(history["return_ma20"].max())
    final_ma20 = float(history["return_ma20"].iloc[-1])
    final_success_ma20 = float(history["success_ma20"].iloc[-1])
    print("\nKey training metrics:")
    print(f"  best return_ma20  : {best_ma20:.4f}")
    print(f"  final return_ma20 : {final_ma20:.4f}")
    print(f"  final success_ma20: {final_success_ma20:.4f}")


def main() -> None:
    cfg = PGConfig()
    set_global_seed(cfg.seed)

    # Baseline policy (randomly initialized, before training)
    cold_policy = PolicyNet(n_states=cfg.n_states, hidden_dim=cfg.hidden_dim)
    pre_eval_sample = evaluate_policy(
        policy=cold_policy,
        n_states=cfg.n_states,
        max_steps=cfg.max_steps,
        episodes=cfg.eval_episodes,
        greedy=False,
    )

    trained_policy, history = train_reinforce(cfg)

    post_eval_sample = evaluate_policy(
        policy=trained_policy,
        n_states=cfg.n_states,
        max_steps=cfg.max_steps,
        episodes=cfg.eval_episodes,
        greedy=False,
    )
    post_eval_greedy = evaluate_policy(
        policy=trained_policy,
        n_states=cfg.n_states,
        max_steps=cfg.max_steps,
        episodes=cfg.eval_episodes,
        greedy=True,
    )

    print("Policy Gradient MVP (REINFORCE on LineWorld)")
    print(
        "config:",
        {
            "n_states": cfg.n_states,
            "max_steps": cfg.max_steps,
            "gamma": cfg.gamma,
            "lr": cfg.lr,
            "train_episodes": cfg.train_episodes,
            "seed": cfg.seed,
        },
    )

    print("\nBefore training (sample policy):")
    print(pre_eval_sample)

    print("\nAfter training (sample policy):")
    print(post_eval_sample)

    print("\nAfter training (greedy policy):")
    print(post_eval_greedy)

    print_training_summary(history)


if __name__ == "__main__":
    main()
