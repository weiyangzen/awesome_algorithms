"""Minimal runnable MVP for Actor-Critic (MATH-0413).

This demo implements an on-policy one-step Actor-Critic method from scratch
using PyTorch on a tiny custom episodic environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class LineWorld:
    """A tiny 1D environment with two terminal states.

    States: 0 ... n_states-1
    Start: center state
    Actions: 0=left, 1=right
    Reward:
    - reach left terminal -> -1.0
    - reach right terminal -> +1.0
    - otherwise -> step_penalty
    """

    def __init__(self, n_states: int = 7, max_steps: int = 20, step_penalty: float = -0.01) -> None:
        if n_states < 3:
            raise ValueError("n_states must be at least 3.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self.n_states = n_states
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.start_state = n_states // 2
        self.state = self.start_state
        self.steps = 0

    def reset(self) -> int:
        self.state = self.start_state
        self.steps = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if action not in (0, 1):
            raise ValueError("action must be 0 (left) or 1 (right).")

        self.steps += 1
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.n_states - 1, self.state + 1)

        done = False
        reward = self.step_penalty

        if self.state == 0:
            done = True
            reward = -1.0
        elif self.state == self.n_states - 1:
            done = True
            reward = 1.0
        elif self.steps >= self.max_steps:
            done = True
            reward = -0.2

        return self.state, reward, done, {}


def one_hot_state(state: int, n_states: int) -> torch.Tensor:
    vec = torch.zeros(n_states, dtype=torch.float32)
    vec[state] = 1.0
    return vec


class ActorNet(nn.Module):
    def __init__(self, n_states: int, n_actions: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticNet(nn.Module):
    def __init__(self, n_states: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainStats:
    episode_returns: np.ndarray
    success_flags: np.ndarray
    actor_losses: np.ndarray
    critic_losses: np.ndarray


def train_actor_critic(
    env: LineWorld,
    episodes: int = 500,
    gamma: float = 0.98,
    actor_lr: float = 1e-2,
    critic_lr: float = 2e-2,
    entropy_coef: float = 1e-3,
    seed: int = 413,
) -> tuple[ActorNet, CriticNet, TrainStats]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")

    np.random.seed(seed)
    torch.manual_seed(seed)

    actor = ActorNet(n_states=env.n_states)
    critic = CriticNet(n_states=env.n_states)

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    episode_returns = np.zeros(episodes, dtype=np.float64)
    success_flags = np.zeros(episodes, dtype=np.int64)
    actor_losses = np.zeros(episodes, dtype=np.float64)
    critic_losses = np.zeros(episodes, dtype=np.float64)

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        ep_actor_loss = 0.0
        ep_critic_loss = 0.0

        while not done:
            s_vec = one_hot_state(state, env.n_states)
            logits = actor(s_vec)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, done, _ = env.step(int(action.item()))
            ep_return += reward

            with torch.no_grad():
                if done:
                    next_v = torch.tensor(0.0, dtype=torch.float32)
                else:
                    next_v = critic(one_hot_state(next_state, env.n_states))

            v = critic(s_vec)
            td_target = torch.tensor(reward, dtype=torch.float32) + gamma * next_v
            advantage = td_target - v

            # Actor: maximize E[log pi(a|s) * advantage + entropy bonus]
            actor_loss = -(dist.log_prob(action) * advantage.detach() + entropy_coef * dist.entropy())

            # Critic: one-step TD regression.
            critic_loss = 0.5 * advantage.pow(2)

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            ep_actor_loss += float(actor_loss.item())
            ep_critic_loss += float(critic_loss.item())
            state = next_state

        episode_returns[ep] = ep_return
        success_flags[ep] = int(state == env.n_states - 1)
        actor_losses[ep] = ep_actor_loss
        critic_losses[ep] = ep_critic_loss

    stats = TrainStats(
        episode_returns=episode_returns,
        success_flags=success_flags,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
    )
    return actor, critic, stats


def evaluate_policy(actor: ActorNet, env: LineWorld, episodes: int = 200) -> tuple[float, float]:
    returns = []
    success = 0

    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                s_vec = one_hot_state(state, env.n_states)
                logits = actor(s_vec)
                action = int(torch.argmax(logits).item())
                state, reward, done, _ = env.step(action)
                ep_return += reward
            returns.append(ep_return)
            if state == env.n_states - 1:
                success += 1

    return float(np.mean(returns)), success / episodes


def main() -> None:
    print("Actor-Critic MVP (MATH-0413)")
    print("=" * 72)

    env = LineWorld(n_states=7, max_steps=20, step_penalty=-0.01)

    actor, critic, stats = train_actor_critic(
        env=env,
        episodes=500,
        gamma=0.98,
        actor_lr=1e-2,
        critic_lr=2e-2,
        entropy_coef=1e-3,
        seed=413,
    )

    eval_avg_return, eval_success_rate = evaluate_policy(actor, env, episodes=200)

    df = pd.DataFrame(
        {
            "episode": np.arange(1, stats.episode_returns.shape[0] + 1),
            "return": stats.episode_returns,
            "success": stats.success_flags,
            "actor_loss": stats.actor_losses,
            "critic_loss": stats.critic_losses,
        }
    )
    df["return_ma20"] = df["return"].rolling(window=20, min_periods=1).mean()
    df["success_rate_ma20"] = df["success"].rolling(window=20, min_periods=1).mean()

    print(f"Training episodes: {len(df)}")
    print(f"Final 20-episode mean return: {df['return'].tail(20).mean():.4f}")
    print(f"Final 20-episode success rate: {df['success'].tail(20).mean():.4f}")
    print(f"Deterministic eval mean return (200 eps): {eval_avg_return:.4f}")
    print(f"Deterministic eval success rate (200 eps): {eval_success_rate:.4f}")
    print("-" * 72)
    print("Last 10 training episodes:")
    print(
        df.tail(10)[
            ["episode", "return", "success", "return_ma20", "success_rate_ma20"]
        ].to_string(index=False, float_format=lambda x: f"{x:8.4f}")
    )
    print("-" * 72)

    # Basic sanity checks for reproducible MVP behavior.
    assert np.all(np.isfinite(stats.episode_returns))
    assert np.all(np.isfinite(stats.actor_losses))
    assert np.all(np.isfinite(stats.critic_losses))
    assert 0.0 <= eval_success_rate <= 1.0
    # This environment is intentionally simple; trained policy should be strong.
    assert eval_success_rate >= 0.90, f"Unexpectedly low success rate: {eval_success_rate:.3f}"

    with torch.no_grad():
        center_state = env.start_state
        logits = actor(one_hot_state(center_state, env.n_states))
        probs = torch.softmax(logits, dim=-1).numpy()
    print(
        "Policy at start state (left, right): "
        f"[{probs[0]:.4f}, {probs[1]:.4f}]"
    )

    # Critic sanity check: value at right-side neighbor should be higher than left-side neighbor.
    with torch.no_grad():
        v_left = float(critic(one_hot_state(env.start_state - 1, env.n_states)).item())
        v_right = float(critic(one_hot_state(env.start_state + 1, env.n_states)).item())
    print(f"Critic values near center: left={v_left:.4f}, right={v_right:.4f}")
    assert v_right > v_left

    print("All checks passed.")


if __name__ == "__main__":
    main()
