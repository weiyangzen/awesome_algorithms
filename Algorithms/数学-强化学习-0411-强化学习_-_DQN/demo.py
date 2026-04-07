"""Minimal runnable MVP for DQN (MATH-0411).

This script implements Deep Q-Network (DQN) from scratch using PyTorch on a
small custom episodic environment. It includes replay buffer, target network,
epsilon-greedy exploration, and non-interactive training/evaluation output.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LineWorld:
    """A tiny 1D environment with terminal rewards.

    States: 0..n_states-1
    Start: center state
    Actions: 0=left, 1=right
    Reward:
    - reach state 0 -> -1.0
    - reach state n_states-1 -> +1.0
    - intermediate step -> step_penalty
    """

    def __init__(self, n_states: int = 7, max_steps: int = 20, step_penalty: float = -0.01) -> None:
        if n_states < 3:
            raise ValueError("n_states must be >= 3.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self.n_states = n_states
        self.n_actions = 2
        self.max_steps = max_steps
        self.step_penalty = float(step_penalty)
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

        reward = self.step_penalty
        done = False

        if self.state == 0:
            reward = -1.0
            done = True
        elif self.state == self.n_states - 1:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -0.2
            done = True

        return self.state, reward, done, {}


class ReplayBuffer:
    """Simple experience replay buffer for off-policy DQN updates."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.buffer: deque[tuple[int, int, float, int, float]] = deque(maxlen=capacity)

    def append(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(self.buffer) < batch_size:
            raise ValueError("not enough samples in replay buffer.")

        indices = rng.choice(len(self.buffer), size=batch_size, replace=False)
        transitions = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.asarray(states, dtype=np.int64),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.int64),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Tiny MLP that maps one-hot state vectors to action values Q(s, a)."""

    def __init__(self, n_states: int, n_actions: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    n_states: int = 7
    max_steps: int = 20
    step_penalty: float = -0.01
    gamma: float = 0.98
    lr: float = 3e-3
    batch_size: int = 32
    replay_capacity: int = 4000
    warmup_steps: int = 64
    train_episodes: int = 500
    target_update_interval: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1200
    hidden_dim: int = 32
    grad_clip: float = 5.0
    eval_episodes: int = 200
    seed: int = 411


@dataclass
class TrainStats:
    episode_returns: np.ndarray
    success_flags: np.ndarray
    episode_losses: np.ndarray
    epsilons: np.ndarray
    steps_per_episode: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot_state_tensor(state: int, n_states: int) -> torch.Tensor:
    idx = torch.tensor([state], dtype=torch.long)
    return F.one_hot(idx, num_classes=n_states).to(torch.float32)


def states_to_tensor(states: np.ndarray, n_states: int) -> torch.Tensor:
    idx = torch.as_tensor(states, dtype=torch.long)
    return F.one_hot(idx, num_classes=n_states).to(torch.float32)


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    if cfg.epsilon_decay_steps <= 0:
        return cfg.epsilon_end
    ratio = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(
    q_net: QNetwork,
    state: int,
    epsilon: float,
    n_states: int,
    n_actions: int,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():
        q_values = q_net(one_hot_state_tensor(state, n_states))
    return int(torch.argmax(q_values, dim=1).item())


def optimize_dqn(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    cfg: DQNConfig,
    rng: np.random.Generator,
) -> float | None:
    min_required = max(cfg.batch_size, cfg.warmup_steps)
    if len(replay) < min_required:
        return None

    states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size, rng)

    states_t = states_to_tensor(states, cfg.n_states)
    next_states_t = states_to_tensor(next_states, cfg.n_states)
    actions_t = torch.as_tensor(actions, dtype=torch.long)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    dones_t = torch.as_tensor(dones, dtype=torch.float32)

    q_sa = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_max = target_net(next_states_t).max(dim=1).values
        td_target = rewards_t + cfg.gamma * (1.0 - dones_t) * next_q_max

    loss = F.smooth_l1_loss(q_sa, td_target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=cfg.grad_clip)
    optimizer.step()

    return float(loss.item())


def train_dqn(cfg: DQNConfig) -> tuple[QNetwork, TrainStats]:
    if cfg.train_episodes <= 0:
        raise ValueError("train_episodes must be positive.")
    if not (0.0 < cfg.gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = LineWorld(n_states=cfg.n_states, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty)

    q_net = QNetwork(env.n_states, env.n_actions, hidden_dim=cfg.hidden_dim)
    target_net = QNetwork(env.n_states, env.n_actions, hidden_dim=cfg.hidden_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(capacity=cfg.replay_capacity)

    episode_returns = np.zeros(cfg.train_episodes, dtype=np.float64)
    success_flags = np.zeros(cfg.train_episodes, dtype=np.int64)
    episode_losses = np.zeros(cfg.train_episodes, dtype=np.float64)
    epsilons = np.zeros(cfg.train_episodes, dtype=np.float64)
    steps_per_episode = np.zeros(cfg.train_episodes, dtype=np.int64)

    global_step = 0

    for ep in range(cfg.train_episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        losses: list[float] = []
        last_epsilon = cfg.epsilon_start
        ep_steps = 0

        while not done:
            epsilon = epsilon_by_step(global_step, cfg)
            last_epsilon = epsilon
            action = select_action(
                q_net=q_net,
                state=state,
                epsilon=epsilon,
                n_states=cfg.n_states,
                n_actions=env.n_actions,
                rng=rng,
            )

            next_state, reward, done, _ = env.step(action)
            replay.append(state, action, reward, next_state, done)

            loss = optimize_dqn(
                q_net=q_net,
                target_net=target_net,
                optimizer=optimizer,
                replay=replay,
                cfg=cfg,
                rng=rng,
            )
            if loss is not None:
                losses.append(loss)

            global_step += 1
            ep_steps += 1
            if global_step % cfg.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

            ep_return += reward
            state = next_state

        episode_returns[ep] = ep_return
        success_flags[ep] = int(state == env.n_states - 1)
        episode_losses[ep] = float(np.mean(losses)) if losses else 0.0
        epsilons[ep] = last_epsilon
        steps_per_episode[ep] = ep_steps

    stats = TrainStats(
        episode_returns=episode_returns,
        success_flags=success_flags,
        episode_losses=episode_losses,
        epsilons=epsilons,
        steps_per_episode=steps_per_episode,
    )
    return q_net, stats


def evaluate_policy(q_net: QNetwork, cfg: DQNConfig, episodes: int) -> tuple[float, float, float]:
    env = LineWorld(n_states=cfg.n_states, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty)
    returns: list[float] = []
    success = 0
    step_count = 0

    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            done = False
            ep_return = 0.0
            ep_steps = 0
            while not done:
                q_values = q_net(one_hot_state_tensor(state, cfg.n_states))
                action = int(torch.argmax(q_values, dim=1).item())
                state, reward, done, _ = env.step(action)
                ep_return += reward
                ep_steps += 1

            returns.append(ep_return)
            step_count += ep_steps
            if state == env.n_states - 1:
                success += 1

    return float(np.mean(returns)), success / episodes, step_count / episodes


def main() -> None:
    cfg = DQNConfig()

    print("DQN MVP (MATH-0411)")
    print("=" * 72)

    # Baseline greedy performance from untrained network.
    set_seed(cfg.seed)
    untrained_q = QNetwork(n_states=cfg.n_states, n_actions=2, hidden_dim=cfg.hidden_dim)
    pre_avg_return, pre_success_rate, pre_avg_steps = evaluate_policy(
        q_net=untrained_q,
        cfg=cfg,
        episodes=cfg.eval_episodes,
    )

    trained_q, stats = train_dqn(cfg)

    post_avg_return, post_success_rate, post_avg_steps = evaluate_policy(
        q_net=trained_q,
        cfg=cfg,
        episodes=cfg.eval_episodes,
    )

    df = pd.DataFrame(
        {
            "episode": np.arange(1, cfg.train_episodes + 1),
            "return": stats.episode_returns,
            "success": stats.success_flags,
            "loss": stats.episode_losses,
            "epsilon": stats.epsilons,
            "steps": stats.steps_per_episode,
        }
    )
    df["return_ma20"] = df["return"].rolling(window=20, min_periods=1).mean()
    df["success_ma20"] = df["success"].rolling(window=20, min_periods=1).mean()

    print(f"Train episodes: {cfg.train_episodes}")
    print(f"Replay warmup steps: {cfg.warmup_steps}")
    print(f"Final epsilon: {stats.epsilons[-1]:.4f}")
    print(f"Final 20-episode mean return: {df['return'].tail(20).mean():.4f}")
    print(f"Final 20-episode success rate: {df['success'].tail(20).mean():.4f}")
    print("-" * 72)
    print(
        "Pre-train greedy eval: "
        f"avg_return={pre_avg_return:.4f}, success_rate={pre_success_rate:.4f}, avg_steps={pre_avg_steps:.2f}"
    )
    print(
        "Post-train greedy eval: "
        f"avg_return={post_avg_return:.4f}, success_rate={post_success_rate:.4f}, avg_steps={post_avg_steps:.2f}"
    )
    print("-" * 72)
    print("Last 10 training episodes:")
    print(
        df.tail(10)[["episode", "return", "success", "loss", "epsilon", "return_ma20", "success_ma20"]].to_string(
            index=False,
            float_format=lambda x: f"{x:8.4f}",
        )
    )

    with torch.no_grad():
        center_q = trained_q(one_hot_state_tensor(cfg.n_states // 2, cfg.n_states)).squeeze(0).numpy()
    print("-" * 72)
    print(f"Q(start_state, left)  = {center_q[0]:.4f}")
    print(f"Q(start_state, right) = {center_q[1]:.4f}")

    # Basic quality checks for this toy task.
    assert np.all(np.isfinite(stats.episode_returns))
    assert np.all(np.isfinite(stats.episode_losses))
    assert 0.0 <= post_success_rate <= 1.0
    assert post_success_rate >= 0.90, f"Unexpectedly low success rate: {post_success_rate:.3f}"
    assert post_success_rate >= pre_success_rate
    assert center_q[1] > center_q[0], "Trained Q-values do not prefer moving right from start."

    print("All checks passed.")


if __name__ == "__main__":
    main()
