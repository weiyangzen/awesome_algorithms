"""Minimal runnable MVP for PPO (MATH-0414).

This script implements PPO-Clip from scratch with PyTorch on a tiny custom
environment. It is intentionally small and fully auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch.distributions import Categorical


class LineWorld:
    """A tiny 1D episodic environment.

    States: 0 .. n_states-1
    Start: center state
    Actions: 0=left, 1=right

    Rewards:
    - hit left terminal  -> -1.0
    - hit right terminal -> +1.0
    - otherwise          -> step_penalty
    """

    def __init__(self, n_states: int = 9, max_steps: int = 24, step_penalty: float = -0.01) -> None:
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


class StateFeaturizer:
    """One-hot state encoding + standardization via scikit-learn."""

    def __init__(self, n_states: int) -> None:
        self.n_states = n_states
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        basis = np.eye(n_states, dtype=np.float32)
        self.scaler.fit(basis)

    def transform(self, state: int) -> np.ndarray:
        vec = np.zeros((1, self.n_states), dtype=np.float32)
        vec[0, state] = 1.0
        out = self.scaler.transform(vec)
        return out.astype(np.float32)[0]


class ActorNet(nn.Module):
    def __init__(self, n_features: int, n_actions: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class PPOConfig:
    n_states: int = 9
    max_steps: int = 24
    step_penalty: float = -0.01

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2

    actor_lr: float = 3e-3
    critic_lr: float = 6e-3
    entropy_coef: float = 1e-2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    steps_per_batch: int = 512
    update_epochs: int = 8
    minibatch_size: int = 64
    train_iterations: int = 60

    eval_episodes: int = 300
    seed: int = 414


@dataclass
class RolloutBatch:
    states: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    values: np.ndarray
    episode_returns: np.ndarray
    episode_success: np.ndarray


@dataclass
class UpdateStats:
    actor_loss: float
    critic_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


def collect_rollout(
    env: LineWorld,
    actor: ActorNet,
    critic: CriticNet,
    featurizer: StateFeaturizer,
    cfg: PPOConfig,
) -> RolloutBatch:
    states: list[np.ndarray] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    values: list[float] = []

    episode_returns: list[float] = []
    episode_success: list[int] = []

    steps_collected = 0

    while steps_collected < cfg.steps_per_batch:
        state = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            state_vec_np = featurizer.transform(state)
            state_vec = torch.from_numpy(state_vec_np)

            with torch.no_grad():
                logits = actor(state_vec)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = float(dist.log_prob(action).item())
                value = float(critic(state_vec).item())

            next_state, reward, done, _ = env.step(int(action.item()))

            states.append(state_vec_np)
            actions.append(int(action.item()))
            old_log_probs.append(log_prob)
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(value)

            ep_return += float(reward)
            steps_collected += 1
            state = next_state

        episode_returns.append(ep_return)
        episode_success.append(int(state == env.n_states - 1))

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)

    advantages = np.zeros_like(rewards_arr)
    last_adv = 0.0

    for t in range(rewards_arr.shape[0] - 1, -1, -1):
        if t == rewards_arr.shape[0] - 1:
            next_value = 0.0
            next_non_terminal = 1.0 - dones_arr[t]
        else:
            next_value = values_arr[t + 1]
            next_non_terminal = 1.0 - dones_arr[t]

        delta = rewards_arr[t] + cfg.gamma * next_value * next_non_terminal - values_arr[t]
        last_adv = float(delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_adv)
        advantages[t] = last_adv

    returns = advantages + values_arr

    return RolloutBatch(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
        returns=returns.astype(np.float32),
        advantages=advantages.astype(np.float32),
        values=values_arr,
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        episode_success=np.asarray(episode_success, dtype=np.int64),
    )


def ppo_update(
    batch: RolloutBatch,
    actor: ActorNet,
    critic: CriticNet,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    cfg: PPOConfig,
) -> UpdateStats:
    states_t = torch.from_numpy(batch.states)
    actions_t = torch.from_numpy(batch.actions)
    old_log_probs_t = torch.from_numpy(batch.old_log_probs)
    returns_t = torch.from_numpy(batch.returns)

    adv_t = torch.from_numpy(batch.advantages)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    n = states_t.shape[0]

    actor_loss_values: list[float] = []
    critic_loss_values: list[float] = []
    entropy_values: list[float] = []
    kl_values: list[float] = []
    clip_fraction_values: list[float] = []

    for _ in range(cfg.update_epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            idx = indices[start : start + cfg.minibatch_size]
            mb_idx = torch.from_numpy(idx)

            s_mb = states_t[mb_idx]
            a_mb = actions_t[mb_idx]
            old_logp_mb = old_log_probs_t[mb_idx]
            ret_mb = returns_t[mb_idx]
            adv_mb = adv_t[mb_idx]

            logits = actor(s_mb)
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(a_mb)
            ratio = torch.exp(new_logp - old_logp_mb)

            surrogate1 = ratio * adv_mb
            surrogate2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            entropy = dist.entropy().mean()

            actor_opt.zero_grad()
            actor_objective = actor_loss - cfg.entropy_coef * entropy
            actor_objective.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

            value_pred = critic(s_mb)
            critic_loss = F.mse_loss(value_pred, ret_mb)

            critic_opt.zero_grad()
            (cfg.value_coef * critic_loss).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_opt.step()

            with torch.no_grad():
                approx_kl = (old_logp_mb - new_logp).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()

            actor_loss_values.append(float(actor_loss.item()))
            critic_loss_values.append(float(critic_loss.item()))
            entropy_values.append(float(entropy.item()))
            kl_values.append(float(approx_kl.item()))
            clip_fraction_values.append(float(clip_fraction.item()))

    return UpdateStats(
        actor_loss=float(np.mean(actor_loss_values)),
        critic_loss=float(np.mean(critic_loss_values)),
        entropy=float(np.mean(entropy_values)),
        approx_kl=float(np.mean(kl_values)),
        clip_fraction=float(np.mean(clip_fraction_values)),
    )


@dataclass
class EvalStats:
    mean_return: float
    success_rate: float
    ci95_low: float
    ci95_high: float


def evaluate_policy(
    env: LineWorld,
    actor: ActorNet,
    featurizer: StateFeaturizer,
    episodes: int,
) -> EvalStats:
    returns: list[float] = []
    success = 0

    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            done = False
            ep_return = 0.0

            while not done:
                state_vec = torch.from_numpy(featurizer.transform(state))
                logits = actor(state_vec)
                action = int(torch.argmax(logits).item())
                state, reward, done, _ = env.step(action)
                ep_return += float(reward)

            returns.append(ep_return)
            success += int(state == env.n_states - 1)

    returns_np = np.asarray(returns, dtype=np.float64)
    mean_return = float(returns_np.mean())

    if returns_np.size > 1:
        sem = stats.sem(returns_np)
        if np.isfinite(sem) and sem > 0.0:
            ci = stats.t.interval(
                confidence=0.95,
                df=returns_np.size - 1,
                loc=mean_return,
                scale=sem,
            )
            ci95_low = float(ci[0])
            ci95_high = float(ci[1])
        else:
            ci95_low = mean_return
            ci95_high = mean_return
    else:
        ci95_low = mean_return
        ci95_high = mean_return

    return EvalStats(
        mean_return=mean_return,
        success_rate=float(success / episodes),
        ci95_low=ci95_low,
        ci95_high=ci95_high,
    )


def train_ppo(cfg: PPOConfig) -> tuple[ActorNet, CriticNet, pd.DataFrame, StateFeaturizer]:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = LineWorld(n_states=cfg.n_states, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty)
    featurizer = StateFeaturizer(n_states=cfg.n_states)

    actor = ActorNet(n_features=cfg.n_states)
    critic = CriticNet(n_features=cfg.n_states)

    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    rows: list[dict[str, float]] = []

    for it in range(1, cfg.train_iterations + 1):
        batch = collect_rollout(env=env, actor=actor, critic=critic, featurizer=featurizer, cfg=cfg)
        update_stats = ppo_update(
            batch=batch,
            actor=actor,
            critic=critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            cfg=cfg,
        )

        rows.append(
            {
                "iteration": float(it),
                "batch_steps": float(batch.states.shape[0]),
                "batch_episode_return_mean": float(batch.episode_returns.mean()),
                "batch_episode_success_rate": float(batch.episode_success.mean()),
                "actor_loss": update_stats.actor_loss,
                "critic_loss": update_stats.critic_loss,
                "entropy": update_stats.entropy,
                "approx_kl": update_stats.approx_kl,
                "clip_fraction": update_stats.clip_fraction,
            }
        )

    df = pd.DataFrame(rows)
    df["return_ma10"] = df["batch_episode_return_mean"].rolling(10, min_periods=1).mean()
    df["success_ma10"] = df["batch_episode_success_rate"].rolling(10, min_periods=1).mean()

    return actor, critic, df, featurizer


def main() -> None:
    cfg = PPOConfig()

    print("PPO-Clip MVP (MATH-0414)")
    print("=" * 72)

    actor, critic, train_df, featurizer = train_ppo(cfg)

    env_eval = LineWorld(n_states=cfg.n_states, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty)
    eval_stats = evaluate_policy(env_eval, actor, featurizer, episodes=cfg.eval_episodes)

    with torch.no_grad():
        start_vec = torch.from_numpy(featurizer.transform(env_eval.start_state))
        start_probs = torch.softmax(actor(start_vec), dim=-1)
        start_right_prob = float(start_probs[1].item())

        state_values = []
        for s in range(cfg.n_states):
            s_vec = torch.from_numpy(featurizer.transform(s))
            state_values.append(float(critic(s_vec).item()))

    rho, _ = stats.spearmanr(np.arange(cfg.n_states), np.asarray(state_values, dtype=np.float64))
    rho = float(np.nan_to_num(rho, nan=1.0))

    print(f"Training iterations: {cfg.train_iterations}")
    print(f"Final return_ma10: {train_df['return_ma10'].iloc[-1]:.4f}")
    print(f"Final success_ma10: {train_df['success_ma10'].iloc[-1]:.4f}")
    print(f"Greedy eval mean return ({cfg.eval_episodes} eps): {eval_stats.mean_return:.4f}")
    print(f"Greedy eval success rate ({cfg.eval_episodes} eps): {eval_stats.success_rate:.4f}")
    print(f"Greedy eval return 95% CI: [{eval_stats.ci95_low:.4f}, {eval_stats.ci95_high:.4f}]")
    print(f"Start-state right-action probability: {start_right_prob:.4f}")
    print(f"Spearman(state_index, V(s)): {rho:.4f}")
    print("-" * 72)
    print("Last 8 training rows:")
    print(
        train_df.tail(8).to_string(
            index=False,
            formatters={
                "iteration": "{:.0f}".format,
                "batch_steps": "{:.0f}".format,
                "batch_episode_return_mean": "{:.4f}".format,
                "batch_episode_success_rate": "{:.4f}".format,
                "actor_loss": "{:.4f}".format,
                "critic_loss": "{:.4f}".format,
                "entropy": "{:.4f}".format,
                "approx_kl": "{:.4f}".format,
                "clip_fraction": "{:.4f}".format,
                "return_ma10": "{:.4f}".format,
                "success_ma10": "{:.4f}".format,
            },
        )
    )

    assert np.isfinite(train_df["actor_loss"]).all(), "Actor loss contains non-finite values."
    assert np.isfinite(train_df["critic_loss"]).all(), "Critic loss contains non-finite values."
    assert eval_stats.success_rate >= 0.90, "PPO did not reach expected success rate."
    assert start_right_prob >= 0.80, "Policy at start state is not sufficiently right-biased."
    assert rho >= 0.30, "Critic value ordering is weaker than expected."

    print("All checks passed.")


if __name__ == "__main__":
    main()
