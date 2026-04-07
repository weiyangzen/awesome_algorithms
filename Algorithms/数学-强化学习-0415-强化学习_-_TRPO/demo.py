"""Minimal runnable MVP for TRPO (MATH-0415).

This script implements a compact Trust Region Policy Optimization pipeline
for a small discrete-control environment from scratch:
- policy network + value network
- generalized advantage estimation (GAE)
- conjugate gradient solver
- Fisher-vector product
- backtracking line search with KL constraint
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class LineWorld:
    """A tiny 1D episodic MDP.

    States: 0...n_states-1, start from center.
    Actions: 0 -> left, 1 -> right.
    Reward: +1 at right terminal, -1 at left terminal, step penalty otherwise.
    """

    def __init__(self, n_states: int = 9, max_steps: int = 20, step_penalty: float = -0.01) -> None:
        if n_states < 3:
            raise ValueError("n_states must be >= 3.")
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
            raise ValueError("action must be 0/1.")
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


def one_hot_state(state: int, n_states: int) -> np.ndarray:
    vec = np.zeros(n_states, dtype=np.float32)
    vec[state] = 1.0
    return vec


class PolicyNet(nn.Module):
    def __init__(self, n_states: int, n_actions: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
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
class TRPOConfig:
    n_states: int = 9
    max_steps: int = 20
    step_penalty: float = -0.01
    train_iters: int = 35
    batch_size: int = 600
    gamma: float = 0.99
    lam: float = 0.97
    max_kl: float = 0.01
    cg_iters: int = 10
    damping: float = 0.1
    backtrack_iters: int = 10
    backtrack_coeff: float = 0.8
    value_lr: float = 1e-2
    value_updates: int = 25
    seed: int = 415


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_logits: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_returns: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_params[offset : offset + n].view_as(p))
        offset += n


def flat_grad(y: torch.Tensor, model: nn.Module, create_graph: bool = False) -> torch.Tensor:
    grads = torch.autograd.grad(y, model.parameters(), create_graph=create_graph, retain_graph=create_graph)
    return torch.cat([g.contiguous().view(-1) for g in grads])


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        out[i] = running
    return out


def collect_batch(env: LineWorld, policy: PolicyNet, value_net: ValueNet, cfg: TRPOConfig) -> Batch:
    states: list[np.ndarray] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    old_logits: list[np.ndarray] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []
    ep_returns: list[float] = []
    timesteps = 0

    while timesteps < cfg.batch_size:
        s = env.reset()
        done = False
        ep_return = 0.0
        traj_rewards: list[float] = []
        traj_values: list[float] = []
        traj_dones: list[bool] = []

        while not done:
            s_vec = one_hot_state(s, env.n_states)
            s_t = torch.from_numpy(s_vec)
            with torch.no_grad():
                logits = policy(s_t)
                dist = Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
                v = value_net(s_t)

            ns, r, done, _ = env.step(int(a.item()))

            states.append(s_vec)
            actions.append(int(a.item()))
            traj_rewards.append(float(r))
            traj_values.append(float(v.item()))
            traj_dones.append(bool(done))
            old_log_probs.append(float(logp.item()))
            old_logits.append(logits.numpy())

            ep_return += r
            timesteps += 1
            s = ns
            if done or timesteps >= cfg.batch_size:
                break

        ep_returns.append(ep_return)

        # Bootstrap value for GAE tail.
        if traj_dones[-1]:
            last_v = 0.0
        else:
            with torch.no_grad():
                last_v = float(value_net(torch.from_numpy(one_hot_state(s, env.n_states))).item())

        traj_rewards_arr = np.asarray(traj_rewards, dtype=np.float64)
        traj_values_arr = np.asarray(traj_values, dtype=np.float64)
        traj_dones_arr = np.asarray(traj_dones, dtype=np.float64)
        vals_ext = np.append(traj_values_arr, last_v)
        deltas = traj_rewards_arr + cfg.gamma * vals_ext[1:] * (1.0 - traj_dones_arr) - vals_ext[:-1]
        traj_advs = discount_cumsum(deltas, cfg.gamma * cfg.lam)
        traj_rets = traj_advs + traj_values_arr

        all_advantages.extend(traj_advs.tolist())
        all_returns.extend(traj_rets.tolist())

    adv = np.asarray(all_advantages, dtype=np.float64)
    ret = np.asarray(all_returns, dtype=np.float64)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    states_t = torch.tensor(np.array(states, dtype=np.float32))
    actions_t = torch.tensor(np.array(actions, dtype=np.int64))
    old_log_probs_t = torch.tensor(np.array(old_log_probs, dtype=np.float32))
    old_logits_t = torch.tensor(np.array(old_logits, dtype=np.float32))
    returns_t = torch.tensor(ret.astype(np.float32))
    adv_t = torch.tensor(adv.astype(np.float32))

    return Batch(
        states=states_t,
        actions=actions_t,
        old_log_probs=old_log_probs_t,
        old_logits=old_logits_t,
        returns=returns_t,
        advantages=adv_t,
        episode_returns=np.array(ep_returns, dtype=np.float64),
    )


def surrogate_loss(policy: PolicyNet, batch: Batch) -> torch.Tensor:
    logits = policy(batch.states)
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(batch.actions)
    ratio = torch.exp(log_probs - batch.old_log_probs)
    return torch.mean(ratio * batch.advantages)


def mean_kl(policy: PolicyNet, batch: Batch) -> torch.Tensor:
    new_dist = Categorical(logits=policy(batch.states))
    old_dist = Categorical(logits=batch.old_logits)
    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist)
    return torch.mean(kl)


def conjugate_gradient(
    fvp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    cg_iters: int,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr_old = torch.dot(r, r)
    for _ in range(cg_iters):
        Ap = fvp(p)
        alpha = rr_old / (torch.dot(p, Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = torch.dot(r, r)
        if rr_new < residual_tol:
            break
        beta = rr_new / (rr_old + 1e-8)
        p = r + beta * p
        rr_old = rr_new
    return x


def trpo_step(policy: PolicyNet, batch: Batch, cfg: TRPOConfig) -> tuple[float, float, bool]:
    old_params = get_flat_params(policy).detach()

    surr = surrogate_loss(policy, batch)
    g = flat_grad(surr, policy, create_graph=False).detach()

    def fvp(v: torch.Tensor) -> torch.Tensor:
        kl = mean_kl(policy, batch)
        grad_kl = flat_grad(kl, policy, create_graph=True)
        kl_v = torch.dot(grad_kl, v)
        grad_grad_kl = torch.autograd.grad(kl_v, policy.parameters(), retain_graph=False)
        flat_hvp = torch.cat([x.contiguous().view(-1) for x in grad_grad_kl]).detach()
        return flat_hvp + cfg.damping * v

    step_dir = conjugate_gradient(fvp, g, cg_iters=cfg.cg_iters)
    shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
    step_scale = torch.sqrt(cfg.max_kl / (shs + 1e-8))
    full_step = step_scale * step_dir

    old_surr = surr.item()
    accepted = False
    final_kl = 0.0

    for i in range(cfg.backtrack_iters):
        frac = cfg.backtrack_coeff**i
        candidate = old_params + frac * full_step
        set_flat_params(policy, candidate)
        new_surr = surrogate_loss(policy, batch).item()
        kl_val = float(mean_kl(policy, batch).item())
        if np.isfinite(new_surr) and np.isfinite(kl_val) and kl_val <= cfg.max_kl and new_surr > old_surr:
            accepted = True
            final_kl = kl_val
            break

    if not accepted:
        set_flat_params(policy, old_params)
        final_kl = float(mean_kl(policy, batch).item())

    improve = float(surrogate_loss(policy, batch).item() - old_surr)
    return improve, final_kl, accepted


def train_value_function(value_net: ValueNet, batch: Batch, cfg: TRPOConfig) -> float:
    opt = optim.Adam(value_net.parameters(), lr=cfg.value_lr)
    loss_val = 0.0
    for _ in range(cfg.value_updates):
        pred = value_net(batch.states)
        loss = torch.mean((pred - batch.returns) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = float(loss.item())
    return loss_val


def evaluate_policy(policy: PolicyNet, env: LineWorld, episodes: int = 200, greedy: bool = True) -> tuple[float, float, float]:
    returns = []
    success = 0
    steps = 0.0
    with torch.no_grad():
        for _ in range(episodes):
            s = env.reset()
            done = False
            ep_ret = 0.0
            ep_steps = 0
            while not done:
                logits = policy(torch.from_numpy(one_hot_state(s, env.n_states)))
                if greedy:
                    a = int(torch.argmax(logits).item())
                else:
                    a = int(Categorical(logits=logits).sample().item())
                s, r, done, _ = env.step(a)
                ep_ret += r
                ep_steps += 1
            returns.append(ep_ret)
            steps += ep_steps
            if s == env.n_states - 1:
                success += 1
    return float(np.mean(returns)), success / episodes, steps / episodes


def main() -> None:
    cfg = TRPOConfig()
    set_seed(cfg.seed)
    env = LineWorld(n_states=cfg.n_states, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty)

    policy = PolicyNet(n_states=cfg.n_states)
    value_net = ValueNet(n_states=cfg.n_states)

    pre_ret, pre_success, pre_steps = evaluate_policy(policy, env, episodes=150, greedy=True)

    logs: list[dict[str, float]] = []
    for it in range(1, cfg.train_iters + 1):
        batch = collect_batch(env, policy, value_net, cfg)
        v_loss = train_value_function(value_net, batch, cfg)
        improve, kl, accepted = trpo_step(policy, batch, cfg)

        with torch.no_grad():
            start_logits = policy(torch.from_numpy(one_hot_state(env.start_state, env.n_states)))
            start_probs = torch.softmax(start_logits, dim=-1).numpy()

        logs.append(
            {
                "iter": it,
                "batch_mean_return": float(batch.episode_returns.mean()),
                "batch_success_rate": float(np.mean(batch.episode_returns > 0)),
                "value_loss": v_loss,
                "surr_improve": improve,
                "mean_kl": kl,
                "line_search_accept": float(accepted),
                "start_left_prob": float(start_probs[0]),
                "start_right_prob": float(start_probs[1]),
            }
        )

    post_ret, post_success, post_steps = evaluate_policy(policy, env, episodes=200, greedy=True)
    post_sample_ret, post_sample_success, _ = evaluate_policy(policy, env, episodes=200, greedy=False)

    df = pd.DataFrame(logs)
    df["return_ma5"] = df["batch_mean_return"].rolling(window=5, min_periods=1).mean()
    df["success_ma5"] = df["batch_success_rate"].rolling(window=5, min_periods=1).mean()

    print("TRPO MVP (MATH-0415)")
    print("=" * 76)
    print(f"Train iters: {cfg.train_iters}, batch_size: {cfg.batch_size}")
    print(
        f"Pre-train greedy eval: return={pre_ret:.4f}, "
        f"success={pre_success:.4f}, avg_steps={pre_steps:.2f}"
    )
    print(
        f"Post-train greedy eval: return={post_ret:.4f}, "
        f"success={post_success:.4f}, avg_steps={post_steps:.2f}"
    )
    print(
        f"Post-train sample eval: return={post_sample_ret:.4f}, "
        f"success={post_sample_success:.4f}"
    )
    print("-" * 76)
    print("Last 8 iterations:")
    print(
        df.tail(8)[
            [
                "iter",
                "batch_mean_return",
                "batch_success_rate",
                "mean_kl",
                "line_search_accept",
                "start_right_prob",
                "return_ma5",
                "success_ma5",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:8.4f}")
    )
    print("-" * 76)

    # Sanity checks for a deterministic minimal benchmark.
    assert np.all(np.isfinite(df["batch_mean_return"].values))
    assert np.all(np.isfinite(df["mean_kl"].values))
    assert (df["mean_kl"].values <= cfg.max_kl * 1.2).all()
    assert post_success >= 0.90, f"Unexpected low greedy success after TRPO: {post_success:.3f}"

    with torch.no_grad():
        center_logits = policy(torch.from_numpy(one_hot_state(env.start_state, env.n_states)))
        center_probs = torch.softmax(center_logits, dim=-1).numpy()
    print(f"Policy at start state (left, right): [{center_probs[0]:.4f}, {center_probs[1]:.4f}]")
    print("All checks passed.")


if __name__ == "__main__":
    main()
