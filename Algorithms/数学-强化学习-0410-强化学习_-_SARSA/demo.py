"""Minimal runnable MVP for SARSA (MATH-0410).

This script implements tabular SARSA on a custom CliffWalking environment.
No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SarsaConfig:
    rows: int = 4
    cols: int = 12
    alpha: float = 0.5
    gamma: float = 1.0
    epsilon_start: float = 0.15
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    train_episodes: int = 700
    eval_episodes: int = 250
    max_steps: int = 200
    seed: int = 410


class CliffWalkingEnv:
    """Classic CliffWalking-like environment.

    Grid size: rows x cols (default 4x12)
    Start: bottom-left
    Goal: bottom-right
    Cliff: bottom row cells between start and goal

    Actions: 0=up, 1=right, 2=down, 3=left
    Reward:
      - each regular step: -1
      - fall into cliff: -100 and reset to start (episode continues)
      - reaching goal: -1 and episode terminates
    """

    def __init__(self, rows: int, cols: int, max_steps: int) -> None:
        if rows < 3 or cols < 4:
            raise ValueError("rows/cols too small for cliff setting.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps

        self.start_pos = (rows - 1, 0)
        self.goal_pos = (rows - 1, cols - 1)

        self.n_states = rows * cols
        self.n_actions = 4

        self._state = self._pos_to_state(self.start_pos)
        self._steps = 0

    def _pos_to_state(self, pos: tuple[int, int]) -> int:
        r, c = pos
        return r * self.cols + c

    def _state_to_pos(self, state: int) -> tuple[int, int]:
        return divmod(state, self.cols)

    def _is_cliff(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return r == self.rows - 1 and 1 <= c <= self.cols - 2

    def is_goal_state(self, state: int) -> bool:
        return self._state_to_pos(state) == self.goal_pos

    def reset(self) -> int:
        self._state = self._pos_to_state(self.start_pos)
        self._steps = 0
        return self._state

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        if action not in (0, 1, 2, 3):
            raise ValueError(f"Invalid action: {action}")

        self._steps += 1
        r, c = self._state_to_pos(self._state)

        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            c = min(self.cols - 1, c + 1)
        elif action == 2:
            r = min(self.rows - 1, r + 1)
        else:
            c = max(0, c - 1)

        next_pos = (r, c)
        reward = -1.0
        done = False
        fell_cliff = False

        if self._is_cliff(next_pos):
            reward = -100.0
            fell_cliff = True
            next_pos = self.start_pos
        elif next_pos == self.goal_pos:
            done = True

        if self._steps >= self.max_steps:
            done = True

        self._state = self._pos_to_state(next_pos)
        info = {"fell_cliff": fell_cliff}
        return self._state, reward, done, info


ARROW = {0: "^", 1: ">", 2: "v", 3: "<"}


def epsilon_by_episode(cfg: SarsaConfig, episode_idx: int) -> float:
    eps = cfg.epsilon_start * (cfg.epsilon_decay ** episode_idx)
    return max(cfg.epsilon_end, eps)


def epsilon_greedy_action(q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q_table.shape[1]))

    q_values = q_table[state]
    max_q = float(np.max(q_values))
    best_actions = np.flatnonzero(np.isclose(q_values, max_q))
    return int(rng.choice(best_actions))


def train_sarsa(cfg: SarsaConfig) -> tuple[np.ndarray, pd.DataFrame, CliffWalkingEnv]:
    env = CliffWalkingEnv(rows=cfg.rows, cols=cfg.cols, max_steps=cfg.max_steps)
    rng = np.random.default_rng(cfg.seed)

    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    rows: list[dict[str, float]] = []

    for ep in range(cfg.train_episodes):
        epsilon = epsilon_by_episode(cfg, ep)
        state = env.reset()
        action = epsilon_greedy_action(q_table, state, epsilon, rng)

        total_reward = 0.0
        steps = 0
        cliff_hits = 0

        while True:
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            cliff_hits += int(info["fell_cliff"])

            if done:
                td_target = reward
                td_error = td_target - q_table[state, action]
                q_table[state, action] += cfg.alpha * td_error
                state = next_state
                break

            next_action = epsilon_greedy_action(q_table, next_state, epsilon, rng)
            td_target = reward + cfg.gamma * q_table[next_state, next_action]
            td_error = td_target - q_table[state, action]
            q_table[state, action] += cfg.alpha * td_error

            state = next_state
            action = next_action

        rows.append(
            {
                "episode": float(ep + 1),
                "epsilon": float(epsilon),
                "return": float(total_reward),
                "steps": float(steps),
                "cliff_hits": float(cliff_hits),
                "goal_reached": float(env.is_goal_state(state)),
            }
        )

    log_df = pd.DataFrame(rows)
    log_df["return_ma30"] = log_df["return"].rolling(30, min_periods=1).mean()
    log_df["goal_rate_ma30"] = log_df["goal_reached"].rolling(30, min_periods=1).mean()
    log_df["cliff_hits_ma30"] = log_df["cliff_hits"].rolling(30, min_periods=1).mean()

    return q_table, log_df, env


def evaluate_policy(
    env: CliffWalkingEnv,
    q_table: np.ndarray,
    episodes: int,
    seed: int,
    epsilon: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    returns: list[float] = []
    steps_list: list[int] = []
    cliff_hits_list: list[int] = []
    goals = 0
    sample_path: list[tuple[int, int]] | None = None

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        cliff_hits = 0
        path = [env._state_to_pos(state)]

        while True:
            action = epsilon_greedy_action(q_table, state, epsilon, rng)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            cliff_hits += int(info["fell_cliff"])
            path.append(env._state_to_pos(next_state))

            state = next_state
            if done:
                break

        returns.append(total_reward)
        steps_list.append(steps)
        cliff_hits_list.append(cliff_hits)
        if env.is_goal_state(state):
            goals += 1
        if ep == 0:
            sample_path = path

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sem = std_return / np.sqrt(len(returns)) if len(returns) > 1 else 0.0

    if len(returns) > 1 and sem > 0.0:
        ci_low, ci_high = stats.t.interval(0.95, df=len(returns) - 1, loc=mean_return, scale=sem)
    else:
        ci_low, ci_high = mean_return, mean_return

    return {
        "avg_return": mean_return,
        "std_return": std_return,
        "return_ci95": (float(ci_low), float(ci_high)),
        "avg_steps": float(np.mean(steps_list)),
        "avg_cliff_hits": float(np.mean(cliff_hits_list)),
        "success_rate": float(goals / episodes),
        "sample_path": sample_path,
    }


def render_greedy_policy(env: CliffWalkingEnv, q_table: np.ndarray) -> str:
    lines: list[str] = []
    for r in range(env.rows):
        cells: list[str] = []
        for c in range(env.cols):
            pos = (r, c)
            if pos == env.start_pos:
                cells.append(" S ")
            elif pos == env.goal_pos:
                cells.append(" G ")
            elif env._is_cliff(pos):
                cells.append(" C ")
            else:
                state = env._pos_to_state(pos)
                action = int(np.argmax(q_table[state]))
                cells.append(f" {ARROW[action]} ")
        lines.append("".join(cells))
    return "\n".join(lines)


def summarize_path(path: list[tuple[int, int]] | None, max_len: int = 25) -> str:
    if not path:
        return "(empty)"

    shown = path[:max_len]
    text = " -> ".join(f"({r},{c})" for r, c in shown)
    if len(path) > max_len:
        text += " -> ..."
    return text


def compact_eval_view(result: dict[str, Any]) -> dict[str, Any]:
    view = dict(result)
    view["sample_path"] = summarize_path(result.get("sample_path"), max_len=20)
    return view


def main() -> None:
    cfg = SarsaConfig()

    # Evaluate untrained table (all zeros) with mild exploration as baseline.
    cold_env = CliffWalkingEnv(rows=cfg.rows, cols=cfg.cols, max_steps=cfg.max_steps)
    cold_q = np.zeros((cold_env.n_states, cold_env.n_actions), dtype=np.float64)
    pre_eval = evaluate_policy(
        env=cold_env,
        q_table=cold_q,
        episodes=cfg.eval_episodes,
        seed=cfg.seed + 1,
        epsilon=0.10,
    )

    q_table, train_log, env = train_sarsa(cfg)

    post_eval_greedy = evaluate_policy(
        env=env,
        q_table=q_table,
        episodes=cfg.eval_episodes,
        seed=cfg.seed + 2,
        epsilon=0.0,
    )

    post_eval_eps = evaluate_policy(
        env=env,
        q_table=q_table,
        episodes=cfg.eval_episodes,
        seed=cfg.seed + 3,
        epsilon=0.05,
    )

    print("SARSA MVP (Tabular, CliffWalking)")
    print("=" * 78)
    print(
        "config:",
        {
            "rows": cfg.rows,
            "cols": cfg.cols,
            "alpha": cfg.alpha,
            "gamma": cfg.gamma,
            "eps_start": cfg.epsilon_start,
            "eps_end": cfg.epsilon_end,
            "eps_decay": cfg.epsilon_decay,
            "train_episodes": cfg.train_episodes,
            "seed": cfg.seed,
        },
    )

    print("\nBefore training (epsilon=0.10):")
    print(compact_eval_view(pre_eval))

    print("\nAfter training (greedy epsilon=0.0):")
    print(compact_eval_view(post_eval_greedy))

    print("\nAfter training (behavior epsilon=0.05):")
    print(compact_eval_view(post_eval_eps))

    print("\nGreedy policy map:")
    print(render_greedy_policy(env, q_table))

    print("\nSample greedy trajectory (first eval episode):")
    print(summarize_path(post_eval_greedy["sample_path"]))

    print("\nTraining log tail (last 12 episodes):")
    print(
        train_log.tail(12)[
            [
                "episode",
                "epsilon",
                "return",
                "steps",
                "cliff_hits",
                "goal_reached",
                "return_ma30",
                "goal_rate_ma30",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:8.4f}")
    )

    # Sanity checks.
    assert np.all(np.isfinite(q_table))
    assert np.all(np.isfinite(train_log["return"].values))
    assert 0.0 <= post_eval_greedy["success_rate"] <= 1.0
    assert post_eval_greedy["success_rate"] >= 0.90, (
        f"Unexpectedly low greedy success rate: {post_eval_greedy['success_rate']:.3f}"
    )
    assert post_eval_greedy["avg_return"] > pre_eval["avg_return"] + 15.0

    start_state = env._pos_to_state(env.start_pos)
    start_q = q_table[start_state]
    print("\nQ-values at start state [up, right, down, left]:")
    print(np.array2string(start_q, precision=3, floatmode="fixed"))

    print("All checks passed.")


if __name__ == "__main__":
    main()
