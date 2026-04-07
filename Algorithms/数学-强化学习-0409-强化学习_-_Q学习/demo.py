"""Q-learning MVP on a small stochastic GridWorld.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


Action = int
State = int
Position = Tuple[int, int]

ACTION_DELTAS: Tuple[Position, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_SYMBOLS = {0: "^", 1: ">", 2: "v", 3: "<"}


@dataclass(frozen=True)
class GridWorldConfig:
    rows: int = 5
    cols: int = 7
    start: Position = (0, 0)
    goal: Position = (4, 6)
    walls: Tuple[Position, ...] = ((1, 1), (2, 1), (3, 3))
    traps: Tuple[Position, ...] = ((1, 3), (2, 3), (3, 1), (3, 4))
    step_reward: float = -1.0
    trap_reward: float = -14.0
    goal_reward: float = 24.0
    slip_prob: float = 0.08
    max_steps: int = 80


@dataclass
class TrainResult:
    q_table: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray


class GridWorld:
    def __init__(self, config: GridWorldConfig, seed: int = 2026) -> None:
        self.config = config
        self.rows = config.rows
        self.cols = config.cols
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.walls = set(config.walls)
        self.traps = set(config.traps)
        self.rng = np.random.default_rng(seed)
        self.agent_pos: Position = config.start
        self.steps = 0
        self._validate()

    def _validate(self) -> None:
        if self.rows <= 1 or self.cols <= 1:
            raise ValueError("rows and cols must both be > 1.")
        if not (0.0 <= self.config.slip_prob < 1.0):
            raise ValueError("slip_prob must be in [0, 1).")
        if self.config.max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        for pos in (self.config.start, self.config.goal):
            if not self._inside(pos):
                raise ValueError(f"Position out of bounds: {pos}")

        if self.config.start in self.walls:
            raise ValueError("start cannot be a wall.")
        if self.config.goal in self.walls:
            raise ValueError("goal cannot be a wall.")

    def _inside(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols

    def pos_to_state(self, pos: Position) -> State:
        return pos[0] * self.cols + pos[1]

    def state_to_pos(self, state: State) -> Position:
        return (state // self.cols, state % self.cols)

    def reset(self) -> State:
        self.agent_pos = self.config.start
        self.steps = 0
        return self.pos_to_state(self.agent_pos)

    def _transition(self, pos: Position, action: Action) -> Position:
        dr, dc = ACTION_DELTAS[action]
        nr = min(max(pos[0] + dr, 0), self.rows - 1)
        nc = min(max(pos[1] + dc, 0), self.cols - 1)
        nxt = (nr, nc)
        if nxt in self.walls:
            return pos
        return nxt

    def step(self, action: Action) -> Tuple[State, float, bool]:
        if not (0 <= action < self.n_actions):
            raise ValueError(f"Invalid action: {action}")

        applied_action = action
        if self.rng.random() < self.config.slip_prob:
            applied_action = int(self.rng.integers(0, self.n_actions))

        self.agent_pos = self._transition(self.agent_pos, applied_action)
        self.steps += 1

        reward = float(self.config.step_reward)
        done = False

        if self.agent_pos in self.traps:
            reward = float(self.config.trap_reward)
            done = True
        elif self.agent_pos == self.config.goal:
            reward = float(self.config.goal_reward)
            done = True
        elif self.steps >= self.config.max_steps:
            done = True

        return self.pos_to_state(self.agent_pos), reward, done


def epsilon_by_episode(
    episode: int,
    decay_episodes: int,
    eps_start: float,
    eps_end: float,
) -> float:
    if decay_episodes <= 1:
        return eps_end
    progress = min(1.0, float(episode) / float(decay_episodes - 1))
    return eps_start + progress * (eps_end - eps_start)


def select_action_epsilon_greedy(
    q_table: np.ndarray,
    state: State,
    epsilon: float,
    rng: np.random.Generator,
) -> Action:
    if rng.random() < epsilon:
        return int(rng.integers(0, q_table.shape[1]))

    q_values = q_table[state]
    best = np.max(q_values)
    candidates = np.flatnonzero(np.isclose(q_values, best))
    return int(rng.choice(candidates))


def q_learning(
    env: GridWorld,
    episodes: int = 900,
    alpha: float = 0.18,
    gamma: float = 0.97,
    eps_start: float = 1.00,
    eps_end: float = 0.05,
    eps_decay_ratio: float = 0.70,
    seed: int = 2026,
) -> TrainResult:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    decay_episodes = max(2, int(episodes * eps_decay_ratio))
    rng = np.random.default_rng(seed)

    q_table = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step_i in range(env.config.max_steps):
            epsilon = epsilon_by_episode(ep, decay_episodes, eps_start, eps_end)
            action = select_action_epsilon_greedy(q_table, state, epsilon, rng)
            next_state, reward, done = env.step(action)

            best_next = 0.0 if done else float(np.max(q_table[next_state]))
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

            if done:
                lengths.append(step_i + 1)
                break
        else:
            lengths.append(env.config.max_steps)

        returns.append(total_reward)

        if (ep + 1) % 100 == 0:
            recent = float(np.mean(returns[-100:]))
            print(
                f"[train] episode={ep + 1:04d}, epsilon={epsilon:.3f}, "
                f"avg_return(last_100)={recent:.3f}"
            )

    return TrainResult(
        q_table=q_table,
        episode_returns=np.asarray(returns, dtype=float),
        episode_lengths=np.asarray(lengths, dtype=int),
    )


def greedy_policy_from_q(q_table: np.ndarray) -> np.ndarray:
    return np.argmax(q_table, axis=1).astype(int)


def evaluate_policy(env: GridWorld, policy: np.ndarray, episodes: int = 300) -> Dict[str, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    total_return = 0.0
    success = 0
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        ep_return = 0.0

        for step_i in range(env.config.max_steps):
            action = int(policy[state])
            state, reward, done = env.step(action)
            ep_return += reward

            if done:
                total_steps += step_i + 1
                if env.agent_pos == env.config.goal:
                    success += 1
                break
        else:
            total_steps += env.config.max_steps

        total_return += ep_return

    return {
        "avg_return": total_return / episodes,
        "success_rate": success / episodes,
        "avg_steps": total_steps / episodes,
    }


def evaluate_random_policy(env: GridWorld, episodes: int = 300, seed: int = 7) -> Dict[str, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    rng = np.random.default_rng(seed)
    total_return = 0.0
    success = 0
    total_steps = 0

    for _ in range(episodes):
        env.reset()
        ep_return = 0.0

        for step_i in range(env.config.max_steps):
            action = int(rng.integers(0, env.n_actions))
            _, reward, done = env.step(action)
            ep_return += reward

            if done:
                total_steps += step_i + 1
                if env.agent_pos == env.config.goal:
                    success += 1
                break
        else:
            total_steps += env.config.max_steps

        total_return += ep_return

    return {
        "avg_return": total_return / episodes,
        "success_rate": success / episodes,
        "avg_steps": total_steps / episodes,
    }


def render_policy_map(env: GridWorld, policy: Sequence[int]) -> str:
    lines: List[str] = []

    for r in range(env.rows):
        row: List[str] = []
        for c in range(env.cols):
            pos = (r, c)

            if pos in env.walls:
                row.append("#")
            elif pos in env.traps:
                row.append("X")
            elif pos == env.config.goal:
                row.append("G")
            elif pos == env.config.start:
                row.append("S")
            else:
                state = env.pos_to_state(pos)
                row.append(ACTION_SYMBOLS[int(policy[state])])

        lines.append(" ".join(row))

    return "\n".join(lines)


def main() -> None:
    seed = 2026
    config = GridWorldConfig()

    train_env = GridWorld(config=config, seed=seed)
    eval_env = GridWorld(config=config, seed=seed + 1)
    random_env = GridWorld(config=config, seed=seed + 2)

    episodes = 900
    result = q_learning(
        env=train_env,
        episodes=episodes,
        alpha=0.18,
        gamma=0.97,
        eps_start=1.00,
        eps_end=0.05,
        eps_decay_ratio=0.70,
        seed=seed,
    )

    policy = greedy_policy_from_q(result.q_table)
    eval_metrics = evaluate_policy(eval_env, policy, episodes=300)
    random_metrics = evaluate_random_policy(random_env, episodes=300, seed=seed + 3)

    print("\n=== Final Evaluation ===")
    print(
        "[eval-greedy] "
        f"avg_return={eval_metrics['avg_return']:.3f}, "
        f"success_rate={eval_metrics['success_rate']:.3f}, "
        f"avg_steps={eval_metrics['avg_steps']:.3f}"
    )
    print(
        "[eval-random] "
        f"avg_return={random_metrics['avg_return']:.3f}, "
        f"success_rate={random_metrics['success_rate']:.3f}, "
        f"avg_steps={random_metrics['avg_steps']:.3f}"
    )

    improvement = eval_metrics["avg_return"] - random_metrics["avg_return"]
    print(f"[compare] improvement_vs_random={improvement:.3f}")

    print("\n=== Greedy Policy Map ===")
    print(render_policy_map(eval_env, policy))

    q_table_finite = bool(np.isfinite(result.q_table).all())
    history_length_ok = bool(len(result.episode_returns) == episodes)
    print("\n=== Checks ===")
    print(f"q_table_finite={q_table_finite}")
    print(f"history_length_ok={history_length_ok}")

    if not q_table_finite:
        raise RuntimeError("Q table contains non-finite values.")


if __name__ == "__main__":
    main()
