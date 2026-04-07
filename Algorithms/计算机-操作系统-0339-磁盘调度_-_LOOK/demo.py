"""LOOK disk scheduling MVP with deterministic demo and lightweight statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HeadMove:
    step: int
    from_pos: int
    to_pos: int
    distance: int
    phase: str
    remaining_after: int


@dataclass
class LookResult:
    requests: list[int]
    start_head: int
    max_cylinder: int
    direction: str
    service_order: list[int]
    moves: list[HeadMove]

    @property
    def total_seek(self) -> int:
        return int(sum(m.distance for m in self.moves))

    @property
    def average_seek_per_request(self) -> float:
        n = len(self.requests)
        return float(self.total_seek / n) if n else 0.0


@dataclass
class FCFSResult:
    requests: list[int]
    start_head: int
    moves: list[HeadMove]

    @property
    def total_seek(self) -> int:
        return int(sum(m.distance for m in self.moves))


def _validate_inputs(
    requests: list[int],
    start_head: int,
    max_cylinder: int,
    direction: str,
) -> None:
    if max_cylinder < 0:
        raise ValueError("max_cylinder must be >= 0")
    if not 0 <= start_head <= max_cylinder:
        raise ValueError("start_head must be within [0, max_cylinder]")
    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")
    for idx, req in enumerate(requests):
        if not 0 <= req <= max_cylinder:
            raise ValueError(
                f"request[{idx}]={req} must be within [0, {max_cylinder}]"
            )


def _record_move(
    moves: list[HeadMove],
    step: int,
    current: int,
    target: int,
    phase: str,
    remaining_after: int,
) -> int:
    distance = abs(target - current)
    moves.append(
        HeadMove(
            step=step,
            from_pos=current,
            to_pos=target,
            distance=distance,
            phase=phase,
            remaining_after=remaining_after,
        )
    )
    return target


def look_schedule(
    requests: list[int],
    start_head: int,
    max_cylinder: int,
    direction: str = "right",
) -> LookResult:
    """LOOK scheduling: sweep in one direction, reverse at the last pending request."""
    _validate_inputs(requests, start_head, max_cylinder, direction)

    equal = [r for r in requests if r == start_head]
    less_desc = sorted([r for r in requests if r < start_head], reverse=True)
    greater_asc = sorted([r for r in requests if r > start_head])

    if direction == "right":
        first_sweep = equal + greater_asc
        second_sweep = less_desc
        first_phase = "right_sweep"
        second_phase = "left_sweep"
    else:
        first_sweep = equal + less_desc
        second_sweep = greater_asc
        first_phase = "left_sweep"
        second_phase = "right_sweep"

    service_order: list[int] = []
    moves: list[HeadMove] = []
    current = start_head
    step = 0

    for req in first_sweep:
        step += 1
        current = _record_move(
            moves=moves,
            step=step,
            current=current,
            target=req,
            phase=first_phase,
            remaining_after=len(requests) - step,
        )
        service_order.append(req)

    for req in second_sweep:
        step += 1
        current = _record_move(
            moves=moves,
            step=step,
            current=current,
            target=req,
            phase=second_phase,
            remaining_after=len(requests) - step,
        )
        service_order.append(req)

    return LookResult(
        requests=list(requests),
        start_head=start_head,
        max_cylinder=max_cylinder,
        direction=direction,
        service_order=service_order,
        moves=moves,
    )


def fcfs_schedule(requests: list[int], start_head: int, max_cylinder: int) -> FCFSResult:
    _validate_inputs(requests, start_head, max_cylinder, direction="right")

    current = start_head
    moves: list[HeadMove] = []

    for idx, req in enumerate(requests, start=1):
        current = _record_move(
            moves=moves,
            step=idx,
            current=current,
            target=req,
            phase="fcfs",
            remaining_after=len(requests) - idx,
        )

    return FCFSResult(requests=list(requests), start_head=start_head, moves=moves)


def moves_to_dataframe(moves: list[HeadMove]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": m.step,
                "from": m.from_pos,
                "to": m.to_pos,
                "distance": m.distance,
                "phase": m.phase,
                "remaining_after": m.remaining_after,
            }
            for m in moves
        ]
    )


def run_random_comparison(
    trials: int = 200,
    req_count: int = 20,
    max_cylinder: int = 199,
    seed: int = 20260407,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int | str]] = []

    for i in range(trials):
        requests = rng.integers(0, max_cylinder + 1, size=req_count).tolist()
        start_head = int(rng.integers(0, max_cylinder + 1))
        direction = "right" if i % 2 == 0 else "left"

        look_res = look_schedule(requests, start_head, max_cylinder, direction)
        fcfs_res = fcfs_schedule(requests, start_head, max_cylinder)

        rows.append(
            {
                "trial": i,
                "direction": direction,
                "look_seek": look_res.total_seek,
                "fcfs_seek": fcfs_res.total_seek,
                "improvement": fcfs_res.total_seek - look_res.total_seek,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    requests = [176, 79, 34, 60, 92, 11, 41, 114]
    start_head = 50
    max_cylinder = 199
    direction = "right"

    look_res = look_schedule(requests, start_head, max_cylinder, direction)
    fcfs_res = fcfs_schedule(requests, start_head, max_cylinder)

    # Deterministic checks for the built-in sample.
    assert look_res.total_seek == 291, f"Unexpected LOOK seek: {look_res.total_seek}"
    assert fcfs_res.total_seek == 510, f"Unexpected FCFS seek: {fcfs_res.total_seek}"
    assert sorted(look_res.service_order) == sorted(requests)
    assert len(look_res.service_order) == len(requests)

    look_moves_df = moves_to_dataframe(look_res.moves)
    fcfs_moves_df = moves_to_dataframe(fcfs_res.moves)

    print("=== LOOK Disk Scheduling Demo ===")
    print(f"requests = {requests}")
    print(
        f"start_head = {start_head}, max_cylinder = {max_cylinder}, direction = {direction}"
    )
    print()

    print("LOOK service order:")
    print(look_res.service_order)
    print()

    print("LOOK movement trace:")
    print(look_moves_df.to_string(index=False))
    print()

    print("FCFS movement trace:")
    print(fcfs_moves_df.to_string(index=False))
    print()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "LOOK",
                "total_seek": look_res.total_seek,
                "avg_seek_per_request": look_res.average_seek_per_request,
            },
            {
                "algorithm": "FCFS",
                "total_seek": fcfs_res.total_seek,
                "avg_seek_per_request": fcfs_res.total_seek / len(requests),
            },
        ]
    )
    print("Summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    random_df = run_random_comparison()
    random_stats = pd.DataFrame(
        [
            {
                "metric": "mean_look_seek",
                "value": float(random_df["look_seek"].mean()),
            },
            {
                "metric": "mean_fcfs_seek",
                "value": float(random_df["fcfs_seek"].mean()),
            },
            {
                "metric": "mean_improvement",
                "value": float(random_df["improvement"].mean()),
            },
            {
                "metric": "look_better_ratio",
                "value": float((random_df["look_seek"] < random_df["fcfs_seek"]).mean()),
            },
        ]
    )

    print("Random comparison over 200 trials:")
    print(random_stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
