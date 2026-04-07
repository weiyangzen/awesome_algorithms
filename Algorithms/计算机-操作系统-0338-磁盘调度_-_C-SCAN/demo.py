"""C-SCAN disk scheduling MVP with deterministic demo and lightweight statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HeadMove:
    from_pos: int
    to_pos: int
    distance: int
    phase: str
    target_type: str  # "request" | "boundary" | "jump"


@dataclass
class CScanResult:
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
    current: int,
    target: int,
    phase: str,
    target_type: str,
) -> int:
    distance = abs(target - current)
    moves.append(
        HeadMove(
            from_pos=current,
            to_pos=target,
            distance=distance,
            phase=phase,
            target_type=target_type,
        )
    )
    return target


def cscan_schedule(
    requests: list[int],
    start_head: int,
    max_cylinder: int,
    direction: str = "right",
) -> CScanResult:
    """C-SCAN: service requests in one direction and wrap around at disk boundary."""
    _validate_inputs(requests, start_head, max_cylinder, direction)

    equal = [r for r in requests if r == start_head]
    less_asc = sorted([r for r in requests if r < start_head])
    greater_asc = sorted([r for r in requests if r > start_head])
    less_desc = sorted([r for r in requests if r < start_head], reverse=True)
    greater_desc = sorted([r for r in requests if r > start_head], reverse=True)

    if direction == "right":
        first_sweep = equal + greater_asc
        second_sweep = less_asc
        boundary = max_cylinder
        wrapped_boundary = 0
        phase = "right_sweep"
    else:
        first_sweep = equal + less_desc
        second_sweep = greater_desc
        boundary = 0
        wrapped_boundary = max_cylinder
        phase = "left_sweep"

    service_order: list[int] = []
    moves: list[HeadMove] = []
    current = start_head

    for req in first_sweep:
        current = _record_move(
            moves,
            current=current,
            target=req,
            phase=phase,
            target_type="request",
        )
        service_order.append(req)

    # If opposite-side requests exist, C-SCAN reaches boundary then wraps to opposite side.
    if second_sweep:
        if current != boundary:
            current = _record_move(
                moves,
                current=current,
                target=boundary,
                phase=phase,
                target_type="boundary",
            )
        current = _record_move(
            moves,
            current=current,
            target=wrapped_boundary,
            phase=phase,
            target_type="jump",
        )

    for req in second_sweep:
        current = _record_move(
            moves,
            current=current,
            target=req,
            phase=phase,
            target_type="request",
        )
        service_order.append(req)

    return CScanResult(
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
    for req in requests:
        current = _record_move(
            moves,
            current=current,
            target=req,
            phase="fcfs",
            target_type="request",
        )

    return FCFSResult(requests=list(requests), start_head=start_head, moves=moves)


def moves_to_dataframe(moves: list[HeadMove]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "from": m.from_pos,
                "to": m.to_pos,
                "distance": m.distance,
                "phase": m.phase,
                "target_type": m.target_type,
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
    rows: list[dict[str, int | float | str]] = []

    for i in range(trials):
        requests = rng.integers(0, max_cylinder + 1, size=req_count).tolist()
        start_head = int(rng.integers(0, max_cylinder + 1))
        direction = "right" if i % 2 == 0 else "left"

        cscan_res = cscan_schedule(requests, start_head, max_cylinder, direction)
        fcfs_res = fcfs_schedule(requests, start_head, max_cylinder)

        rows.append(
            {
                "trial": i,
                "direction": direction,
                "cscan_seek": cscan_res.total_seek,
                "fcfs_seek": fcfs_res.total_seek,
                "improvement": fcfs_res.total_seek - cscan_res.total_seek,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    requests = [176, 79, 34, 60, 92, 11, 41, 114]
    start_head = 50
    max_cylinder = 199
    direction = "right"

    cscan_res = cscan_schedule(requests, start_head, max_cylinder, direction)
    fcfs_res = fcfs_schedule(requests, start_head, max_cylinder)

    # Deterministic checks for the built-in sample.
    assert cscan_res.total_seek == 389, f"Unexpected C-SCAN seek: {cscan_res.total_seek}"
    assert fcfs_res.total_seek == 510, f"Unexpected FCFS seek: {fcfs_res.total_seek}"
    assert sorted(cscan_res.service_order) == sorted(requests)
    assert len(cscan_res.service_order) == len(requests)

    cscan_moves_df = moves_to_dataframe(cscan_res.moves)
    fcfs_moves_df = moves_to_dataframe(fcfs_res.moves)

    print("=== C-SCAN Disk Scheduling Demo ===")
    print(f"requests = {requests}")
    print(
        f"start_head = {start_head}, max_cylinder = {max_cylinder}, direction = {direction}"
    )
    print()

    print("C-SCAN service order:")
    print(cscan_res.service_order)
    print()

    print("C-SCAN movement trace:")
    print(cscan_moves_df.to_string(index=False))
    print()

    print("FCFS movement trace:")
    print(fcfs_moves_df.to_string(index=False))
    print()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "C-SCAN",
                "total_seek": cscan_res.total_seek,
                "avg_seek_per_request": cscan_res.average_seek_per_request,
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
                "metric": "mean_cscan_seek",
                "value": float(random_df["cscan_seek"].mean()),
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
                "metric": "cscan_better_ratio",
                "value": float((random_df["cscan_seek"] < random_df["fcfs_seek"]).mean()),
            },
        ]
    )

    print("Random comparison over 200 trials:")
    print(random_stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    assert len(random_df) == 200
    assert (random_df["cscan_seek"] >= 0).all()
    assert (random_df["fcfs_seek"] >= 0).all()

    print("All assertions passed.")


if __name__ == "__main__":
    main()
