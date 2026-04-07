"""SCAN (Elevator) disk scheduling MVP with a deterministic demo and lightweight stats."""

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
    target_type: str  # "request" or "boundary"


@dataclass
class ScanResult:
    algorithm: str
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


def scan_schedule(
    requests: list[int],
    start_head: int,
    max_cylinder: int,
    direction: str = "right",
) -> ScanResult:
    """Strict SCAN (elevator): if reversal is needed, head goes to boundary then turns."""
    _validate_inputs(requests, start_head, max_cylinder, direction)

    equal = [r for r in requests if r == start_head]
    less_desc = sorted([r for r in requests if r < start_head], reverse=True)
    greater_asc = sorted([r for r in requests if r > start_head])

    if direction == "right":
        first_sweep = equal + greater_asc
        second_sweep = less_desc
        boundary = max_cylinder
        first_phase = "right_sweep"
        second_phase = "left_sweep"
    else:
        first_sweep = equal + less_desc
        second_sweep = greater_asc
        boundary = 0
        first_phase = "left_sweep"
        second_phase = "right_sweep"

    service_order: list[int] = []
    moves: list[HeadMove] = []
    current = start_head

    for req in first_sweep:
        current = _record_move(
            moves,
            current=current,
            target=req,
            phase=first_phase,
            target_type="request",
        )
        service_order.append(req)

    # SCAN (not LOOK): if opposite-side requests exist, reach disk boundary before reversing.
    if second_sweep and current != boundary:
        current = _record_move(
            moves,
            current=current,
            target=boundary,
            phase=first_phase,
            target_type="boundary",
        )

    for req in second_sweep:
        current = _record_move(
            moves,
            current=current,
            target=req,
            phase=second_phase,
            target_type="request",
        )
        service_order.append(req)

    return ScanResult(
        algorithm="SCAN",
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
    rows: list[dict[str, float | int | str]] = []

    for i in range(trials):
        requests = rng.integers(0, max_cylinder + 1, size=req_count).tolist()
        start_head = int(rng.integers(0, max_cylinder + 1))
        direction = "right" if i % 2 == 0 else "left"

        scan_res = scan_schedule(requests, start_head, max_cylinder, direction)
        fcfs_res = fcfs_schedule(requests, start_head, max_cylinder)

        rows.append(
            {
                "trial": i,
                "scan_seek": scan_res.total_seek,
                "fcfs_seek": fcfs_res.total_seek,
                "improvement": fcfs_res.total_seek - scan_res.total_seek,
                "direction": direction,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    requests = [176, 79, 34, 60, 92, 11, 41, 114]
    start_head = 50
    max_cylinder = 199
    direction = "right"

    scan_res = scan_schedule(requests, start_head, max_cylinder, direction)
    fcfs_res = fcfs_schedule(requests, start_head, max_cylinder)

    # Deterministic checks for the built-in sample.
    assert scan_res.total_seek == 337, f"Unexpected SCAN seek: {scan_res.total_seek}"
    assert fcfs_res.total_seek == 510, f"Unexpected FCFS seek: {fcfs_res.total_seek}"
    assert sorted(scan_res.service_order) == sorted(requests)
    assert len(scan_res.service_order) == len(requests)

    scan_moves_df = moves_to_dataframe(scan_res.moves)
    fcfs_moves_df = moves_to_dataframe(fcfs_res.moves)

    print("=== SCAN Disk Scheduling Demo ===")
    print(f"requests = {requests}")
    print(
        f"start_head = {start_head}, max_cylinder = {max_cylinder}, direction = {direction}"
    )
    print()

    print("SCAN service order:")
    print(scan_res.service_order)
    print()

    print("SCAN movement trace:")
    print(scan_moves_df.to_string(index=False))
    print()

    print("FCFS movement trace:")
    print(fcfs_moves_df.to_string(index=False))
    print()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "SCAN",
                "total_seek": scan_res.total_seek,
                "avg_seek_per_request": scan_res.average_seek_per_request,
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
    mean_scan = float(random_df["scan_seek"].mean())
    mean_fcfs = float(random_df["fcfs_seek"].mean())
    better_ratio = float((random_df["scan_seek"] < random_df["fcfs_seek"]).mean())

    print("Random workload (200 trials) average total seek:")
    print(f"SCAN={mean_scan:.3f}, FCFS={mean_fcfs:.3f}")
    print(f"SCAN better ratio={better_ratio:.3f}")

    # Weak sanity bounds for randomized part.
    assert len(random_df) == 200
    assert 0.0 <= better_ratio <= 1.0
    assert mean_scan >= 0 and mean_fcfs >= 0

    print("All assertions passed.")


if __name__ == "__main__":
    main()
