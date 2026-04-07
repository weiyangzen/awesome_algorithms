"""FCFS disk scheduling MVP with deterministic demo and lightweight statistics."""

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
    remaining_after: int


@dataclass
class FCFSResult:
    requests: list[int]
    start_head: int
    max_cylinder: int
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
class SSTFResult:
    requests: list[int]
    start_head: int
    max_cylinder: int
    service_order: list[int]
    moves: list[HeadMove]

    @property
    def total_seek(self) -> int:
        return int(sum(m.distance for m in self.moves))


def _validate_inputs(requests: list[int], start_head: int, max_cylinder: int) -> None:
    if max_cylinder < 0:
        raise ValueError("max_cylinder must be >= 0")
    if not 0 <= start_head <= max_cylinder:
        raise ValueError("start_head must be within [0, max_cylinder]")
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
    remaining_after: int,
) -> int:
    distance = abs(target - current)
    moves.append(
        HeadMove(
            step=step,
            from_pos=current,
            to_pos=target,
            distance=distance,
            remaining_after=remaining_after,
        )
    )
    return target


def fcfs_schedule(
    requests: list[int],
    start_head: int,
    max_cylinder: int,
) -> FCFSResult:
    """First-Come, First-Served disk scheduling."""
    _validate_inputs(requests, start_head, max_cylinder)

    current = start_head
    moves: list[HeadMove] = []
    service_order: list[int] = []

    for idx, req in enumerate(requests, start=1):
        current = _record_move(
            moves=moves,
            step=idx,
            current=current,
            target=req,
            remaining_after=len(requests) - idx,
        )
        service_order.append(req)

    return FCFSResult(
        requests=list(requests),
        start_head=start_head,
        max_cylinder=max_cylinder,
        service_order=service_order,
        moves=moves,
    )


def sstf_schedule(
    requests: list[int],
    start_head: int,
    max_cylinder: int,
) -> SSTFResult:
    """Shortest Seek Time First used only as an interpretable baseline comparator."""
    _validate_inputs(requests, start_head, max_cylinder)

    pending = list(requests)
    service_order: list[int] = []
    moves: list[HeadMove] = []
    current = start_head
    step = 0

    while pending:
        best_idx = 0
        best_req = pending[0]
        best_dist = abs(best_req - current)

        for idx in range(1, len(pending)):
            req = pending[idx]
            dist = abs(req - current)
            if dist < best_dist or (dist == best_dist and req < best_req):
                best_idx = idx
                best_req = req
                best_dist = dist

        pending.pop(best_idx)
        step += 1
        current = _record_move(
            moves=moves,
            step=step,
            current=current,
            target=best_req,
            remaining_after=len(pending),
        )
        service_order.append(best_req)

    return SSTFResult(
        requests=list(requests),
        start_head=start_head,
        max_cylinder=max_cylinder,
        service_order=service_order,
        moves=moves,
    )


def moves_to_dataframe(moves: list[HeadMove]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": m.step,
                "from": m.from_pos,
                "to": m.to_pos,
                "distance": m.distance,
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
    rows: list[dict[str, int]] = []

    for i in range(trials):
        requests = rng.integers(0, max_cylinder + 1, size=req_count).tolist()
        start_head = int(rng.integers(0, max_cylinder + 1))

        fcfs_res = fcfs_schedule(requests, start_head, max_cylinder)
        sstf_res = sstf_schedule(requests, start_head, max_cylinder)

        rows.append(
            {
                "trial": i,
                "fcfs_seek": fcfs_res.total_seek,
                "sstf_seek": sstf_res.total_seek,
                "fcfs_minus_sstf": fcfs_res.total_seek - sstf_res.total_seek,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    requests = [98, 183, 37, 122, 14, 124, 65, 67]
    start_head = 53
    max_cylinder = 199

    fcfs_res = fcfs_schedule(requests, start_head, max_cylinder)
    sstf_res = sstf_schedule(requests, start_head, max_cylinder)

    # Deterministic checks for the built-in sample.
    assert fcfs_res.total_seek == 640, f"Unexpected FCFS seek: {fcfs_res.total_seek}"
    assert sstf_res.total_seek == 236, f"Unexpected SSTF seek: {sstf_res.total_seek}"
    assert fcfs_res.service_order == requests
    assert len(fcfs_res.service_order) == len(requests)

    fcfs_moves_df = moves_to_dataframe(fcfs_res.moves)
    sstf_moves_df = moves_to_dataframe(sstf_res.moves)

    print("=== FCFS Disk Scheduling Demo ===")
    print(f"requests = {requests}")
    print(f"start_head = {start_head}, max_cylinder = {max_cylinder}")
    print()

    print("FCFS service order:")
    print(fcfs_res.service_order)
    print()

    print("FCFS movement trace:")
    print(fcfs_moves_df.to_string(index=False))
    print()

    print("SSTF movement trace (baseline comparison):")
    print(sstf_moves_df.to_string(index=False))
    print()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "FCFS",
                "total_seek": fcfs_res.total_seek,
                "avg_seek_per_request": fcfs_res.average_seek_per_request,
            },
            {
                "algorithm": "SSTF",
                "total_seek": sstf_res.total_seek,
                "avg_seek_per_request": sstf_res.total_seek / len(requests),
            },
        ]
    )

    print("Summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    random_df = run_random_comparison()
    mean_fcfs = float(random_df["fcfs_seek"].mean())
    mean_sstf = float(random_df["sstf_seek"].mean())
    worse_ratio = float((random_df["fcfs_seek"] > random_df["sstf_seek"]).mean())

    print("Random workload (200 trials) average total seek:")
    print(f"FCFS={mean_fcfs:.3f}, SSTF={mean_sstf:.3f}")
    print(f"FCFS worse ratio vs SSTF={worse_ratio:.3f}")

    assert len(random_df) == 200
    assert 0.0 <= worse_ratio <= 1.0
    assert mean_fcfs >= 0 and mean_sstf >= 0

    print("All assertions passed.")


if __name__ == "__main__":
    main()
