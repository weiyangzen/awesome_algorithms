"""Runnable MVP for CS-0017: Linear Search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar, cast

import numpy as np

T = TypeVar("T")
K = TypeVar("K")


@dataclass(frozen=True)
class SearchResult:
    """Container for linear search result."""

    index: int
    comparisons: int

    @property
    def found(self) -> bool:
        return self.index >= 0


def _identity(value: T) -> T:
    return value


def linear_search(
    data: Sequence[T],
    target: K,
    key: Callable[[T], K] | None = None,
) -> SearchResult:
    """Return first match index and comparison count using linear scan."""

    if key is None:
        projector = cast(Callable[[T], K], _identity)
    else:
        projector = key
    comparisons = 0

    for idx, item in enumerate(data):
        comparisons += 1
        if projector(item) == target:
            return SearchResult(index=idx, comparisons=comparisons)

    return SearchResult(index=-1, comparisons=comparisons)


def _run_self_checks() -> None:
    """Minimal non-interactive checks for core branches."""

    assert linear_search([], 10) == SearchResult(index=-1, comparisons=0)
    assert linear_search([4, 7, 7], 7) == SearchResult(index=1, comparisons=2)
    assert linear_search([1, 2, 3], 99) == SearchResult(index=-1, comparisons=3)

    rows = [
        {"id": 101, "name": "Ada"},
        {"id": 102, "name": "Grace"},
        {"id": 103, "name": "Linus"},
    ]
    assert linear_search(rows, 102, key=lambda row: row["id"]) == SearchResult(
        index=1,
        comparisons=2,
    )


def main() -> None:
    _run_self_checks()

    list_data = [12, 5, 9, 42, 7]
    result_list = linear_search(list_data, 42)

    np_data = np.array([3, 1, 4, 1, 5, 9])
    result_np = linear_search(np_data, 9)

    records = [
        {"user": "alice", "score": 68},
        {"user": "bob", "score": 92},
        {"user": "carol", "score": 75},
    ]
    result_records = linear_search(records, 92, key=lambda row: row["score"])

    result_missing = linear_search(list_data, 100)

    print("Linear Search MVP")
    print(f"list_data={list_data}, target=42, result={result_list}")
    print(f"np_data={np_data.tolist()}, target=9, result={result_np}")
    print(
        "records(score), target=92, "
        f"result={result_records}, found_user={records[result_records.index]['user']}"
    )
    print(f"list_data={list_data}, target=100, result={result_missing}")
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
