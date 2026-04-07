"""Minimal runnable MVP for LRU page replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class _Node:
    """Node in a doubly linked list representing one page frame."""

    page: int
    prev: Optional["_Node"] = None
    next: Optional["_Node"] = None


class _DoublyLinkedList:
    """Maintains pages in recency order: head=LRU, tail=MRU."""

    def __init__(self) -> None:
        self.head: Optional[_Node] = None
        self.tail: Optional[_Node] = None

    def append_mru(self, node: _Node) -> None:
        node.prev = self.tail
        node.next = None
        if self.tail is not None:
            self.tail.next = node
        else:
            self.head = node
        self.tail = node

    def remove(self, node: _Node) -> None:
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next is not None:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        node.prev = None
        node.next = None

    def pop_lru(self) -> Optional[_Node]:
        if self.head is None:
            return None
        lru_node = self.head
        self.remove(lru_node)
        return lru_node

    def as_list(self) -> List[int]:
        pages: List[int] = []
        cur = self.head
        while cur is not None:
            pages.append(cur.page)
            cur = cur.next
        return pages


class LRUPageReplacement:
    """LRU page replacement simulator with O(1) access updates."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._list = _DoublyLinkedList()
        self._nodes: Dict[int, _Node] = {}
        self.hits = 0
        self.faults = 0

    def access(self, page: int) -> dict:
        evicted: Optional[int] = None
        hit = page in self._nodes

        if hit:
            node = self._nodes[page]
            self._list.remove(node)
            self._list.append_mru(node)
            self.hits += 1
        else:
            if len(self._nodes) >= self.capacity:
                lru_node = self._list.pop_lru()
                if lru_node is None:
                    raise RuntimeError("internal state error: expected non-empty list")
                evicted = lru_node.page
                del self._nodes[evicted]

            new_node = _Node(page=page)
            self._list.append_mru(new_node)
            self._nodes[page] = new_node
            self.faults += 1

        return {
            "page": page,
            "hit": hit,
            "evicted": evicted,
            "frames_lru_to_mru": self._list.as_list(),
        }

    def simulate(self, reference_string: Iterable[int]) -> dict:
        steps: List[dict] = []
        for page in reference_string:
            steps.append(self.access(page))

        total = self.hits + self.faults
        fault_rate = (self.faults / total) if total else 0.0
        return {
            "capacity": self.capacity,
            "total": total,
            "hits": self.hits,
            "faults": self.faults,
            "fault_rate": fault_rate,
            "steps": steps,
        }


def main() -> None:
    reference_string = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]
    capacity = 3

    simulator = LRUPageReplacement(capacity=capacity)
    result = simulator.simulate(reference_string)

    print("LRU Page Replacement Demo")
    print(f"Reference string: {reference_string}")
    print(f"Frame capacity : {capacity}")
    print()
    print("Step | Ref | Result | Evicted | Frames (LRU -> MRU)")
    print("-----+-----+--------+---------+---------------------")

    for i, step in enumerate(result["steps"], start=1):
        result_text = "HIT" if step["hit"] else "FAULT"
        evicted_text = "-" if step["evicted"] is None else str(step["evicted"])
        frames_text = str(step["frames_lru_to_mru"])
        print(f"{i:>4} | {step['page']:>3} | {result_text:>6} | {evicted_text:>7} | {frames_text}")

    print()
    print(
        "Summary: "
        f"total={result['total']}, hits={result['hits']}, faults={result['faults']}, "
        f"fault_rate={result['fault_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
