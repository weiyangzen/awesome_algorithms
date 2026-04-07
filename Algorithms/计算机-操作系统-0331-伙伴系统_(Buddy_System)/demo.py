"""Buddy System allocator MVP (single-process simulation, deterministic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class AllocationRecord:
    request_size: int
    order: int
    block_size: int


@dataclass(frozen=True)
class Operation:
    kind: str  # "alloc" or "free"
    name: str
    size: int | None = None


class BuddyAllocator:
    def __init__(self, total_size: int, min_block_size: int) -> None:
        if total_size <= 0 or min_block_size <= 0:
            raise ValueError("sizes must be positive")
        if not self._is_power_of_two(total_size):
            raise ValueError("total_size must be a power of two")
        if not self._is_power_of_two(min_block_size):
            raise ValueError("min_block_size must be a power of two")
        if min_block_size > total_size:
            raise ValueError("min_block_size must be <= total_size")

        self.total_size = total_size
        self.min_block_size = min_block_size
        self.min_order = min_block_size.bit_length() - 1
        self.max_order = total_size.bit_length() - 1

        self.free_lists: Dict[int, set[int]] = {
            order: set() for order in range(self.min_order, self.max_order + 1)
        }
        self.free_lists[self.max_order].add(0)

        self.allocations: Dict[int, AllocationRecord] = {}

    @staticmethod
    def _is_power_of_two(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0

    @staticmethod
    def _next_power_of_two(x: int) -> int:
        return 1 << (x - 1).bit_length()

    @staticmethod
    def _block_size(order: int) -> int:
        return 1 << order

    def _required_order(self, request_size: int) -> int:
        if request_size <= 0:
            raise ValueError("request_size must be > 0")
        rounded = max(self.min_block_size, self._next_power_of_two(request_size))
        return rounded.bit_length() - 1

    def allocate(self, request_size: int) -> int | None:
        need_order = self._required_order(request_size)
        if need_order > self.max_order:
            return None

        source_order: int | None = None
        for order in range(need_order, self.max_order + 1):
            if self.free_lists[order]:
                source_order = order
                break
        if source_order is None:
            return None

        addr = min(self.free_lists[source_order])
        self.free_lists[source_order].remove(addr)

        current_order = source_order
        while current_order > need_order:
            current_order -= 1
            half_size = self._block_size(current_order)
            buddy_addr = addr + half_size
            self.free_lists[current_order].add(buddy_addr)

        block_size = self._block_size(need_order)
        self.allocations[addr] = AllocationRecord(
            request_size=request_size,
            order=need_order,
            block_size=block_size,
        )
        return addr

    def free(self, addr: int) -> None:
        record = self.allocations.pop(addr, None)
        if record is None:
            raise KeyError(f"address {addr} was not allocated")

        current_addr = addr
        current_order = record.order

        while current_order < self.max_order:
            block_size = self._block_size(current_order)
            buddy_addr = current_addr ^ block_size
            if buddy_addr in self.free_lists[current_order]:
                self.free_lists[current_order].remove(buddy_addr)
                current_addr = min(current_addr, buddy_addr)
                current_order += 1
            else:
                break

        self.free_lists[current_order].add(current_addr)

    def stats(self) -> dict[str, float]:
        total_free = sum(
            len(addresses) * self._block_size(order)
            for order, addresses in self.free_lists.items()
        )
        largest_free = 0
        for order, addresses in self.free_lists.items():
            if addresses:
                largest_free = max(largest_free, self._block_size(order))

        allocated_block_bytes = sum(r.block_size for r in self.allocations.values())
        requested_bytes = sum(r.request_size for r in self.allocations.values())
        internal_frag_bytes = allocated_block_bytes - requested_bytes

        external_frag_ratio = 0.0
        if total_free > 0:
            external_frag_ratio = 1.0 - (largest_free / total_free)

        return {
            "total_free": float(total_free),
            "largest_free": float(largest_free),
            "allocated_block_bytes": float(allocated_block_bytes),
            "requested_bytes": float(requested_bytes),
            "internal_frag_bytes": float(internal_frag_bytes),
            "external_frag_ratio": float(external_frag_ratio),
        }

    def format_free_lists(self) -> str:
        parts: List[str] = []
        for order in range(self.max_order, self.min_order - 1, -1):
            addrs = sorted(self.free_lists[order])
            if addrs:
                parts.append(f"{self._block_size(order)}:{addrs}")
        return " | ".join(parts) if parts else "(empty)"

    def assert_integrity(self) -> None:
        intervals: List[tuple[int, int, str]] = []

        for addr, rec in self.allocations.items():
            block_size = rec.block_size
            assert addr % block_size == 0, (
                f"allocated block alignment error: addr={addr}, block={block_size}"
            )
            intervals.append((addr, addr + block_size, "alloc"))

        for order, addrs in self.free_lists.items():
            block_size = self._block_size(order)
            for addr in addrs:
                assert addr % block_size == 0, (
                    f"free block alignment error: addr={addr}, block={block_size}"
                )
                intervals.append((addr, addr + block_size, "free"))

        assert intervals, "allocator has no blocks at all"
        intervals.sort(key=lambda x: (x[0], x[1]))

        cursor = 0
        total_covered = 0
        for start, end, _ in intervals:
            assert 0 <= start < end <= self.total_size, (
                f"block out of range: [{start}, {end})"
            )
            assert start == cursor, (
                f"gap or overlap detected at cursor={cursor}, block=[{start}, {end})"
            )
            cursor = end
            total_covered += end - start

        assert cursor == self.total_size, (
            f"coverage tail mismatch: covered until {cursor}, total={self.total_size}"
        )
        assert total_covered == self.total_size, (
            f"coverage size mismatch: covered={total_covered}, total={self.total_size}"
        )


def run_demo() -> None:
    allocator = BuddyAllocator(total_size=1024, min_block_size=32)

    ops = [
        Operation("alloc", "A", 70),
        Operation("alloc", "B", 200),
        Operation("alloc", "C", 60),
        Operation("alloc", "D", 130),
        Operation("free", "B"),
        Operation("alloc", "E", 100),
        Operation("free", "C"),
        Operation("free", "E"),
        Operation("free", "A"),
        Operation("alloc", "F", 300),
        Operation("free", "D"),
        Operation("free", "F"),
        Operation("alloc", "G", 900),
        Operation("alloc", "H", 200),  # expected to fail
        Operation("free", "G"),
    ]

    name_to_addr: Dict[str, int] = {}
    alloc_failures = 0
    successful_request_sizes: List[int] = []
    successful_block_sizes: List[int] = []

    print("=== Buddy System Demo ===")
    print(f"total_size={allocator.total_size}, min_block_size={allocator.min_block_size}")
    print()

    for step, op in enumerate(ops, start=1):
        if op.kind == "alloc":
            assert op.size is not None
            addr = allocator.allocate(op.size)
            if addr is None:
                alloc_failures += 1
                status = f"ALLOC {op.name}({op.size}) -> FAIL"
            else:
                name_to_addr[op.name] = addr
                rec = allocator.allocations[addr]
                successful_request_sizes.append(rec.request_size)
                successful_block_sizes.append(rec.block_size)
                status = (
                    f"ALLOC {op.name}({op.size}) -> addr={addr}, block={rec.block_size}"
                )
        elif op.kind == "free":
            if op.name not in name_to_addr:
                raise KeyError(f"name {op.name} not allocated before free")
            addr = name_to_addr.pop(op.name)
            allocator.free(addr)
            status = f"FREE  {op.name} -> addr={addr}"
        else:
            raise ValueError(f"unknown operation kind: {op.kind}")

        allocator.assert_integrity()
        s = allocator.stats()
        print(
            f"[{step:02d}] {status:<36} | "
            f"free={int(s['total_free']):>4} "
            f"largest={int(s['largest_free']):>4} "
            f"int_frag={int(s['internal_frag_bytes']):>4} "
            f"ext_frag={s['external_frag_ratio']:.3f}"
        )
        print(f"     free_lists: {allocator.format_free_lists()}")

    final_stats = allocator.stats()

    req_arr = np.array(successful_request_sizes, dtype=np.int64)
    blk_arr = np.array(successful_block_sizes, dtype=np.int64)
    waste_arr = blk_arr - req_arr

    mean_waste = float(waste_arr.mean()) if waste_arr.size else 0.0
    p95_waste = float(np.percentile(waste_arr, 95)) if waste_arr.size else 0.0
    max_waste = int(waste_arr.max()) if waste_arr.size else 0

    print()
    print("=== Summary ===")
    print(f"successful_allocs={req_arr.size}, alloc_failures={alloc_failures}")
    print(
        "internal_waste(bytes): "
        f"mean={mean_waste:.2f}, p95={p95_waste:.2f}, max={max_waste}"
    )
    print(
        "final_state: "
        f"free={int(final_stats['total_free'])}, "
        f"largest={int(final_stats['largest_free'])}, "
        f"ext_frag={final_stats['external_frag_ratio']:.3f}"
    )

    # Final assertions for MVP correctness.
    assert alloc_failures >= 1, "expected at least one allocation failure"
    assert not allocator.allocations, "all allocations should have been released"
    assert final_stats["total_free"] == allocator.total_size
    assert final_stats["largest_free"] == allocator.total_size
    assert final_stats["external_frag_ratio"] == 0.0
    assert allocator.free_lists[allocator.max_order] == {0}
    for order in range(allocator.min_order, allocator.max_order):
        assert not allocator.free_lists[order], f"lower order list not empty: {order}"

    print("All assertions passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
