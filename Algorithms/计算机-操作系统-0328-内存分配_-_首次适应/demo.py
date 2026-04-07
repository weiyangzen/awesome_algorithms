"""First-Fit memory allocation MVP.

This script simulates contiguous dynamic partition allocation in an OS:
- `allocate(pid, size)`: pick the first free block (by address order) that fits
- `free(pid)`: release a process block and coalesce adjacent free blocks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class MemoryBlock:
    """A contiguous memory segment."""

    start: int
    size: int
    free: bool
    pid: str | None = None

    @property
    def end(self) -> int:
        return self.start + self.size

    def label(self) -> str:
        return "FREE" if self.free else f"PID={self.pid}"


class FirstFitAllocator:
    """Contiguous allocator using the first-fit policy."""

    def __init__(self, total_memory: int) -> None:
        if total_memory <= 0:
            raise ValueError("total_memory must be positive")
        self.total_memory = total_memory
        self.blocks: list[MemoryBlock] = [MemoryBlock(0, total_memory, True, None)]

    def allocate(self, pid: str, size: int) -> int | None:
        """Allocate memory for a process and return start address."""
        if not pid:
            raise ValueError("pid must be non-empty")
        if size <= 0:
            raise ValueError("size must be positive")
        if any((not block.free) and block.pid == pid for block in self.blocks):
            raise ValueError(f"pid '{pid}' already exists")

        first_index: int | None = None
        for idx, block in enumerate(self.blocks):
            if block.free and block.size >= size:
                first_index = idx
                break

        if first_index is None:
            return None

        chosen = self.blocks[first_index]
        alloc_start = chosen.start
        remaining = chosen.size - size

        if remaining == 0:
            chosen.free = False
            chosen.pid = pid
        else:
            allocated = MemoryBlock(alloc_start, size, False, pid)
            residue = MemoryBlock(alloc_start + size, remaining, True, None)
            self.blocks[first_index : first_index + 1] = [allocated, residue]
        return alloc_start

    def free(self, pid: str) -> bool:
        """Release memory by process id."""
        for idx, block in enumerate(self.blocks):
            if (not block.free) and block.pid == pid:
                block.free = True
                block.pid = None
                self._coalesce_around(idx)
                return True
        return False

    def _coalesce_around(self, idx: int) -> None:
        """Merge neighboring free blocks around an index."""
        i = max(0, idx - 1)
        while i < len(self.blocks) - 1:
            curr = self.blocks[i]
            nxt = self.blocks[i + 1]
            if curr.free and nxt.free and curr.end == nxt.start:
                curr.size += nxt.size
                del self.blocks[i + 1]
            else:
                i += 1

    def total_free(self) -> int:
        return sum(block.size for block in self.blocks if block.free)

    def largest_free_block(self) -> int:
        return max((block.size for block in self.blocks if block.free), default=0)

    def external_fragmentation(self) -> int:
        """A simple indicator: free_total - largest_free."""
        return self.total_free() - self.largest_free_block()

    def iter_rows(self) -> Iterable[tuple[int, int, int, str]]:
        for block in self.blocks:
            yield (block.start, block.end, block.size, block.label())


def print_state(allocator: FirstFitAllocator, title: str) -> None:
    print(f"\n=== {title} ===")
    print("start  end    size   state")
    print("-" * 32)
    for start, end, size, label in allocator.iter_rows():
        print(f"{start:>5}  {end:>5}  {size:>5}  {label}")
    print("-" * 32)
    print(
        "free_total={free_total}, largest_free={largest}, external_frag={frag}".format(
            free_total=allocator.total_free(),
            largest=allocator.largest_free_block(),
            frag=allocator.external_fragmentation(),
        )
    )


def main() -> None:
    allocator = FirstFitAllocator(total_memory=1024)

    print_state(allocator, "Initial")

    plan: list[tuple[str, str, int | None]] = [
        ("alloc", "A", 120),
        ("alloc", "B", 300),
        ("alloc", "C", 180),
        ("alloc", "D", 90),
        ("free", "B", None),
        ("free", "D", None),
        ("alloc", "E", 80),   # first-fit picks the first free block (former B, size 300)
        ("alloc", "F", 260),  # first-fit skips 220 and uses the next suitable free block
        ("alloc", "G", 50),   # first-fit returns to the low-address free block
    ]

    for op, pid, size in plan:
        if op == "alloc":
            assert size is not None
            addr = allocator.allocate(pid, size)
            if addr is None:
                print(f"\nallocate(pid={pid}, size={size}) -> FAILED")
            else:
                print(f"\nallocate(pid={pid}, size={size}) -> start={addr}")
        else:
            ok = allocator.free(pid)
            print(f"\nfree(pid={pid}) -> {'OK' if ok else 'NOT_FOUND'}")

        print_state(allocator, f"After {op} {pid}")


if __name__ == "__main__":
    main()
