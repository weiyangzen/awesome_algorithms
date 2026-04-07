"""Virtual memory management MVP.

This script simulates a small demand-paging subsystem with:
- one-level page table
- LRU-style TLB (OrderedDict)
- CLOCK page replacement
- dirty-page write-back to backing store
- invariant checks for internal consistency
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PageTableEntry:
    present: bool = False
    frame_id: int = -1
    dirty: bool = False
    referenced: bool = False
    last_touch: int = -1


@dataclass
class MemoryStats:
    accesses: int = 0
    reads: int = 0
    writes: int = 0
    tlb_hits: int = 0
    tlb_misses: int = 0
    page_faults: int = 0
    evictions: int = 0
    write_backs: int = 0


@dataclass
class MemoryOp:
    vaddr: int
    is_write: bool
    value: int


@dataclass
class VirtualMemoryManager:
    num_virtual_pages: int
    num_frames: int
    page_size: int
    tlb_capacity: int

    page_table: List[PageTableEntry] = field(init=False)
    frame_to_vpn: List[Optional[int]] = field(init=False)
    free_frames: List[int] = field(init=False)
    tlb: OrderedDict[int, int] = field(init=False)
    memory: np.ndarray = field(init=False)
    backing_store: Dict[int, np.ndarray] = field(init=False)
    stats: MemoryStats = field(default_factory=MemoryStats)
    tick: int = 0
    clock_hand: int = 0

    def __post_init__(self) -> None:
        if self.num_virtual_pages <= 0:
            raise ValueError("num_virtual_pages must be positive")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.tlb_capacity <= 0:
            raise ValueError("tlb_capacity must be positive")

        self.tlb_capacity = min(self.tlb_capacity, self.num_virtual_pages)
        self.page_table = [PageTableEntry() for _ in range(self.num_virtual_pages)]
        self.frame_to_vpn = [None] * self.num_frames
        self.free_frames = list(range(self.num_frames - 1, -1, -1))
        self.tlb = OrderedDict()
        self.memory = np.zeros((self.num_frames, self.page_size), dtype=np.uint8)

        self.backing_store = {
            vpn: self._make_initial_page(vpn) for vpn in range(self.num_virtual_pages)
        }

    def _make_initial_page(self, vpn: int) -> np.ndarray:
        base = (vpn * 37 + 11) % 251
        row = (np.arange(self.page_size, dtype=np.uint16) + base) % 256
        return row.astype(np.uint8)

    def _decode_vaddr(self, vaddr: int) -> Tuple[int, int]:
        upper = self.num_virtual_pages * self.page_size
        if vaddr < 0 or vaddr >= upper:
            raise ValueError(f"virtual address {vaddr} out of range [0, {upper})")
        return divmod(vaddr, self.page_size)

    def _tlb_lookup(self, vpn: int) -> Optional[int]:
        frame_id = self.tlb.get(vpn)
        if frame_id is None:
            self.stats.tlb_misses += 1
            return None
        self.stats.tlb_hits += 1
        self.tlb.move_to_end(vpn)
        return frame_id

    def _tlb_insert(self, vpn: int, frame_id: int) -> None:
        self.tlb[vpn] = frame_id
        self.tlb.move_to_end(vpn)
        while len(self.tlb) > self.tlb_capacity:
            self.tlb.popitem(last=False)

    def _invalidate_tlb_vpn(self, vpn: int) -> None:
        self.tlb.pop(vpn, None)

    def _pick_victim_frame(self) -> int:
        for _ in range(self.num_frames * 2):
            frame_id = self.clock_hand
            self.clock_hand = (self.clock_hand + 1) % self.num_frames

            victim_vpn = self.frame_to_vpn[frame_id]
            if victim_vpn is None:
                return frame_id

            victim_pte = self.page_table[victim_vpn]
            if victim_pte.referenced:
                victim_pte.referenced = False
                continue
            return frame_id
        raise RuntimeError("CLOCK failed to find victim frame")

    def _handle_page_fault(self, vpn: int) -> int:
        self.stats.page_faults += 1

        if self.free_frames:
            frame_id = self.free_frames.pop()
        else:
            frame_id = self._pick_victim_frame()
            victim_vpn = self.frame_to_vpn[frame_id]
            if victim_vpn is None:
                raise RuntimeError("selected victim frame has no mapped vpn")

            victim_pte = self.page_table[victim_vpn]
            self.stats.evictions += 1

            if victim_pte.dirty:
                self.backing_store[victim_vpn] = self.memory[frame_id].copy()
                self.stats.write_backs += 1

            victim_pte.present = False
            victim_pte.frame_id = -1
            victim_pte.dirty = False
            victim_pte.referenced = False
            self._invalidate_tlb_vpn(victim_vpn)

        self.memory[frame_id] = self.backing_store[vpn]
        self.frame_to_vpn[frame_id] = vpn

        pte = self.page_table[vpn]
        pte.present = True
        pte.frame_id = frame_id
        pte.dirty = False
        pte.referenced = True
        pte.last_touch = self.tick

        self._tlb_insert(vpn, frame_id)
        return frame_id

    def access(self, vaddr: int, is_write: bool, value: int = 0) -> int:
        self.tick += 1
        self.stats.accesses += 1
        if is_write:
            self.stats.writes += 1
        else:
            self.stats.reads += 1

        vpn, offset = self._decode_vaddr(vaddr)

        frame_id = self._tlb_lookup(vpn)
        if frame_id is None:
            pte = self.page_table[vpn]
            if pte.present:
                frame_id = pte.frame_id
                if frame_id < 0:
                    raise RuntimeError("present page with invalid frame id")
                self._tlb_insert(vpn, frame_id)
            else:
                frame_id = self._handle_page_fault(vpn)

        pte = self.page_table[vpn]
        pte.referenced = True
        pte.last_touch = self.tick

        if is_write:
            self.memory[frame_id, offset] = int(value) & 0xFF
            pte.dirty = True

        return int(self.memory[frame_id, offset])

    def validate(self) -> List[str]:
        errors: List[str] = []

        present_count = 0
        for vpn, pte in enumerate(self.page_table):
            if pte.present:
                present_count += 1
                if pte.frame_id < 0 or pte.frame_id >= self.num_frames:
                    errors.append(f"vpn {vpn} has invalid frame id {pte.frame_id}")
                else:
                    back_vpn = self.frame_to_vpn[pte.frame_id]
                    if back_vpn != vpn:
                        errors.append(
                            f"vpn {vpn} points frame {pte.frame_id}, but frame maps to {back_vpn}"
                        )
            elif pte.frame_id != -1:
                errors.append(
                    f"vpn {vpn} not present but frame_id={pte.frame_id} should be -1"
                )

        mapped_frames = 0
        for frame_id, vpn in enumerate(self.frame_to_vpn):
            if vpn is not None:
                mapped_frames += 1
                pte = self.page_table[vpn]
                if not pte.present:
                    errors.append(
                        f"frame {frame_id} maps to vpn {vpn}, but pte is not present"
                    )

        if mapped_frames != present_count:
            errors.append(
                f"mapped_frames ({mapped_frames}) != present_pages ({present_count})"
            )

        if mapped_frames + len(self.free_frames) != self.num_frames:
            errors.append(
                "mapped frames + free frames does not equal total frame count"
            )

        if len(set(self.free_frames)) != len(self.free_frames):
            errors.append("duplicate frame id found in free_frames")

        for vpn, frame_id in self.tlb.items():
            if vpn < 0 or vpn >= self.num_virtual_pages:
                errors.append(f"TLB contains out-of-range vpn {vpn}")
                continue
            pte = self.page_table[vpn]
            if not pte.present or pte.frame_id != frame_id:
                errors.append(
                    f"stale TLB entry vpn={vpn}, frame={frame_id}, pte={pte}"
                )

        if self.stats.tlb_hits + self.stats.tlb_misses != self.stats.accesses:
            errors.append("TLB hit/miss counts do not sum to total accesses")

        if self.stats.reads + self.stats.writes != self.stats.accesses:
            errors.append("reads + writes does not sum to total accesses")

        return errors

    def snapshot(self) -> Dict[str, float]:
        resident_pages = sum(1 for pte in self.page_table if pte.present)
        dirty_pages = sum(1 for pte in self.page_table if pte.present and pte.dirty)
        free_frames = len(self.free_frames)
        return {
            "resident_pages": float(resident_pages),
            "dirty_pages": float(dirty_pages),
            "free_frames": float(free_frames),
            "accesses": float(self.stats.accesses),
            "reads": float(self.stats.reads),
            "writes": float(self.stats.writes),
            "tlb_hits": float(self.stats.tlb_hits),
            "tlb_misses": float(self.stats.tlb_misses),
            "page_faults": float(self.stats.page_faults),
            "evictions": float(self.stats.evictions),
            "write_backs": float(self.stats.write_backs),
        }


def build_trace(
    n_ops: int,
    num_virtual_pages: int,
    page_size: int,
    seed: int,
    write_prob: float,
) -> List[MemoryOp]:
    rng = np.random.default_rng(seed)
    trace: List[MemoryOp] = []

    hotspots = [2, 9, 17, 26, 34, 43, 51]
    phase_len = max(1, n_ops // 4)

    for i in range(n_ops):
        phase = min(i // phase_len, 3)
        center = hotspots[(phase * 2) % len(hotspots)]

        if float(rng.random()) < 0.82:
            vpn = int(np.clip(center + int(rng.integers(-3, 4)), 0, num_virtual_pages - 1))
        else:
            vpn = int(rng.integers(0, num_virtual_pages))

        offset = int(rng.integers(0, page_size))
        is_write = bool(float(rng.random()) < write_prob)
        value = int(rng.integers(0, 256))
        trace.append(MemoryOp(vaddr=vpn * page_size + offset, is_write=is_write, value=value))

    return trace


def run_trace(vm: VirtualMemoryManager, trace: Sequence[MemoryOp]) -> Tuple[List[str], List[float]]:
    errors: List[str] = []
    fault_rate_points: List[float] = []

    for idx, op in enumerate(trace, start=1):
        vm.access(op.vaddr, op.is_write, op.value)

        if idx % 200 == 0:
            fault_rate = vm.stats.page_faults / vm.stats.accesses
            fault_rate_points.append(float(fault_rate))

        if idx % 1000 == 0:
            errors.extend(vm.validate())

    errors.extend(vm.validate())
    return errors, fault_rate_points


def percentile_text(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    p50, p90, p99 = np.percentile(arr, [50, 90, 99])
    return f"p50={p50:.3f}, p90={p90:.3f}, p99={p99:.3f}"


def run_sanity_checks() -> List[str]:
    errors: List[str] = []

    vm = VirtualMemoryManager(
        num_virtual_pages=8,
        num_frames=2,
        page_size=64,
        tlb_capacity=2,
    )

    vm.access(0 * 64 + 1, True, 123)
    vm.access(1 * 64 + 2, True, 45)

    if vm.stats.page_faults != 2:
        errors.append(f"expected 2 page faults, got {vm.stats.page_faults}")

    vm.access(0 * 64 + 3, False)
    if vm.stats.page_faults != 2:
        errors.append("re-accessed resident page should not fault")

    vm.access(2 * 64 + 4, False)
    if vm.stats.page_faults != 3:
        errors.append("access to third page should trigger one more fault")

    if vm.stats.evictions != 1:
        errors.append(f"expected exactly one eviction, got {vm.stats.evictions}")

    if vm.stats.write_backs != 1:
        errors.append(f"expected one dirty write-back, got {vm.stats.write_backs}")

    errors.extend(vm.validate())
    return errors


def main() -> None:
    sanity_errors = run_sanity_checks()

    vm = VirtualMemoryManager(
        num_virtual_pages=64,
        num_frames=12,
        page_size=256,
        tlb_capacity=8,
    )

    n_ops = 6000
    trace = build_trace(
        n_ops=n_ops,
        num_virtual_pages=vm.num_virtual_pages,
        page_size=vm.page_size,
        seed=20260407,
        write_prob=0.31,
    )

    t0 = perf_counter()
    runtime_errors, checkpoints = run_trace(vm, trace)
    t1 = perf_counter()

    snap = vm.snapshot()
    all_errors = sanity_errors + runtime_errors + vm.validate()
    ok = len(all_errors) == 0

    tlb_hit_rate = vm.stats.tlb_hits / vm.stats.accesses if vm.stats.accesses else 0.0
    fault_rate = vm.stats.page_faults / vm.stats.accesses if vm.stats.accesses else 0.0

    print("=== Virtual Memory Management MVP ===")
    print(
        "Config: "
        f"virtual_pages={vm.num_virtual_pages}, "
        f"frames={vm.num_frames}, "
        f"page_size={vm.page_size}, "
        f"tlb_capacity={vm.tlb_capacity}"
    )
    print(f"Ops: total={vm.stats.accesses}, reads={vm.stats.reads}, writes={vm.stats.writes}")
    print(
        "TLB: "
        f"hits={vm.stats.tlb_hits}, misses={vm.stats.tlb_misses}, "
        f"hit_rate={tlb_hit_rate:.2%}"
    )
    print(
        "Paging: "
        f"faults={vm.stats.page_faults}, fault_rate={fault_rate:.2%}, "
        f"evictions={vm.stats.evictions}, write_backs={vm.stats.write_backs}"
    )
    print(
        "Resident: "
        f"pages={int(snap['resident_pages'])}, "
        f"dirty_pages={int(snap['dirty_pages'])}, "
        f"free_frames={int(snap['free_frames'])}"
    )
    print(f"Fault-rate checkpoints: {percentile_text(checkpoints)}")
    print(f"Elapsed: {(t1 - t0) * 1e3:.2f} ms")
    print(f"Validation: {'PASS' if ok else 'FAIL'}")

    if not ok:
        print("Errors:")
        for i, err in enumerate(all_errors[:10], start=1):
            print(f"  {i}. {err}")


if __name__ == "__main__":
    main()
