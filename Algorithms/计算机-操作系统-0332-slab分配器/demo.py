"""Slab allocator MVP.

This script implements a small slab cache simulator:
- fixed-size objects grouped into slabs
- slab states: empty / partial / full
- deterministic random workload (allocate/free)
- invariant checks and memory-fragmentation stats
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

Handle = Tuple[int, int]  # (slab_id, object_index)


@dataclass
class Slab:
    slab_id: int
    obj_size: int
    capacity: int
    free_list: List[int] = field(default_factory=list)
    in_use: List[bool] = field(default_factory=list)
    payloads: List[bytearray] = field(default_factory=list)

    @classmethod
    def create(cls, slab_id: int, obj_size: int, capacity: int) -> "Slab":
        if capacity <= 0:
            raise ValueError("slab capacity must be positive")
        free_list = list(range(capacity - 1, -1, -1))
        in_use = [False] * capacity
        payloads = [bytearray(obj_size) for _ in range(capacity)]
        return cls(
            slab_id=slab_id,
            obj_size=obj_size,
            capacity=capacity,
            free_list=free_list,
            in_use=in_use,
            payloads=payloads,
        )

    @property
    def inuse_count(self) -> int:
        return self.capacity - len(self.free_list)

    @property
    def is_full(self) -> bool:
        return self.inuse_count == self.capacity

    @property
    def is_empty(self) -> bool:
        return self.inuse_count == 0

    def allocate(self, payload: bytes) -> int:
        if not self.free_list:
            raise RuntimeError("allocate on full slab")
        idx = self.free_list.pop()
        self.in_use[idx] = True

        buf = self.payloads[idx]
        write_n = min(len(payload), self.obj_size)
        if write_n > 0:
            buf[:write_n] = payload[:write_n]
        if write_n < self.obj_size:
            buf[write_n:] = b"\x00" * (self.obj_size - write_n)
        return idx

    def free(self, idx: int) -> None:
        if idx < 0 or idx >= self.capacity:
            raise ValueError(f"object index {idx} out of range in slab {self.slab_id}")
        if not self.in_use[idx]:
            raise ValueError(
                f"double free or invalid free on slab {self.slab_id}, index {idx}"
            )
        self.in_use[idx] = False
        self.payloads[idx][:] = b"\xDD" * self.obj_size  # poison pattern
        self.free_list.append(idx)


@dataclass
class SlabCache:
    name: str
    obj_size: int
    slab_size: int = 4096
    max_empty_slabs: int = 2

    slabs: Dict[int, Slab] = field(default_factory=dict)
    empty_ids: Set[int] = field(default_factory=set)
    partial_ids: Set[int] = field(default_factory=set)
    full_ids: Set[int] = field(default_factory=set)

    next_slab_id: int = 0
    alloc_ops: int = 0
    free_ops: int = 0
    current_inuse_objects: int = 0
    peak_inuse_objects: int = 0
    slabs_created: int = 0
    slabs_reaped: int = 0

    def __post_init__(self) -> None:
        if self.obj_size <= 0:
            raise ValueError("obj_size must be positive")
        if self.slab_size <= 0:
            raise ValueError("slab_size must be positive")
        self.capacity_per_slab = self.slab_size // self.obj_size
        if self.capacity_per_slab <= 0:
            raise ValueError("obj_size larger than slab_size; capacity would be zero")

    def _create_slab(self) -> Slab:
        sid = self.next_slab_id
        self.next_slab_id += 1
        slab = Slab.create(sid, self.obj_size, self.capacity_per_slab)
        self.slabs[sid] = slab
        self.empty_ids.add(sid)
        self.slabs_created += 1
        return slab

    def _reclassify(self, slab: Slab) -> None:
        sid = slab.slab_id
        self.empty_ids.discard(sid)
        self.partial_ids.discard(sid)
        self.full_ids.discard(sid)
        if slab.is_full:
            self.full_ids.add(sid)
        elif slab.is_empty:
            self.empty_ids.add(sid)
        else:
            self.partial_ids.add(sid)

    def _pick_alloc_slab(self) -> Slab:
        if self.partial_ids:
            sid = min(self.partial_ids)
            return self.slabs[sid]
        if self.empty_ids:
            sid = min(self.empty_ids)
            return self.slabs[sid]
        return self._create_slab()

    def _reap_extra_empty_slabs(self) -> None:
        # Keep at most max_empty_slabs fully empty slabs cached.
        while len(self.empty_ids) > self.max_empty_slabs:
            sid = max(self.empty_ids)
            self.empty_ids.remove(sid)
            del self.slabs[sid]
            self.slabs_reaped += 1

    def allocate(self, payload: bytes) -> Handle:
        slab = self._pick_alloc_slab()
        idx = slab.allocate(payload)
        self._reclassify(slab)

        self.alloc_ops += 1
        self.current_inuse_objects += 1
        if self.current_inuse_objects > self.peak_inuse_objects:
            self.peak_inuse_objects = self.current_inuse_objects
        return (slab.slab_id, idx)

    def free(self, handle: Handle) -> None:
        sid, idx = handle
        slab = self.slabs.get(sid)
        if slab is None:
            raise ValueError(f"free on unknown slab id: {sid}")

        slab.free(idx)
        self._reclassify(slab)
        self._reap_extra_empty_slabs()

        self.free_ops += 1
        self.current_inuse_objects -= 1
        if self.current_inuse_objects < 0:
            raise RuntimeError("negative in-use object count")

    def validate(self) -> List[str]:
        errors: List[str] = []
        all_ids = set(self.slabs.keys())
        union_ids = self.empty_ids | self.partial_ids | self.full_ids
        overlap = (
            (self.empty_ids & self.partial_ids)
            | (self.empty_ids & self.full_ids)
            | (self.partial_ids & self.full_ids)
        )

        if overlap:
            errors.append(f"state sets overlap: {sorted(overlap)}")
        if union_ids != all_ids:
            errors.append(
                f"state sets mismatch; union={sorted(union_ids)}, slabs={sorted(all_ids)}"
            )

        computed_inuse = 0
        for sid, slab in self.slabs.items():
            computed_inuse += slab.inuse_count
            free_set = set(slab.free_list)
            if len(free_set) != len(slab.free_list):
                errors.append(f"duplicate entries in free list of slab {sid}")
            if slab.is_full and sid not in self.full_ids:
                errors.append(f"slab {sid} full but not in full_ids")
            if slab.is_empty and sid not in self.empty_ids:
                errors.append(f"slab {sid} empty but not in empty_ids")
            if (not slab.is_full) and (not slab.is_empty) and sid not in self.partial_ids:
                errors.append(f"slab {sid} partial but not in partial_ids")

        if computed_inuse != self.current_inuse_objects:
            errors.append(
                f"inuse mismatch: computed={computed_inuse}, tracked={self.current_inuse_objects}"
            )
        if self.alloc_ops < self.free_ops:
            errors.append(f"free_ops ({self.free_ops}) > alloc_ops ({self.alloc_ops})")
        return errors

    def snapshot(self) -> Dict[str, float]:
        active_slabs = len(self.slabs)
        reserved_bytes = active_slabs * self.slab_size
        inuse_bytes = self.current_inuse_objects * self.obj_size
        internal_frag_bytes = max(0, reserved_bytes - inuse_bytes)
        internal_frag_ratio = (
            internal_frag_bytes / reserved_bytes if reserved_bytes > 0 else 0.0
        )
        return {
            "active_slabs": float(active_slabs),
            "reserved_bytes": float(reserved_bytes),
            "inuse_bytes": float(inuse_bytes),
            "internal_frag_bytes": float(internal_frag_bytes),
            "internal_frag_ratio": float(internal_frag_ratio),
            "empty_slabs": float(len(self.empty_ids)),
            "partial_slabs": float(len(self.partial_ids)),
            "full_slabs": float(len(self.full_ids)),
        }


def run_sanity_checks() -> List[str]:
    errors: List[str] = []
    cache = SlabCache(name="sanity", obj_size=64, slab_size=1024, max_empty_slabs=1)

    handles: List[Handle] = []
    for i in range(cache.capacity_per_slab):
        handles.append(cache.allocate(bytes([i % 256])))
    if len(cache.full_ids) != 1:
        errors.append("expected one full slab after filling first slab")

    cache.free(handles[0])
    if not cache.partial_ids:
        errors.append("expected at least one partial slab after one free")
    reused = cache.allocate(b"\xAB\xCD")
    if reused[0] != handles[0][0]:
        errors.append("expected allocator to reuse the partially used slab first")

    try:
        cache.free((999, 0))
        errors.append("expected invalid slab free to raise ValueError")
    except ValueError:
        pass

    try:
        cache.free(reused)
        cache.free(reused)
        errors.append("expected double free to raise ValueError")
    except ValueError:
        pass

    errors.extend(cache.validate())
    return errors


def run_workload(
    cache: SlabCache, n_ops: int, alloc_prob: float, seed: int
) -> Tuple[List[str], List[float]]:
    rng = np.random.default_rng(seed)
    live_handles: List[Handle] = []
    errors: List[str] = []
    frag_history: List[float] = []

    for step in range(n_ops):
        should_alloc = (not live_handles) or (float(rng.random()) < alloc_prob)
        if should_alloc:
            payload_len = int(rng.integers(1, cache.obj_size + 1))
            payload = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()
            live_handles.append(cache.allocate(payload))
        else:
            ridx = int(rng.integers(0, len(live_handles)))
            handle = live_handles[ridx]
            live_handles[ridx] = live_handles[-1]
            live_handles.pop()
            cache.free(handle)

        if (step + 1) % 500 == 0:
            errors.extend(cache.validate())
        if (step + 1) % 100 == 0:
            frag_history.append(cache.snapshot()["internal_frag_ratio"])

    # Teardown phase: free all survivors.
    for handle in reversed(live_handles):
        cache.free(handle)
    errors.extend(cache.validate())
    frag_history.append(cache.snapshot()["internal_frag_ratio"])
    return errors, frag_history


def format_percentiles(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    p50, p90, p99 = np.percentile(arr, [50, 90, 99])
    return f"p50={p50:.2f}, p90={p90:.2f}, p99={p99:.2f}"


def main() -> None:
    sanity_errors = run_sanity_checks()

    cache = SlabCache(
        name="inode_cache",
        obj_size=192,
        slab_size=4096,
        max_empty_slabs=2,
    )

    n_ops = 8000
    alloc_prob = 0.63
    seed = 20260407

    t0 = perf_counter()
    workload_errors, frag_history = run_workload(
        cache=cache, n_ops=n_ops, alloc_prob=alloc_prob, seed=seed
    )
    t1 = perf_counter()

    snap = cache.snapshot()

    all_errors = sanity_errors + workload_errors + cache.validate()
    ok = len(all_errors) == 0

    print("=== Slab Allocator MVP ===")
    print(f"Cache name: {cache.name}")
    print(f"Object size: {cache.obj_size} B")
    print(f"Slab size: {cache.slab_size} B")
    print(f"Capacity per slab: {cache.capacity_per_slab}")
    print(f"Ops: alloc={cache.alloc_ops}, free={cache.free_ops}, total={n_ops}")
    print(f"Peak in-use objects: {cache.peak_inuse_objects}")
    print(
        "Active slab states: "
        f"empty={int(snap['empty_slabs'])}, "
        f"partial={int(snap['partial_slabs'])}, "
        f"full={int(snap['full_slabs'])}"
    )
    print(
        "Slab lifecycle: "
        f"created={cache.slabs_created}, "
        f"reaped={cache.slabs_reaped}, "
        f"active={int(snap['active_slabs'])}"
    )
    print(
        "Memory: "
        f"reserved={int(snap['reserved_bytes'])} B, "
        f"in-use={int(snap['inuse_bytes'])} B, "
        f"internal_frag={int(snap['internal_frag_bytes'])} B "
        f"({snap['internal_frag_ratio']:.2%})"
    )
    print(f"Fragmentation percentiles: {format_percentiles(frag_history)}")
    print(f"Elapsed: {(t1 - t0) * 1e3:.2f} ms")
    print(f"Validation: {'PASS' if ok else 'FAIL'}")

    if not ok:
        print("Errors:")
        for i, err in enumerate(all_errors[:10], start=1):
            print(f"  {i}. {err}")


if __name__ == "__main__":
    main()
