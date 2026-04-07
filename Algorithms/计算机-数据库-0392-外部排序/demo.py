"""External sort MVP: run generation + bounded fan-in multi-pass merge."""

from __future__ import annotations

import heapq
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class Record:
    """Sortable record with stable tie-breaker."""

    key: int
    seq: int


@dataclass(frozen=True)
class SortStats:
    """Basic metrics for one external sort run."""

    initial_runs: int
    merge_passes: int


def parse_record_line(line: str) -> Record:
    """Parse one CSV-like record line: key,seq."""
    key_text, seq_text = line.strip().split(",")
    return Record(key=int(key_text), seq=int(seq_text))


def format_record(record: Record) -> str:
    """Serialize one record to file line."""
    return f"{record.key},{record.seq}\n"


def generate_input_file(path: Path, count: int, seed: int = 2026) -> None:
    """Generate deterministic data with duplicated keys to test stability."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for seq in range(count):
            # Narrow key range deliberately to create many duplicate keys.
            key = rng.randint(-2_000, 2_000)
            f.write(format_record(Record(key=key, seq=seq)))


def _flush_chunk_to_run(chunk: list[Record], run_id: int, runs_dir: Path) -> Path:
    """Sort one in-memory chunk and flush to a run file."""
    chunk.sort(key=lambda rec: (rec.key, rec.seq))
    run_path = runs_dir / f"run_{run_id:05d}.txt"

    with run_path.open("w", encoding="utf-8") as f:
        for record in chunk:
            f.write(format_record(record))

    return run_path


def split_into_sorted_runs(input_path: Path, runs_dir: Path, chunk_size: int) -> list[Path]:
    """Read a large file incrementally and create sorted run files."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    runs_dir.mkdir(parents=True, exist_ok=True)

    run_paths: list[Path] = []
    chunk: list[Record] = []
    run_id = 0

    with input_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            chunk.append(parse_record_line(raw_line))

            if len(chunk) >= chunk_size:
                run_paths.append(_flush_chunk_to_run(chunk, run_id, runs_dir))
                run_id += 1
                chunk = []

    if chunk:
        run_paths.append(_flush_chunk_to_run(chunk, run_id, runs_dir))

    return run_paths


def merge_run_group(run_group: list[Path], output_path: Path) -> None:
    """Merge one group of sorted runs into a new sorted run file."""
    if not run_group:
        output_path.write_text("", encoding="utf-8")
        return

    files = [path.open("r", encoding="utf-8") for path in run_group]
    heap: list[tuple[int, int, int]] = []  # (key, seq, run_index)

    try:
        for run_index, file_obj in enumerate(files):
            line = file_obj.readline().strip()
            if line:
                record = parse_record_line(line)
                heapq.heappush(heap, (record.key, record.seq, run_index))

        with output_path.open("w", encoding="utf-8") as out:
            while heap:
                key, seq, run_index = heapq.heappop(heap)
                out.write(format_record(Record(key=key, seq=seq)))

                next_line = files[run_index].readline().strip()
                if next_line:
                    next_record = parse_record_line(next_line)
                    heapq.heappush(heap, (next_record.key, next_record.seq, run_index))
    finally:
        for file_obj in files:
            file_obj.close()


def chunk_paths(paths: list[Path], size: int) -> list[list[Path]]:
    """Split path list into contiguous groups."""
    return [paths[idx : idx + size] for idx in range(0, len(paths), size)]


def multi_pass_merge(run_paths: list[Path], merge_root: Path, fan_in: int) -> tuple[Path | None, int]:
    """Repeatedly merge runs with bounded fan-in until one run remains."""
    if fan_in < 2:
        raise ValueError("fan_in must be at least 2")

    if not run_paths:
        return None, 0

    current_runs = run_paths[:]
    pass_index = 0

    while len(current_runs) > 1:
        pass_index += 1
        pass_dir = merge_root / f"pass_{pass_index:02d}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        next_runs: list[Path] = []
        grouped_runs = chunk_paths(current_runs, fan_in)

        for group_index, run_group in enumerate(grouped_runs):
            merged_run = pass_dir / f"run_{group_index:05d}.txt"
            merge_run_group(run_group, merged_run)
            next_runs.append(merged_run)

        for old_run in current_runs:
            old_run.unlink(missing_ok=True)

        current_runs = next_runs

    return current_runs[0], pass_index


def external_sort(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    chunk_size: int,
    fan_in: int,
) -> SortStats:
    """Full external sort pipeline with temporary-file cleanup."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if fan_in < 2:
        raise ValueError("fan_in must be at least 2")

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    runs_pass0_dir = work_dir / "runs_pass0"
    merge_root = work_dir / "merge_runs"

    run_paths = split_into_sorted_runs(input_path, runs_pass0_dir, chunk_size)
    initial_runs = len(run_paths)

    final_run, merge_passes = multi_pass_merge(run_paths, merge_root, fan_in)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if final_run is None:
        output_path.write_text("", encoding="utf-8")
    else:
        shutil.move(str(final_run), str(output_path))

    shutil.rmtree(work_dir, ignore_errors=True)
    return SortStats(initial_runs=initial_runs, merge_passes=merge_passes)


def count_records(path: Path) -> int:
    """Count non-empty lines in a file."""
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def is_sorted_and_stable(path: Path) -> bool:
    """Check global order by key and stable order by seq for equal keys."""
    prev: Record | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cur = parse_record_line(line)
            if prev is not None:
                if cur.key < prev.key:
                    return False
                if cur.key == prev.key and cur.seq < prev.seq:
                    return False
            prev = cur
    return True


def sample_head_tail_keys(path: Path, n: int = 10) -> tuple[list[int], list[int]]:
    """Collect first/last n keys for quick inspection."""
    keys: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            keys.append(parse_record_line(line).key)

    if not keys:
        return [], []

    return keys[:n], keys[-n:]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "mvp_data"
    input_path = data_dir / "input_records.txt"
    output_path = data_dir / "output_sorted.txt"
    work_dir = data_dir / "tmp_external_sort"

    record_count = 12_000
    chunk_size = 700
    fan_in = 8

    generate_input_file(input_path, count=record_count, seed=2026)
    stats = external_sort(
        input_path=input_path,
        output_path=output_path,
        work_dir=work_dir,
        chunk_size=chunk_size,
        fan_in=fan_in,
    )

    records_in = count_records(input_path)
    records_out = count_records(output_path)
    stable_sorted = is_sorted_and_stable(output_path)
    head_keys, tail_keys = sample_head_tail_keys(output_path, n=10)

    print("External Sorting MVP")
    print(f"input_file       : {input_path}")
    print(f"output_file      : {output_path}")
    print(f"records_in       : {records_in}")
    print(f"records_out      : {records_out}")
    print(f"chunk_size       : {chunk_size}")
    print(f"fan_in           : {fan_in}")
    print(f"initial_runs     : {stats.initial_runs}")
    print(f"merge_passes     : {stats.merge_passes}")
    print(f"is_sorted_stable : {stable_sorted}")
    print(f"head_keys(10)    : {head_keys}")
    print(f"tail_keys(10)    : {tail_keys}")


if __name__ == "__main__":
    main()
