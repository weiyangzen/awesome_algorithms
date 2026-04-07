"""External merge sort MVP for large-file sorting with bounded memory."""

from __future__ import annotations

import heapq
import random
from pathlib import Path


def generate_input_file(path: Path, count: int, seed: int = 42) -> None:
    """Generate a deterministic integer dataset (one value per line)."""
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(count):
            value = rng.randint(-100_000, 100_000)
            f.write(f"{value}\n")


def _flush_chunk_to_run(chunk: list[int], run_id: int, temp_dir: Path) -> Path:
    """Sort one in-memory chunk and persist it as a run file."""
    chunk.sort()
    run_path = temp_dir / f"run_{run_id:04d}.txt"
    with run_path.open("w", encoding="utf-8") as f:
        for value in chunk:
            f.write(f"{value}\n")
    return run_path


def split_into_sorted_runs(input_path: Path, temp_dir: Path, chunk_size: int) -> list[Path]:
    """Split a big file into sorted run files with bounded chunk memory."""
    run_paths: list[Path] = []
    chunk: list[int] = []
    run_id = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk.append(int(line))
            if len(chunk) >= chunk_size:
                run_paths.append(_flush_chunk_to_run(chunk, run_id, temp_dir))
                run_id += 1
                chunk = []

    if chunk:
        run_paths.append(_flush_chunk_to_run(chunk, run_id, temp_dir))

    return run_paths


def merge_sorted_runs(run_paths: list[Path], output_path: Path) -> None:
    """K-way merge all sorted runs into one sorted output file."""
    if not run_paths:
        output_path.write_text("", encoding="utf-8")
        return

    files = [p.open("r", encoding="utf-8") for p in run_paths]
    heap: list[tuple[int, int]] = []

    try:
        for run_index, f in enumerate(files):
            line = f.readline().strip()
            if line:
                heapq.heappush(heap, (int(line), run_index))

        with output_path.open("w", encoding="utf-8") as out:
            while heap:
                value, run_index = heapq.heappop(heap)
                out.write(f"{value}\n")

                next_line = files[run_index].readline().strip()
                if next_line:
                    heapq.heappush(heap, (int(next_line), run_index))
    finally:
        for f in files:
            f.close()


def external_merge_sort(
    input_path: Path,
    output_path: Path,
    temp_dir: Path,
    chunk_size: int,
) -> dict[str, int]:
    """Run full external merge sort and return basic stats."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    temp_dir.mkdir(parents=True, exist_ok=True)
    run_paths = split_into_sorted_runs(input_path, temp_dir, chunk_size)
    merge_sorted_runs(run_paths, output_path)

    for p in run_paths:
        p.unlink(missing_ok=True)

    return {"run_count": len(run_paths)}


def is_file_sorted(path: Path) -> bool:
    """Check whether file content is non-decreasing."""
    prev: int | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = int(line.strip())
            if prev is not None and value < prev:
                return False
            prev = value
    return True


def count_records(path: Path) -> int:
    """Count non-empty lines."""
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def sample_head_tail(path: Path, n: int = 8) -> tuple[list[int], list[int]]:
    """Collect first/last n values for quick inspection."""
    values: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))
    if not values:
        return [], []
    return values[:n], values[-n:]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "mvp_data"
    temp_dir = data_dir / "tmp_runs"
    input_path = data_dir / "input_numbers.txt"
    output_path = data_dir / "output_sorted.txt"

    data_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    record_count = 5_000
    chunk_size = 512

    generate_input_file(input_path, count=record_count, seed=2026)
    stats = external_merge_sort(
        input_path=input_path,
        output_path=output_path,
        temp_dir=temp_dir,
        chunk_size=chunk_size,
    )

    sorted_ok = is_file_sorted(output_path)
    head, tail = sample_head_tail(output_path, n=8)

    print("External Merge Sort MVP")
    print(f"input_file   : {input_path}")
    print(f"output_file  : {output_path}")
    print(f"records      : {count_records(input_path)}")
    print(f"chunk_size   : {chunk_size}")
    print(f"run_count    : {stats['run_count']}")
    print(f"is_sorted    : {sorted_ok}")
    print(f"head(8)      : {head}")
    print(f"tail(8)      : {tail}")


if __name__ == "__main__":
    main()
