"""WAL (Write-Ahead Logging) MVP demo.

This script builds a tiny key-value store with:
- WAL append + fsync before snapshot update
- transaction commit marker (COMMIT)
- crash recovery that replays committed transactions only

Run:
    uv run python demo.py
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Operation:
    op: str
    key: str
    value: Any = None


class SimpleWALKV:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.wal_path = self.root / "wal.log"
        self.data_path = self.root / "data.json"

        self.state: dict[str, Any] = {}
        self.last_applied_lsn = 0
        self.active_ops: dict[str, list[Operation]] = {}

        self._load_snapshot()
        self.next_lsn = self._discover_next_lsn()
        self._wal_fp = self.wal_path.open("a", encoding="utf-8")

        self.recover()

    def close(self) -> None:
        self._wal_fp.close()

    def _load_snapshot(self) -> None:
        if not self.data_path.exists():
            self.state = {}
            self.last_applied_lsn = 0
            return

        with self.data_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.state = dict(payload.get("state", {}))
        self.last_applied_lsn = int(payload.get("last_applied_lsn", 0))

    def _discover_next_lsn(self) -> int:
        max_lsn = self.last_applied_lsn
        if self.wal_path.exists():
            with self.wal_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    max_lsn = max(max_lsn, int(rec.get("lsn", 0)))
        return max_lsn + 1

    def _append_log(self, record: dict[str, Any]) -> int:
        lsn = self.next_lsn
        self.next_lsn += 1
        record = dict(record)
        record["lsn"] = lsn

        self._wal_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._wal_fp.flush()
        os.fsync(self._wal_fp.fileno())
        return lsn

    def _persist_snapshot(self) -> None:
        payload = {
            "state": self.state,
            "last_applied_lsn": self.last_applied_lsn,
        }
        tmp_path = self.data_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.data_path)

    def _apply_op(self, op: Operation) -> None:
        if op.op == "SET":
            self.state[op.key] = op.value
        elif op.op == "DELETE":
            self.state.pop(op.key, None)
        else:
            raise ValueError(f"unknown op: {op.op}")

    def begin(self, txid: str) -> None:
        if txid in self.active_ops:
            raise ValueError(f"tx already active: {txid}")
        self.active_ops[txid] = []
        self._append_log({"txid": txid, "op": "BEGIN"})

    def set(self, txid: str, key: str, value: Any) -> None:
        if txid not in self.active_ops:
            raise ValueError(f"tx not active: {txid}")
        self.active_ops[txid].append(Operation(op="SET", key=key, value=value))
        self._append_log({"txid": txid, "op": "SET", "key": key, "value": value})

    def delete(self, txid: str, key: str) -> None:
        if txid not in self.active_ops:
            raise ValueError(f"tx not active: {txid}")
        self.active_ops[txid].append(Operation(op="DELETE", key=key))
        self._append_log({"txid": txid, "op": "DELETE", "key": key})

    def commit(self, txid: str) -> None:
        if txid not in self.active_ops:
            raise ValueError(f"tx not active: {txid}")

        commit_lsn = self._append_log({"txid": txid, "op": "COMMIT"})
        for op in self.active_ops[txid]:
            self._apply_op(op)
        del self.active_ops[txid]

        self.last_applied_lsn = commit_lsn
        self._persist_snapshot()

    def recover(self) -> None:
        if not self.wal_path.exists():
            return

        pending_ops: dict[str, list[Operation]] = {}
        commit_lsn: dict[str, int] = {}

        with self.wal_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                txid = rec.get("txid")
                op = rec.get("op")
                lsn = int(rec.get("lsn", 0))
                if not txid or not op:
                    continue

                if op == "BEGIN":
                    pending_ops.setdefault(txid, [])
                elif op == "SET":
                    pending_ops.setdefault(txid, []).append(
                        Operation(op="SET", key=rec["key"], value=rec.get("value"))
                    )
                elif op == "DELETE":
                    pending_ops.setdefault(txid, []).append(
                        Operation(op="DELETE", key=rec["key"])
                    )
                elif op == "COMMIT":
                    commit_lsn[txid] = lsn

        recovered = False
        ordered_commits = sorted(commit_lsn.items(), key=lambda item: item[1])
        for txid, c_lsn in ordered_commits:
            if c_lsn <= self.last_applied_lsn:
                continue
            for op in pending_ops.get(txid, []):
                self._apply_op(op)
            self.last_applied_lsn = c_lsn
            recovered = True

        if recovered:
            self._persist_snapshot()

    def inject_uncommitted_for_demo(self, txid: str, key: str, value: Any) -> None:
        # Used only by demo: write partial tx records without COMMIT.
        self._append_log({"txid": txid, "op": "BEGIN"})
        self._append_log({"txid": txid, "op": "SET", "key": key, "value": value})


def main() -> None:
    base = Path(__file__).resolve().parent
    db_dir = base / "tmp_wal_db"

    if db_dir.exists():
        shutil.rmtree(db_dir)

    print("[phase-1] start DB and commit two transactions")
    db = SimpleWALKV(db_dir)

    db.begin("tx1")
    db.set("tx1", "x", 1)
    db.set("tx1", "y", 2)
    db.commit("tx1")

    db.begin("tx2")
    db.set("tx2", "x", 10)
    db.delete("tx2", "y")
    db.commit("tx2")

    print(f"state before crash injection: {db.state}")

    print("[phase-2] inject uncommitted tx and simulate crash")
    db.inject_uncommitted_for_demo("tx3", "z", 999)
    db.close()

    print("[phase-3] restart and recover")
    recovered = SimpleWALKV(db_dir)
    print(f"state after recovery: {recovered.state}")

    assert recovered.state == {"x": 10}, (
        "recovery invariant failed: uncommitted tx should be ignored"
    )
    print("recovery check passed: uncommitted tx was discarded")

    recovered.close()


if __name__ == "__main__":
    main()
