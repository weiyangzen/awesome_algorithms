"""ARIES recovery algorithm MVP.

This demo intentionally models three ARIES phases on a tiny page-based KV store:
1) Analysis
2) Redo (repeat history)
3) Undo (with CLR records)

Run:
    uv run python demo.py
"""

from __future__ import annotations

import heapq
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class AriesKVEngine:
    """Small educational ARIES-like engine for crash recovery demos."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.wal_path = base_dir / "wal.jsonl"
        self.data_path = base_dir / "data_pages.json"
        self.master_path = base_dir / "master.json"

        self.log: List[Dict[str, Any]] = self._load_wal()
        self.record_index: Dict[int, Dict[str, Any]] = {int(r["lsn"]): r for r in self.log}
        self.next_lsn = (max(self.record_index) + 1) if self.record_index else 1

        self.disk_pages: Dict[str, Dict[str, int]] = self._load_disk_pages()
        self.buffer_pages: Dict[str, Dict[str, int]] = {
            pid: {"value": page["value"], "page_lsn": page["page_lsn"]}
            for pid, page in self.disk_pages.items()
        }

        self.master = self._load_master()

        # Runtime structures used while producing log records.
        self.tx_table: Dict[int, Dict[str, Any]] = {}
        self.dirty_page_table: Dict[str, int] = {}

    # -----------------------
    # Persistent I/O helpers
    # -----------------------
    def _load_wal(self) -> List[Dict[str, Any]]:
        if not self.wal_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.wal_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _load_disk_pages(self) -> Dict[str, Dict[str, int]]:
        if not self.data_path.exists():
            return {}
        with self.data_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        pages = payload.get("pages", {})
        normalized: Dict[str, Dict[str, int]] = {}
        for pid, page in pages.items():
            normalized[pid] = {
                "value": int(page.get("value", 0)),
                "page_lsn": int(page.get("page_lsn", 0)),
            }
        return normalized

    def _load_master(self) -> Dict[str, Any]:
        if not self.master_path.exists():
            return {"last_checkpoint_lsn": None}
        with self.master_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return {"last_checkpoint_lsn": payload.get("last_checkpoint_lsn")}

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _persist_disk_pages(self) -> None:
        self._atomic_write_json(self.data_path, {"pages": self.disk_pages})

    def _persist_master(self) -> None:
        self._atomic_write_json(self.master_path, self.master)

    def _append_log(self, record: Dict[str, Any]) -> int:
        rec = dict(record)
        rec["lsn"] = self.next_lsn
        self.next_lsn += 1

        self.log.append(rec)
        self.record_index[int(rec["lsn"])] = rec

        with self.wal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        return int(rec["lsn"])

    # -----------------------
    # Runtime transaction API
    # -----------------------
    def _ensure_page_in_buffer(self, page_id: str) -> Dict[str, int]:
        if page_id not in self.buffer_pages:
            disk = self.disk_pages.get(page_id, {"value": 0, "page_lsn": 0})
            self.buffer_pages[page_id] = {
                "value": int(disk["value"]),
                "page_lsn": int(disk["page_lsn"]),
            }
        return self.buffer_pages[page_id]

    def begin(self, txid: int) -> int:
        lsn = self._append_log({"type": "BEGIN", "txid": txid, "prev_lsn": None})
        self.tx_table[txid] = {"status": "RUNNING", "last_lsn": lsn}
        return lsn

    def update(self, txid: int, page_id: str, delta: int) -> int:
        if txid not in self.tx_table:
            raise ValueError(f"Transaction {txid} does not exist")

        tx = self.tx_table[txid]
        page = self._ensure_page_in_buffer(page_id)
        before = int(page["value"])
        after = before + int(delta)

        lsn = self._append_log(
            {
                "type": "UPDATE",
                "txid": txid,
                "prev_lsn": tx["last_lsn"],
                "page_id": page_id,
                "before": before,
                "after": after,
            }
        )

        page["value"] = after
        page["page_lsn"] = lsn
        tx["last_lsn"] = lsn
        tx["status"] = "RUNNING"

        self.dirty_page_table.setdefault(page_id, lsn)
        return lsn

    def commit(self, txid: int, write_end: bool = True) -> int:
        if txid not in self.tx_table:
            raise ValueError(f"Transaction {txid} does not exist")

        tx = self.tx_table[txid]
        commit_lsn = self._append_log(
            {"type": "COMMIT", "txid": txid, "prev_lsn": tx["last_lsn"]}
        )
        tx["status"] = "COMMITTING"
        tx["last_lsn"] = commit_lsn

        if write_end:
            end_lsn = self._append_log(
                {"type": "END", "txid": txid, "prev_lsn": tx["last_lsn"]}
            )
            self.tx_table.pop(txid, None)
            return end_lsn

        return commit_lsn

    def flush_page(self, page_id: str) -> None:
        page = self._ensure_page_in_buffer(page_id)
        self.disk_pages[page_id] = {
            "value": int(page["value"]),
            "page_lsn": int(page["page_lsn"]),
        }
        self.dirty_page_table.pop(page_id, None)
        self._persist_disk_pages()

    def create_checkpoint(self) -> Tuple[int, int]:
        begin_lsn = self._append_log({"type": "BEGIN_CKPT"})

        att_snapshot: Dict[str, Dict[str, Any]] = {}
        for txid, entry in self.tx_table.items():
            att_snapshot[str(txid)] = {
                "status": entry["status"],
                "last_lsn": int(entry["last_lsn"]),
            }

        dpt_snapshot = {pid: int(rec_lsn) for pid, rec_lsn in self.dirty_page_table.items()}

        end_lsn = self._append_log(
            {
                "type": "END_CKPT",
                "begin_lsn": begin_lsn,
                "att": att_snapshot,
                "dpt": dpt_snapshot,
            }
        )

        self.master["last_checkpoint_lsn"] = begin_lsn
        self._persist_master()
        return begin_lsn, end_lsn

    def crash(self) -> None:
        # Simulate volatile-memory loss.
        self.buffer_pages = {}
        self.tx_table = {}
        self.dirty_page_table = {}

    # -----------------------
    # Recovery logic
    # -----------------------
    def _find_scan_start_index(self) -> int:
        start_lsn = self.master.get("last_checkpoint_lsn")
        if start_lsn is None:
            return 0
        for idx, rec in enumerate(self.log):
            if int(rec["lsn"]) == int(start_lsn):
                return idx
        return 0

    def _analysis_phase(self) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
        scan_start = self._find_scan_start_index()
        records = list(self.log[scan_start:])

        tx_table: Dict[int, Dict[str, Any]] = {}
        dpt: Dict[str, int] = {}

        for rec in records:
            rtype = rec["type"]

            if rtype == "END_CKPT":
                for txid_str, entry in rec.get("att", {}).items():
                    txid = int(txid_str)
                    old = tx_table.get(txid)
                    if old is None or int(entry["last_lsn"]) >= int(old["last_lsn"]):
                        tx_table[txid] = {
                            "status": entry["status"],
                            "last_lsn": int(entry["last_lsn"]),
                        }
                for pid, rec_lsn in rec.get("dpt", {}).items():
                    dpt[pid] = min(int(rec_lsn), int(dpt.get(pid, rec_lsn)))
                continue

            txid = rec.get("txid")
            if txid is not None:
                txid = int(txid)

            if rtype == "BEGIN":
                tx_table[txid] = {"status": "RUNNING", "last_lsn": int(rec["lsn"])}
            elif rtype == "UPDATE":
                tx_table.setdefault(txid, {"status": "RUNNING", "last_lsn": int(rec["lsn"])})
                tx_table[txid]["last_lsn"] = int(rec["lsn"])
                page_id = rec["page_id"]
                dpt.setdefault(page_id, int(rec["lsn"]))
            elif rtype == "CLR":
                tx_table.setdefault(txid, {"status": "ABORTING", "last_lsn": int(rec["lsn"])})
                tx_table[txid]["status"] = "ABORTING"
                tx_table[txid]["last_lsn"] = int(rec["lsn"])
                page_id = rec["page_id"]
                dpt.setdefault(page_id, int(rec["lsn"]))
            elif rtype == "COMMIT":
                tx_table.setdefault(txid, {"status": "COMMITTING", "last_lsn": int(rec["lsn"])})
                tx_table[txid]["status"] = "COMMITTING"
                tx_table[txid]["last_lsn"] = int(rec["lsn"])
            elif rtype == "ABORT":
                tx_table.setdefault(txid, {"status": "ABORTING", "last_lsn": int(rec["lsn"])})
                tx_table[txid]["status"] = "ABORTING"
                tx_table[txid]["last_lsn"] = int(rec["lsn"])
            elif rtype == "END":
                tx_table.pop(txid, None)

        # Finalization as in ARIES analysis:
        # - RUNNING -> ABORTING (write ABORT)
        # - COMMITTING -> END
        for txid in sorted(list(tx_table.keys())):
            status = tx_table[txid]["status"]
            if status == "RUNNING":
                abort_lsn = self._append_log(
                    {
                        "type": "ABORT",
                        "txid": txid,
                        "prev_lsn": tx_table[txid]["last_lsn"],
                    }
                )
                tx_table[txid]["status"] = "ABORTING"
                tx_table[txid]["last_lsn"] = abort_lsn
            elif status == "COMMITTING":
                end_lsn = self._append_log(
                    {
                        "type": "END",
                        "txid": txid,
                        "prev_lsn": tx_table[txid]["last_lsn"],
                    }
                )
                _ = end_lsn
                tx_table.pop(txid, None)

        losers = {
            txid: entry
            for txid, entry in tx_table.items()
            if entry["status"] == "ABORTING"
        }

        print("[Analysis] start_index=", scan_start)
        print("[Analysis] DPT=", dpt)
        print("[Analysis] losers=", {k: v["last_lsn"] for k, v in losers.items()})
        return losers, dpt

    def _redo_phase(self, dpt: Dict[str, int]) -> int:
        if not dpt:
            print("[Redo] DPT is empty, skip redo")
            return 0

        redo_start = min(dpt.values())
        applied = 0

        for rec in self.log:
            lsn = int(rec["lsn"])
            if lsn < redo_start:
                continue
            if rec["type"] not in {"UPDATE", "CLR"}:
                continue

            page_id = rec["page_id"]
            rec_lsn = dpt.get(page_id)
            if rec_lsn is None or lsn < rec_lsn:
                continue

            page = self.disk_pages.get(page_id, {"value": 0, "page_lsn": 0})
            if int(page["page_lsn"]) >= lsn:
                continue

            self.disk_pages[page_id] = {"value": int(rec["after"]), "page_lsn": lsn}
            applied += 1

        print(f"[Redo] redo_start_lsn={redo_start}, applied={applied}")
        return applied

    def _undo_phase(self, losers: Dict[int, Dict[str, Any]]) -> int:
        if not losers:
            print("[Undo] no loser transaction")
            return 0

        to_undo: List[Tuple[int, int]] = []
        for txid, entry in losers.items():
            heapq.heappush(to_undo, (-int(entry["last_lsn"]), txid))

        undone_updates = 0

        while to_undo:
            neg_lsn, txid = heapq.heappop(to_undo)
            lsn = -neg_lsn
            rec = self.record_index.get(lsn)
            if rec is None:
                raise RuntimeError(f"Missing log record for LSN={lsn}")

            rtype = rec["type"]

            if rtype == "UPDATE":
                clr_lsn = self._append_log(
                    {
                        "type": "CLR",
                        "txid": txid,
                        "prev_lsn": losers[txid]["last_lsn"],
                        "page_id": rec["page_id"],
                        "before": self.disk_pages.get(rec["page_id"], {"value": 0})["value"],
                        "after": int(rec["before"]),
                        "undo_next_lsn": rec.get("prev_lsn"),
                    }
                )
                losers[txid]["last_lsn"] = clr_lsn

                self.disk_pages[rec["page_id"]] = {
                    "value": int(rec["before"]),
                    "page_lsn": clr_lsn,
                }
                undone_updates += 1

                next_lsn = rec.get("prev_lsn")
            elif rtype == "CLR":
                next_lsn = rec.get("undo_next_lsn")
            else:
                next_lsn = rec.get("prev_lsn")

            if next_lsn is None:
                end_lsn = self._append_log(
                    {
                        "type": "END",
                        "txid": txid,
                        "prev_lsn": losers[txid]["last_lsn"],
                    }
                )
                _ = end_lsn
                losers.pop(txid, None)
            else:
                heapq.heappush(to_undo, (-int(next_lsn), txid))

        print(f"[Undo] undone_updates={undone_updates}")
        return undone_updates

    def recover(self) -> None:
        losers, dpt = self._analysis_phase()
        self._redo_phase(dpt)
        self._undo_phase(losers)
        self._persist_disk_pages()

    def print_disk_pages(self, title: str) -> None:
        ordered = {pid: self.disk_pages[pid] for pid in sorted(self.disk_pages)}
        print(title, ordered)


def reset_demo_files(base_dir: Path) -> None:
    for name in ["wal.jsonl", "data_pages.json", "master.json"]:
        path = base_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    reset_demo_files(base_dir)

    print("=== Phase 0: Build crash scenario ===")
    engine = AriesKVEngine(base_dir)

    engine.begin(1)
    engine.update(1, "P1", 10)
    engine.commit(1, write_end=True)
    engine.flush_page("P1")

    ckpt_begin, ckpt_end = engine.create_checkpoint()
    print(f"Checkpoint created: BEGIN_CKPT LSN={ckpt_begin}, END_CKPT LSN={ckpt_end}")

    engine.begin(2)
    engine.update(2, "P1", 5)
    engine.flush_page("P1")  # STEAL: flushes uncommitted change to disk.
    engine.update(2, "P2", 7)

    engine.begin(3)
    engine.update(3, "P3", 4)
    engine.commit(3, write_end=False)  # COMMIT written, END missing due to crash.

    engine.print_disk_pages("Disk pages right before crash:")
    engine.crash()

    print("\n=== Phase 1: Restart and ARIES recovery ===")
    recovered = AriesKVEngine(base_dir)
    recovered.print_disk_pages("Disk pages at restart:")
    recovered.recover()
    recovered.print_disk_pages("Disk pages after recovery:")

    p1 = recovered.disk_pages.get("P1", {"value": None})["value"]
    p2 = recovered.disk_pages.get("P2", {"value": None})["value"]
    p3 = recovered.disk_pages.get("P3", {"value": None})["value"]

    assert p1 == 10, f"P1 expected 10, got {p1}"
    assert p2 == 0, f"P2 expected 0, got {p2}"
    assert p3 == 4, f"P3 expected 4, got {p3}"

    print("\nAssertions passed:")
    print("- uncommitted T2 effects were undone (P1 rollback from 15 -> 10, P2 rollback 7 -> 0)")
    print("- committed T3 effect survived through REDO (P3 recovered to 4)")


if __name__ == "__main__":
    main()
