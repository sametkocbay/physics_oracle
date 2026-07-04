#!/usr/bin/env python3
"""Delete case folders listed in rejection_log.csv from the cases/ directory.

Every row in rejection_log.csv is a case that failed generation or QC (an
exception during extraction, or residuals that never dropped 4 orders). Their
of_case folders are dead weight, so this removes cases/<case_id> for each.

Dry-run by default — prints what it *would* delete. Pass --apply to delete.

    # from /tokp/work/sako/physics_oracle/dataset
    python prune_rejected.py                 # preview
    python prune_rejected.py --apply         # actually delete
    python prune_rejected.py -d /path/to/dataset --apply
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset-dir", type=Path, default=Path.cwd(),
                    help="dir holding rejection_log.csv and cases/ (default: cwd)")
    ap.add_argument("--rejection-log", type=Path, default=None,
                    help="override path to rejection_log.csv")
    ap.add_argument("--cases-dir", type=Path, default=None,
                    help="override path to cases/ directory")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is a dry-run preview)")
    args = ap.parse_args()

    log_path = args.rejection_log or (args.dataset_dir / "rejection_log.csv")
    cases_dir = args.cases_dir or (args.dataset_dir / "cases")

    if not log_path.is_file():
        print(f"error: rejection log not found: {log_path}", file=sys.stderr)
        return 1
    if not cases_dir.is_dir():
        print(f"error: cases dir not found: {cases_dir}", file=sys.stderr)
        return 1

    # Collect unique case_ids from the log (a case can be listed more than once).
    case_ids: list[str] = []
    seen: set[str] = set()
    with log_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "case_id" not in reader.fieldnames:
            print(f"error: no 'case_id' column in {log_path} "
                  f"(header: {reader.fieldnames})", file=sys.stderr)
            return 1
        for row in reader:
            cid = (row.get("case_id") or "").strip()
            if cid and cid not in seen:
                seen.add(cid)
                case_ids.append(cid)

    to_delete = [(cid, cases_dir / cid) for cid in case_ids]
    present = [(cid, p) for cid, p in to_delete if p.is_dir()]
    missing = [cid for cid, p in to_delete if not p.is_dir()]

    print(f"rejection log : {log_path}")
    print(f"cases dir     : {cases_dir}")
    print(f"rejected cases: {len(case_ids)}   present: {len(present)}   "
          f"already gone: {len(missing)}")
    print()

    if not present:
        print("Nothing to delete.")
        return 0

    verb = "Deleting" if args.apply else "[dry-run] would delete"
    freed = 0
    for cid, path in present:
        print(f"  {verb}: {path}")
        if args.apply:
            shutil.rmtree(path)
            freed += 1

    print()
    if args.apply:
        print(f"Deleted {freed} case folder(s).")
    else:
        print(f"{len(present)} folder(s) would be deleted. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
