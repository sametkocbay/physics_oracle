"""§7 — Run quality-control checks on a generated case.

Reads fields.h5, convergence.h5 from the case directory and applies the
threshold table from §7.  Returns (accepted, reasons, flags) and appends a
row to rejection_log.csv when a case is rejected.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import h5py
import numpy as np

from common import REJECTION_LOG_PATH, setup_logging

LOG = setup_logging()


# ---------------------------------------------------------------------------
# Threshold table  (§7)
# ---------------------------------------------------------------------------

ORDERS_DROP_MIN = 4.0
Y_PLUS_MAX = 5.0
ITER_LIMIT = 5000


def quality_check(case_dir: Path, max_iter: int = ITER_LIMIT) -> dict:
    fields_path = case_dir / "fields.h5"
    conv_path = case_dir / "convergence.h5"

    rejections: list[str] = []
    flags: list[str] = []

    if not fields_path.exists() or not conv_path.exists():
        return {
            "accepted": False, "rejections": ["missing_outputs"], "flags": flags,
            "metrics": {},
        }

    with h5py.File(fields_path, "r") as h:
        k = h["k"][:]
        omega = h["omega"][:]
    with h5py.File(conv_path, "r") as h:
        y_plus = h["y_plus"][:] if "y_plus" in h else np.array([])
        iters_total = int(h.attrs.get("iterations_total", 0))
        iters_to_conv = int(h.attrs.get("iterations_to_convergence", 0))
        converged = bool(h.attrs.get("converged", False))
        drops = {key.replace("orders_drop_", ""): float(h.attrs[key])
                 for key in h.attrs if key.startswith("orders_drop_")}

    metrics = {
        "iterations_total": iters_total,
        "iterations_to_convergence": iters_to_conv,
        "n_iter": iters_total,
        "converged": converged,
        "min_k": float(np.min(k)) if k.size else None,
        "min_omega": float(np.min(omega)) if omega.size else None,
        "max_y_plus": float(np.nanmax(y_plus)) if y_plus.size and not np.all(np.isnan(y_plus)) else None,
        "drops": drops,
    }

    # §7 Residuals dropped < 4 orders → reject
    if any(v < ORDERS_DROP_MIN for v in drops.values()):
        rejections.append("residuals_under_4_orders")
    if not drops:
        rejections.append("no_residuals_parsed")

    # §7 Negative k anywhere
    if k.size and np.any(k < 0):
        rejections.append("negative_k")

    # §7 Negative omega anywhere
    if omega.size and np.any(omega < 0):
        rejections.append("negative_omega")

    # §7 y+ > 5 → reject
    if y_plus.size and np.nanmax(y_plus) > Y_PLUS_MAX:
        rejections.append("y_plus_over_5")

    # §7 > 5000 iterations → flag (implies non-convergence)
    if iters_total >= max_iter:
        flags.append("max_iterations_hit")
    if not converged:
        flags.append("did_not_converge")

    return {
        "accepted": len(rejections) == 0,
        "rejections": rejections,
        "flags": flags,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Rejection log
# ---------------------------------------------------------------------------

def append_rejection(case_id: str, reasons: list[str],
                     log_path: Path = REJECTION_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with log_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["case_id", "reason", "timestamp"])
        w.writerow([
            case_id,
            ";".join(reasons) if reasons else "unknown",
            dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run QC on a single case directory.")
    p.add_argument("case_dir", type=Path)
    p.add_argument("--case-id", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    case_id = args.case_id or args.case_dir.name
    result = quality_check(args.case_dir)
    LOG.info("[%s] accepted=%s flags=%s reject=%s", case_id,
             result["accepted"], result["flags"], result["rejections"])
    if not result["accepted"]:
        append_rejection(case_id, result["rejections"])
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
