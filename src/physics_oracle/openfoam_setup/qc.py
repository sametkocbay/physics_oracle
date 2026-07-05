"""§7 — Run quality-control checks on a generated case.

Thresholds are loaded from configs/postprocess.yaml at module import time.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import yaml

from physics_oracle.core.logging import setup_logging
from physics_oracle.core.paths import POSTPROCESS_CONFIG_PATH, REJECTION_LOG_PATH

LOG = setup_logging()


@lru_cache(maxsize=1)
def _qc_config() -> dict:
    return yaml.safe_load(POSTPROCESS_CONFIG_PATH.read_text())["qc"]


def _force_stationarity(cl: np.ndarray, cd: np.ndarray, window: int,
                        std_rel_max: float, drift_rel_max: float) -> dict:
    """Trailing-window stationarity of the force coefficients.

    A case is 'stationary' when BOTH Cl and Cd have relative scatter
    (std/|mean|) and half-window mean drift below the bounds over the last
    ``window`` iterations.  |mean| is floored to avoid a symmetric-case
    Cl ~ 0 blowing up the relative measure."""
    out: dict = {"stationary": False}
    if cl.size < window or cd.size < window:
        return out
    half = window // 2
    for name, hist in (("cl", cl), ("cd", cd)):
        tail = hist[-window:]
        mean = float(np.mean(tail))
        denom = max(abs(mean), 1e-3)
        out[f"{name}_mean"] = mean
        out[f"{name}_std_rel"] = float(np.std(tail) / denom)
        out[f"{name}_drift_rel"] = float(
            abs(np.mean(tail[half:]) - np.mean(tail[:half])) / denom)
    out["stationary"] = all(
        out[f"{n}_{m}_rel"] <= bound
        for n in ("cl", "cd")
        for m, bound in (("std", std_rel_max), ("drift", drift_rel_max))
    )
    return out


def quality_check(case_dir: Path, max_iter: int | None = None) -> dict:
    cfg = _qc_config()
    orders_drop_min = float(cfg["orders_drop_min"])
    orders_drop_hard_min = float(cfg.get("orders_drop_hard_min", orders_drop_min))
    y_plus_max = float(cfg["y_plus_max"])
    iter_limit = int(max_iter if max_iter is not None else cfg["iter_limit"])

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
        cl_hist = h["cl_history"][:] if "cl_history" in h else np.array([])
        cd_hist = h["cd_history"][:] if "cd_history" in h else np.array([])

    force_stat = _force_stationarity(
        cl_hist, cd_hist,
        window=int(cfg.get("force_tail_window", 500)),
        std_rel_max=float(cfg.get("force_std_rel_max", 0.02)),
        drift_rel_max=float(cfg.get("force_drift_rel_max", 0.02)),
    )

    metrics = {
        "iterations_total": iters_total,
        "iterations_to_convergence": iters_to_conv,
        "n_iter": iters_total,
        "converged": converged,
        "min_k": float(np.min(k)) if k.size else None,
        "min_omega": float(np.min(omega)) if omega.size else None,
        "max_y_plus": float(np.nanmax(y_plus)) if y_plus.size and not np.all(np.isnan(y_plus)) else None,
        "drops": drops,
        "force_stationarity": force_stat,
    }

    # Residual gate with a force-stationarity gray zone:
    #   min_drop >= orders_drop_min                    -> pass
    #   hard_min <= min_drop < orders_drop_min         -> pass ONLY if the
    #       trailing Cl/Cd are stationary (slow p-norm convergence at high Re
    #       can hold a physically steady case just under the full gate)
    #   min_drop < hard_min                            -> reject (empirically
    #       these are genuinely unsteady flows)
    if not drops:
        rejections.append("no_residuals_parsed")
    else:
        min_drop = min(drops.values())
        if min_drop >= orders_drop_min:
            pass
        elif min_drop >= orders_drop_hard_min and force_stat["stationary"]:
            flags.append("gray_zone_force_accepted")
        else:
            rejections.append("residuals_under_4_orders")
            if min_drop >= orders_drop_hard_min:
                rejections.append("forces_not_stationary")
    if k.size and np.any(k < 0):
        rejections.append("negative_k")
    if omega.size and np.any(omega < 0):
        rejections.append("negative_omega")
    if y_plus.size and np.nanmax(y_plus) > y_plus_max:
        rejections.append("y_plus_over_5")
    if iters_total >= iter_limit:
        flags.append("max_iterations_hit")
    if not converged:
        flags.append("did_not_converge")

    return {
        "accepted": len(rejections) == 0,
        "rejections": rejections,
        "flags": flags,
        "metrics": metrics,
    }


def append_rejection(case_id: str, reasons: list[str],
                     log_path: Path = REJECTION_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.writer(f)
            if f.seek(0, 2) == 0:
                w.writerow(["case_id", "reason", "timestamp"])
            w.writerow([
                case_id,
                ";".join(reasons) if reasons else "unknown",
                dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ])
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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
