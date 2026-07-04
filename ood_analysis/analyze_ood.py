"""Analyze whether the pulled OOD split is genuinely out-of-distribution
relative to train/val/test, across geometry, flow params, and Cl/Cd."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

ROOT = Path("/home/samet_kocbay/projects/NACA_4_Digit_for_ML")
OUT = Path(__file__).resolve().parent
SPLITS = ["train", "val", "test", "ood"]
COLORS = {"train": "#1f77b4", "val": "#2ca02c", "test": "#ff7f0e", "ood": "#d62728"}
ZORDER = {"train": 1, "val": 2, "test": 3, "ood": 5}

def load(split):
    df = pd.read_csv(ROOT / split / "metadata.csv")
    df["split"] = split
    code = df["naca_code"].astype(str).str.zfill(4)
    df["camber"] = code.str[0].astype(int)
    df["camber_pos"] = code.str[1].astype(int) * 10
    df["thickness"] = code.str[2:].astype(int)
    df["abs_aoa"] = df["aoa_deg"].abs()
    df["abs_cl"] = df["cl"].abs()
    return df

splits = {s: load(s) for s in SPLITS}
alldf = pd.concat(splits.values(), ignore_index=True)
alldf.to_csv(OUT / "combined_with_ood.csv", index=False)

def overlay(ax, xc, yc, logx=False, logy=False):
    for s in SPLITS:
        d = splits[s]
        ax.scatter(d[xc], d[yc], s=(30 if s == "ood" else 14),
                   alpha=(0.85 if s == "ood" else 0.4), label=f"{s} (n={len(d)})",
                   color=COLORS[s], zorder=ZORDER[s],
                   edgecolors="k" if s == "ood" else "none", linewidths=0.3)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel(xc); ax.set_ylabel(yc); ax.grid(alpha=0.3); ax.legend(fontsize=8)

# ---- Figure 1: Cl-Cd and the lift/condition story ----
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
overlay(axs[0, 0], "cd", "cl");                axs[0, 0].set_title("Cl vs Cd")
overlay(axs[0, 1], "cd", "cl", logx=True);     axs[0, 1].set_title("Cl vs Cd (log Cd)")
overlay(axs[1, 0], "aoa_deg", "cl");           axs[1, 0].set_title("Cl vs AoA")
overlay(axs[1, 1], "abs_aoa", "abs_cl");       axs[1, 1].set_title("|Cl| vs |AoA|  (lift/downforce magnitude)")
fig.tight_layout(); fig.savefig(OUT / "ood_cl_cd.png", dpi=120); plt.close(fig)

# ---- Figure 2: geometry + flow parameter space ----
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
overlay(axs[0, 0], "thickness", "camber");     axs[0, 0].set_title("camber vs thickness (geometry)")
overlay(axs[0, 1], "aoa_deg", "reynolds");     axs[0, 1].set_title("Reynolds vs AoA")
overlay(axs[1, 0], "thickness", "cl");         axs[1, 0].set_title("Cl vs thickness")
overlay(axs[1, 1], "camber", "cl");            axs[1, 1].set_title("Cl vs camber")
fig.tight_layout(); fig.savefig(OUT / "ood_geometry.png", dpi=120); plt.close(fig)

# ---- Figure 3: marginal histograms ----
fig, axs = plt.subplots(2, 3, figsize=(17, 9))
for ax, c in zip(axs.ravel(), ["thickness", "camber", "abs_aoa", "reynolds", "abs_cl", "cd"]):
    for s in SPLITS:
        ax.hist(splits[s][c], bins=18, alpha=0.5, label=s, color=COLORS[s], density=True)
    ax.set_title(c); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ood_marginals.png", dpi=120); plt.close(fig)

# ---- Figure 4: stacked count of signed Cl by split (train on top of the stack) ----
# hist stacks bottom->top in the order given, so pass ood..train to put train on top.
stack_order = ["ood", "test", "val", "train"]
fig, ax = plt.subplots(figsize=(11, 6))
data = [splits[s]["cl"].values for s in stack_order]
cols = [COLORS[s] for s in stack_order]
labs = [f"{s} (n={len(splits[s])})" for s in stack_order]
ax.hist(data, bins=40, stacked=True, color=cols, label=labs,
        edgecolor="white", linewidth=0.2)
# legend reads top-of-stack first: train, val, test, ood
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], fontsize=9)
ax.set_xlabel("Cl (signed)"); ax.set_ylabel("count")
ax.set_title("Cl distribution by split (stacked count, train on top)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "ood_cl_stacked.png", dpi=120); plt.close(fig)

# ---- Quantitative OOD report ----
tr = splits["train"]
print("\n===== RANGES (min .. max) per split =====")
for c in ["thickness", "camber", "camber_pos", "aoa_deg", "abs_aoa", "reynolds", "cl", "abs_cl", "cd"]:
    line = f"{c:11s}"
    for s in SPLITS:
        v = splits[s][c]
        line += f"  {s}:[{v.min():.4g},{v.max():.4g}]"
    print(line)

print("\n===== fraction of each split OUTSIDE the TRAIN per-feature range =====")
for c in ["thickness", "camber", "abs_aoa", "reynolds", "abs_cl", "cd"]:
    lo, hi = tr[c].min(), tr[c].max()
    row = f"{c:11s} train[{lo:.4g},{hi:.4g}]"
    for s in ["val", "test", "ood"]:
        v = splits[s][c]
        row += f"  {s}:{100*((v<lo)|(v>hi)).mean():5.1f}%"
    print(row)

print("\n===== KS D-statistic vs train (1.0 = fully separated) =====")
for c in ["thickness", "camber", "abs_aoa", "reynolds", "abs_cl", "cd"]:
    row = f"{c:11s}"
    for s in ["val", "test", "ood"]:
        D, p = stats.ks_2samp(tr[c], splits[s][c])
        row += f"  {s}: D={D:.2f}(p={p:.1g})"
    print(row)

# geometry novelty: are OOD profiles unseen shapes?
seen = set(pd.concat([splits[s] for s in ["train","val","test"]])["naca_code"].astype(str).str.zfill(4))
ood_codes = set(splits["ood"]["naca_code"].astype(str).str.zfill(4))
print(f"\nOOD profiles: {len(ood_codes)} unique, {len(ood_codes & seen)} overlap with train/val/test "
      f"-> {len(ood_codes - seen)} are NEW geometries")
print("Saved:", *(p.name for p in OUT.glob("ood_*.png")))
