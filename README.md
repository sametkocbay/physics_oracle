# physics_oracle
## OpenFOAM-backed CFD oracle for 2D NACA airfoils (RANS k-ω SST)

`physics_oracle` is an installable Python package (`pip install` / `uv add`)
that wraps an OpenFOAM 2D airfoil pipeline. It does three things:

- **Dataset generation** — sample NACA profiles, mesh, solve, extract, and
  export an ML-ready dataset (`physics-oracle-generate`, `physics-oracle-build-ml`).
- **ML warm-start & residual evaluation** — take an ML-predicted flow field and
  either measure its per-cell physics residual (computed in OpenFOAM, or in a
  differentiable PyTorch reimplementation), probe how the solver reacts over
  N iterations, or run it to convergence (`physics-oracle-run-step` /
  `from physics_oracle import run_step`). See §10.
- **Reusable building blocks** — NACA geometry, meshing, OpenFOAM case setup,
  field extraction, and `mesh.h5 → polyMesh` conversion, all importable under
  the `physics_oracle.*` namespace.

It installs as a single top-level package and is meant to be consumed by other
projects (see §9 — *Using `physics_oracle` from another project*). The sections
below cover sampling, meshing, OpenFOAM setup, field extraction, quality
control, and ML export, in order.

---

## 1. Scope and Operating Envelope

- **Geometry family:** NACA 4-digit profiles
- **Dimensionality:** 2D
- **Flow regime:** Steady, incompressible, fully turbulent (no transition modeling)
- **Turbulence model:** k-ω SST
- **Solver:** `foamRun -solver incompressibleFluid` (OpenFOAM v13 Foundation; steady SIMPLE — the v13 successor of simpleFoam)
- **AoA range (trained envelope):** −5° to +5°
- **Reynolds number range (trained envelope):** 1×10⁵ to 5×10⁵ (log-spaced)
- **Validated extended band (OOD generation):** |AoA| up to ~12°, Re up to ~2×10⁶
  — enabled by the Reynolds-adaptive C-mesh (§4) and the two-stage startup
  continuation (§5.5). Genuinely unsteady corners (e.g. very thin sections at
  AoA ≳ 10°) have no steady RANS solution and are rejected at QC by design.

Any case outside the trained envelope is an "OOD probe" — stored separately, never used for training or hyperparameter tuning.

---

## 2. Sampling Strategy

All sampling uses Latin Hypercube (no uniform grids).

### 2.1 Geometry sampling
- NACA 4-digit parameters: max camber (0–6%), camber position (20–60% chord), thickness (8–18%)
- Joint (camber, position, thickness) space sampled with LHS
- Target: ~100–150 distinct profiles for training, ~30 for validation, ~30 for test

### 2.2 Flow condition sampling
- AoA: linear sampling in [−5°, +15°]
- Re: **log-uniform** in [1×10⁵, 5×10⁵] — never linear
- Joint (profile_id, AoA, log Re) space sampled with LHS
- Target: ~500 total cases

### 2.3 Dataset splits

| Split | Profiles | Purpose |
|-------|----------|---------|
| Train | 70% of profiles | Model training |
| Val | 15% of profiles (unseen) | Hyperparameter tuning, early stopping |
| Test | 15% of profiles (unseen) | Final reporting |
| OOD probe | Atypical conditions | Reporting only |

OOD probe conditions: AoA > 15°, Re < 1×10⁵, Re > 1×10⁷.  
Splits are stored as explicit case-ID lists in `dataset/splits/`.

---

## 3. Case Naming Convention

```
NACA[CODE]_[AoA]_[Re]
```

- `CODE`: 4-digit NACA code, e.g. `2412`, `0012`, `4415`
- `AoA`: signed, one decimal place, `p` for positive, `n` for negative
  - `+5.0°` → `p5.0`, `−2.5°` → `n2.5`, `0.0°` → `p0.0`
- `Re`: scientific notation, e.g. `1.5e6`, `3.0e5`

Examples: `NACA2412_p5.0_1.5e6`, `NACA0012_n2.5_3.0e5`, `NACA4415_p10.0_5.0e5`

---

## 4. Mesh Generation

C-grid topology, identical structure for every case; only the airfoil coordinates and first-layer height vary. This keeps dataset size manageable and avoids mesh variability as a confounding factor.

### 4.1 Tooling

Two meshers share the same `(of_case_dir, case_id) -> quality_dict` interface:

- **Structured C-mesh** (`meshing/c_mesh.py`, selected with `--c-mesh`) — the
  mesher used for the dataset. Builds the structured node array in Python
  (closed-TE NACA + freestream-aligned wake cut + half-circle far field + TFI),
  serialises it to Gmsh `.msh` and imports via `gmshToFoam`.
- Gmsh unstructured (`meshing/gmsh_mesh.py`) — legacy/default path with a
  boundary-layer field and triangular-prism outer mesh.

Both are fully deterministic: same case id → same mesh, always.
Domain: 20c upstream, 25c downstream, ±20c vertical (far field ≥ 20c from surface).

**Reynolds adaptivity (C-mesh, mesh v2):**
- First-layer height is sized from the flat-plate correlation *at the actual
  case Re*, holding wall y+ ≈ 0.8 across the whole Reynolds band (the wall-normal
  cell count grows above ~3×10⁶ to keep the geometric growth ratio ≤ 1.2).
- The wake-cut first spacing grows with distance from the trailing edge
  (`WAKE_CUT_AR_CAP = 100`), so the cut's first-layer cells keep a bounded
  in-plane aspect ratio. Mesh v1 applied the wall y1 along the entire cut,
  producing ~30,000:1 slivers with ~90° non-orthogonality in the far wake —
  these destabilised the pressure solve at high Re and floored its residual.

### 4.2 Quality requirements
- **y+ < 1** at the wall — k-ω SST low-Re mode, resolves viscous sublayer
- **Growth ratio ≤ 1.2** in boundary-layer normal direction
- **≥ 30 cells** in the boundary layer
- `checkMesh` non-orthogonality < 70 (mesh v2 measures 66–68°), skewness < 4

---

## 5. OpenFOAM Case Setup

Each case lives at `dataset/cases/<case_id>/of_case/`. All files are written by `setup_openfoam_case.py`.

### 5.1 Directory structure

```
of_case/
├── 0/
│   ├── U
│   ├── p
│   ├── k
│   ├── omega
│   └── nut
├── constant/
│   ├── polyMesh/                  # written by Gmsh + gmshToFoam
│   ├── momentumTransport          # turbulence model selection (OF v13)
│   └── physicalProperties         # kinematic viscosity (OF v13)
└── system/
    ├── controlDict
    ├── fvSchemes
    └── fvSolution
```

### 5.2 Boundary conditions (`0/`)

**`0/U`** — inlet velocity is rotated by AoA (airfoil stays chord-aligned, flow direction varies)
```
internalField   uniform (U_x U_y 0);
inlet     fixedValue (U_x U_y 0)
outlet    zeroGradient
airfoil   noSlip
top/bottom  slip
```

**Inlet turbulence** (I = 0.01, L = 0.07·chord, C_μ = 0.09):
```
k_inlet     = 1.5 · (|U| · I)²
omega_inlet = sqrt(k) / (C_μ^0.25 · L)
```

### 5.3 `system/fvSchemes`
```
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear;  grad(U) cellLimited Gauss linear 1; }
divSchemes {
    div(phi,U)                   bounded Gauss linearUpwindV grad(U);
    div(phi,k)                   bounded Gauss upwind;
    div(phi,omega)               bounded Gauss upwind;
    div((nuEff*dev(T(grad(U))))) Gauss linear;
    div(div(phi,U))              Gauss linear;   // potentialFoam -writep (diagnostics only)
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
snGradSchemes    { default limited corrected 0.5; }
```

The non-orthogonal correction is **limited** (coefficient 0.5): identical to
plain `corrected` on faces below ~63° non-orthogonality, but it bounds the
explicit correction on the worst faces so it cannot diverge. The
differentiable PyTorch residual (§10) implements the same limiter.

### 5.4 `system/fvSolution`
```
p     GAMG / DICGaussSeidel   tol 1e-7  relTol 0.01
U     smoothSolver            tol 1e-8  relTol 0.1
k     smoothSolver            tol 1e-8  relTol 0.1
omega smoothSolver            tol 1e-8  relTol 0.1

SIMPLE (SIMPLEC, consistent yes)
       relaxationFactors: p 0.3, U/k/omega 0.5
       nNonOrthogonalCorrectors 2
       residualControl: p/U/k/omega 1e-6
```

### 5.5 Startup and initialisation

Cases **cold-start from the uniform inlet field**. There is deliberately no
potentialFoam pre-step: the zero-circulation potential field (no Kutta
condition) forces the solver to shed a starting vortex along the wake cut and
leaves the pressure residual limit-cycling around 1e-2 instead of converging
(`runner.run_potential_init` remains available for diagnostics only).

Cases outside the trained envelope (|AoA| > 5° or Re > 5×10⁵) run a
**two-stage startup continuation** (`startup:` section in
`configs/openfoam.yaml`, `runner._run_two_stage`):

- **Stage A** (400 iterations): first-order `div(phi,U) upwind`, heavy
  under-relaxation (p 0.2, U/k/omega 0.3), 3 non-orthogonal correctors, and
  `fvConstraints` field limiting — `limitMag` on U (4×U∞) and a `bound` floor
  on omega — to survive the cold-start transient at high Re/AoA.
- **Stage B**: restarts from the stage-A field with the production schemes
  above and slightly damped relaxation (p 0.2, U/k/omega 0.4) to full
  convergence. Both stages append to one `simpleFoam.log`; stage metadata is
  recorded in `run_info.yaml` and surfaced as `solver_run_info` in `meta.yaml`.

In-band cases never enter this path, so trained-envelope behaviour is unchanged.

### 5.6 Convergence criteria

A case converges when **all** residuals (Ux, Uy, p, k, omega) drop ≥ 4 orders
of magnitude, measured robustly: reference = max over the first 10 iterations,
final = median over the last 50 (`qc.orders_drop_ref_window` /
`orders_drop_tail_window` in `configs/postprocess.yaml`). The solver itself
exits early at absolute residuals ≤ 1e-6 (`residualControl`). Maximum 5000
iterations; anything hitting the limit without meeting the criterion is
flagged, and acceptance falls to QC (§7).

---

## 6. Data Stored Per Case

### 6.1 `fields.h5` — converged solution
| Dataset | Shape | Description |
|---------|-------|-------------|
| `U` | (Ncells, 2) | Velocity (x, y components) |
| `p` | (Ncells,) | Kinematic pressure |
| `k` | (Ncells,) | Turbulent kinetic energy |
| `omega` | (Ncells,) | Specific dissipation rate |
| `nut` | (Ncells,) | Turbulent viscosity |
| `wall_distance` | (Ncells,) | Distance to nearest airfoil surface |

### 6.2 `mesh.h5` — geometry and connectivity
| Dataset | Shape | Description |
|---------|-------|-------------|
| `cell_centers` | (Ncells, 2) | 2D cell-center coordinates |
| `points` | (Npoints, 2) | Mesh vertex coordinates |
| `connectivity` | (Ncells, Nverts) | Cell-to-vertex mapping |
| `boundary_markers` | (Ncells,) | 0=interior, 1=wall, 2=inlet, 3=outlet, 4=farfield |

### 6.3 `geometry.h5` — airfoil surface
- `airfoil_coordinates` — (N, 2), ordered trailing-edge → upper → LE → lower → trailing-edge (cosine spacing)
- `airfoil_mesh_points` — wall point coordinates extracted from polyMesh

### 6.4 `convergence.h5` — solver diagnostics
| Item | Description |
|------|-------------|
| `residual_history` | (Niter, 4) — [U, p, k, omega] |
| `cl_history` / `cd_history` | Force coefficient histories |
| `y_plus` | y+ at wall-adjacent cells |
| attrs: `converged`, `iterations_total`, `iterations_to_convergence` | Scalar metadata |
| attrs: `orders_drop_*`, `final_residual_*` | Per-field diagnostics |

### 6.5 `meta.yaml`
```yaml
case_id: NACA2412_p5.0_3.0e5
naca_code: "2412"
aoa_deg: 5.0
Re: 300000.0
U_inlet: [2.985, 0.261]   # (U_x, U_y) — rotated by AoA
U_mag: 3.0
nu: 1.0e-05
chord: 1.0
k_inlet: 3.375e-04
omega_inlet: 0.1234
mesh_version: v2
openfoam_version: v13
solver_settings_hash: "51bc03cd..."   # md5(configs/openfoam.yaml)
generation_timestamp: "2026-05-07T15:23:22Z"
converged: true
iterations_to_convergence: 1842
flags: []
split: train
```

---

## 7. Quality Control

`quality_check.py` runs after every case. Rejections are appended to `dataset/rejection_log.csv`.

| Check | Threshold | Action |
|-------|-----------|--------|
| Residuals dropped | ≥ 4.0 orders (all fields) | accept (residual gate) |
| Residuals dropped | 3.5 – 4.0 orders (gray zone) | accept **only if forces stationary**, flag `gray_zone_force_accepted` |
| Residuals dropped | < 3.5 orders | reject |
| Negative k in field | any | reject |
| Negative omega in field | any | reject |
| y+ at wall | > 5 | reject |
| Iterations hit limit | ≥ 5000 | flag |
| Not converged | — | flag |

**Force stationarity** (gray zone only): over the trailing 500 iterations,
both Cl and Cd must have relative scatter (std/|mean|) **and** half-window
mean drift ≤ 2% (`force_*` keys in `configs/postprocess.yaml`). Empirically,
every case under ~3.7 residual orders is a genuinely unsteady flow (5–46% Cl
scatter) and stays rejected, while force-steady cases whose pressure-residual
norm merely converges slowly at high Re (e.g. thick sections at Re 2×10⁶) are
rescued instead of falsely rejected.

Target acceptance rate: ~80–90% in-band. High-|AoA| OOD batches reject more —
mostly thin sections beyond their steady-RANS limit — and the fill modes
oversample to compensate. `scripts/diagnose_robustness.py` runs a fixed
6-case diagnostic matrix (high AoA / high Re + an in-band control) through
the production solver path and classifies any failures.

---

## 8. ML Dataset Export

`build_ml_dataset.py` post-processes all converged cases into a flat folder of `.npz` (or `.h5`) files suitable for direct use in PyTorch / JAX dataloaders.

### 8.1 Bounding box

The full CFD domain spans ±20 chord lengths. The ML dataset is cropped to a smaller region centred on the airfoil (chord = 1, leading edge at x = 0, trailing edge at x = 1):

```
x ∈ [−1.5,  3.5]   (1.5c in front of LE, 2.5c behind TE)
y ∈ [−1.5,  1.5]   (1.5c above and below)
```

This retains the near-wake and boundary-layer region while discarding the far-field padding cells.

### 8.2 Output fields (per `.npz` file)

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `x`, `y` | (N,) | float32 | Cell-center coordinates |
| `sdf` | (N,) | float32 | Signed distance to nearest airfoil surface (≥ 0 for exterior cells) |
| `u_init`, `v_init` | (N,) | float32 | Uniform inlet velocity (initial condition) |
| `u`, `v` | (N,) | float32 | Converged velocity components |
| `p` | (N,) | float32 | Kinematic pressure |
| `omega` | (N,) | float32 | Specific dissipation rate |
| `k` | (N,) | float32 | Turbulent kinetic energy |
| `nut` | (N,) | float32 | Turbulent viscosity |
| `reynolds` | scalar | float32 | Reynolds number |
| `is_wall` | (N,) | uint8 | 1 if cell is adjacent to airfoil wall, else 0 |

N ≈ 220 000 points per case (after bounding-box crop from ~281 000 total cells).

In addition to the volume point cloud, each file carries a separate
airfoil-surface table with one row per wall face (M rows, M ≈ 240, aligned by
wall-face index). These sit on the airfoil surface itself, unlike `is_wall`
which marks the first layer of volume cells just off the wall. Quantities are
kinematic (divided by density, consistent with `p`), matching OpenFOAM's
`wallShearStress` function object.

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `wall_xy` | (M, 2) | float32 | Wall-face-center coordinates (same frame as `x`, `y`) |
| `wall_normal` | (M, 2) | float32 | Unit surface normal, pointing from the wall into the fluid |
| `wall_shear` | (M, 2) | float32 | Kinematic wall shear stress vector τ_w / ρ (m²/s²) |
| `wall_p` | (M,) | float32 | Kinematic surface pressure (m²/s²) |
| `wall_length` | (M,) | float32 | Face edge length (for surface integration) |
| `wall_cell` | (M,) | int64 | Owner cell index — links each face back to the volume cloud |

Skin friction and pressure coefficients follow directly, `cf = |wall_shear| /
(0.5·U∞²)` and `cp = wall_p / (0.5·U∞²)`. The wall arrays are present for all
splits (`train`, `val`, `test`, `ood`).

### 8.3 Usage

```bash
# Default: .npz output to dataset/ML_dataset/
uv run physics-oracle-build-ml

# HDF5 output
uv run physics-oracle-build-ml --fmt h5

# Custom paths
uv run physics-oracle-build-ml \
    --cases-dir dataset/cases \
    --output-dir dataset/ML_dataset
```

The crop box, exported fields, output format, and dtype are configured in
`configs/postprocess.yaml`.

Loading a sample in Python:
```python
import numpy as np
data = np.load("dataset/ML_dataset/train/NACA2412_p5.0_3.0e5.npz")
# data['x'], data['u'], data['sdf'], data['reynolds'], ...
```

---

## 9. Running the Full Pipeline

Install dependencies once with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This installs `physics_oracle` as an importable package and puts three console
scripts on PATH.

```bash
# Full run: 50 profiles, 200 cases, 10 OOD
uv run physics-oracle-generate \
    --n-profiles 50 --n-cases 200 --n-ood 10 --seed 0

# Manifest + splits only (no meshing/solving)
uv run physics-oracle-generate \
    --n-profiles 50 --n-cases 200 --skip-of

# Run specific cases
uv run physics-oracle-generate \
    --n-profiles 50 --n-cases 200 \
    --cases NACA2412_p5.0_3.0e5 NACA0012_p0.0_2.0e5

# After cases are done, build the ML dataset
uv run physics-oracle-build-ml

# Warm-start an OpenFOAM run from an ML prediction (.pt) on a saved mesh.h5
uv run physics-oracle-run-step \
    --prediction predictions.pt --mesh NACA2412_p5.0_3.0e5_mesh.h5 \
    --work-dir /tmp/run01 --only-residual
```

Runtime outputs default to `<cwd>/dataset/`; override with the
`PHYSICS_ORACLE_DATASET_ROOT` environment variable.

### Using `physics_oracle` from another project

`physics_oracle` is a normal installable package. From another repo's
`pyproject.toml` (uv example):

```toml
[project]
dependencies = ["physics-oracle"]

[tool.uv.sources]
physics-oracle = { path = "/path/to/physics_oracle", editable = true }
# or, git-pinned for CI:
# physics-oracle = { git = "https://github.com/sametkocbay/physics_oracle.git", rev = "<sha>" }
```

```python
from physics_oracle import run_step, write_polymesh_from_h5, compute_inlet_conditions
result = run_step(prediction, mesh_h5, work_dir, mode="only-residual")
```

---

## 10. ML Warm-Start: Residual Evaluation & Solver Probing

`physics-oracle-run-step` (`physics_oracle.run_step`) takes an ML-predicted flow
field (a `predictions.pt` with `pos`, `target`, `meta`) on a saved `mesh.h5` and
evaluates it. Four mutually-exclusive modes:

| Mode | What it does | Output |
|------|--------------|--------|
| `--only-residual` | **0-step physics check.** Per-cell residual of the momentum, continuity, k and omega equations *at the prediction* — evaluated by an OpenFOAM `coded` function object run through `foamPostProcess` (no SIMPLE iteration). | per-cell fields → `residuals.npz` + volScalarFields in `of_case/0/` |
| `--only-residual-diff` | The **same residual, recomputed in PyTorch** — no OpenFOAM solve, differentiable end-to-end. | per-cell fields → `residuals_diff.npz` |
| `--n-steps N` | **Solver-consistency probe.** Run N SIMPLE iterations (1–5 recommended) warm-started from the prediction; report OpenFOAM's per-iteration residuals — does convergence start, or does it diverge? | per-iteration array → `iteration_residuals.npz` |
| `--full-run` | Standard warm-started refinement to convergence (endTime 500). | convergence summary |

`--summary` (with either `--only-residual` mode) reduces each per-cell field to
scalar stats `{median, p99, mean, max, L2}` instead of the full field.

The four residual fields are `momentumResidual`, `continuityResidual`,
`kResidual`, `omegaResidual`. Cells adjacent to the airfoil carry wall functions
that override the transport equations, so they are masked to zero. The omega
residual is large by nature in the near-wall band (steep omega profile) — read
its **median**, not mean/max.

### Differentiable residual (`physics_oracle.residual_diff`)

`--only-residual` runs inside a separate OpenFOAM process — accurate, but a black
box to autograd. `residual_diff` reimplements the identical finite-volume
residual in **PyTorch**, so gradients flow from a prediction tensor back to the
residual (usable as a physics-informed training loss):

- Mesh geometry (face-area vectors, cell volumes, interpolation weights,
  non-orthogonal correction vectors, wall distance) is exported **once** from
  OpenFOAM and cached next to `mesh.h5` as `<mesh>.geom.pt`; later calls need no
  OpenFOAM at all.
- The FVM operators reproduce the case's `fvSchemes` — linear / upwind /
  `linearUpwindV` interpolation, `Gauss` and `cellLimited` gradients, `bounded`
  divergence, `limited corrected 0.5` laplacian (pass `limit_coeff=None` for
  the pre-mesh-v2 unlimited form) — plus the explicit kOmegaSST source terms.
- It matches the OpenFOAM coded-FO residual to **correlation 1.0** (field-L2
  ≤ 0.2 %) on realistic predictions.

```python
from physics_oracle.residual_diff import export_mesh_geometry, DifferentiableResidual

geom  = export_mesh_geometry(mesh_h5, work_dir, spec)   # one-time, then cached
model = DifferentiableResidual(geom, nu, inlet)
res   = model(U, p, k, omega, nut)                      # autograd-friendly tensors
loss  = res["momentumResidual"].pow(2).mean() + res["continuityResidual"].pow(2).mean()
loss.backward()                                         # gradients reach U, p, k, …
```

### CLI examples

```bash
# 0-step per-cell physics residual at the prediction (OpenFOAM coded FO)
uv run physics-oracle-run-step --prediction predictions.pt \
    --mesh NACA2412_p5.0_3.0e5_mesh.h5 --work-dir /tmp/run01 --only-residual

# Same residual, differentiable (PyTorch, no OpenFOAM solve)
uv run physics-oracle-run-step --prediction predictions.pt \
    --mesh NACA2412_p5.0_3.0e5_mesh.h5 --work-dir /tmp/run01 --only-residual-diff

# Solver-consistency probe — 3 SIMPLE iterations
uv run physics-oracle-run-step --prediction predictions.pt \
    --mesh NACA2412_p5.0_3.0e5_mesh.h5 --work-dir /tmp/run01 --n-steps 3
```

Rough per-input cost (≈57k-cell mesh): `--only-residual` ~30 s (runs OpenFOAM
every call); `--only-residual-diff` ~3 s once geometry is cached (the residual
itself is sub-second); `--n-steps 3` ~10 s.

---

## 11. Full Directory Layout

```
physics_oracle/
├── pyproject.toml                          # uv / hatchling — single physics_oracle package
├── uv.lock
├── scripts/
│   ├── diagnose_robustness.py              # 6-case solver-robustness matrix + failure classifier
│   └── prune_rejected.py                   # delete rejected case dirs listed in rejection_log.csv
└── src/
    └── physics_oracle/
        ├── __init__.py                     # public API re-exports
        ├── configs/
        │   ├── openfoam.yaml               # solver + BC + turbulence (source of truth)
        │   └── postprocess.yaml            # ML crop box, fields, QC thresholds, viz panels
        ├── core/                           # CaseSpec, paths, envelope, logging, repro
        ├── geometry/                       # NACA math + LHS sampling
        ├── meshing/                        # Gmsh unstructured + structured C-mesh
        ├── openfoam_setup/                 # case_setup, of_writer, runner, extract, qc,
        │                                   #   mesh_h5_to_polymesh, _residual_functions,
        │                                   #   _geometry_export
        ├── residual_diff/                  # differentiable PyTorch residual:
        │                                   #   geometry, operators, boundary, residual
        ├── utils/                          # visualize_npz
        └── cli/                            # generate_dataset, build_ml_dataset,
                                            #   run_ml_initialized_step (4 residual modes)
```

Runtime outputs (gitignored) land under `<cwd>/dataset/` — or
`$PHYSICS_ORACLE_DATASET_ROOT` — as:

```
dataset/
├── manifest.yaml                       # dataset-level metadata, seeds, envelope
├── rejection_log.csv                   # QC rejections: case_id, reason, timestamp
├── splits/                             # train.txt, val.txt, test.txt, ood_probe.txt
├── cases/<case_id>/                    # meta.yaml, fields.h5, mesh.h5, geometry.h5,
│                                       #   convergence.h5, of_case/
└── ML_dataset/                         # per-split subfolders with .npz + metadata.csv
    ├── train/  ├── val/  ├── test/  └── ood/
```

---

## 12. Reproducibility Checklist

- [ ] OpenFOAM v13 Foundation installed and sourced (`/opt/openfoam13/etc/bashrc`)
- [ ] `uv sync` ran cleanly; `.venv/` matches `uv.lock`
- [ ] `manifest.yaml` records `openfoam_version`, `mesh_version`, all LHS seeds
- [ ] Mesh generation is deterministic: same NACA code → byte-identical mesh
- [ ] `solver_settings_hash` (md5 of `configs/openfoam.yaml`) is identical across all cases
- [ ] Split lists in `splits/` are committed — no random re-splitting at load time
- [ ] At least 5 cases re-run end-to-end and produce identical `fields.h5` (use `repro_hashes.json`)
- [ ] `rejection_log.csv` preserved and non-empty after any full run
