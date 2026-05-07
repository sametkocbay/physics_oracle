# CFD Data Generator
## RANS k-ω SST Dataset for 2D Airfoil Flow

Implementation reference for generating the dataset. Covers sampling, meshing, OpenFOAM setup, field extraction, quality control, and ML export. Follow the sections in order.

---

## 1. Scope and Operating Envelope

- **Geometry family:** NACA 4-digit profiles
- **Dimensionality:** 2D
- **Flow regime:** Steady, incompressible, fully turbulent (no transition modeling)
- **Turbulence model:** k-ω SST
- **Solver:** simpleFoam (OpenFOAM v13 Foundation)
- **AoA range:** −5° to +15° (attached and mildly separated flow)
- **Reynolds number range:** 1×10⁵ to 5×10⁵ (log-spaced)

Any case outside this envelope is an "OOD probe" — stored separately, never used for training or hyperparameter tuning.

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
- Gmsh with a parameterized script (`generate_mesh.py`)
- Fully deterministic: same airfoil coordinates → same mesh, always
- Domain: 20c upstream, 25c downstream, ±20c vertical (far field ≥ 20c from surface)

### 4.2 Quality requirements
- **y+ < 1** at the wall — k-ω SST low-Re mode, resolves viscous sublayer
- **Growth ratio < 1.2** in boundary-layer normal direction
- **≥ 30 cells** in the boundary layer
- `checkMesh` non-orthogonality < 70, skewness < 4

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
gradSchemes     { default Gauss linear; }
divSchemes {
    div(phi,U)                   Gauss linearUpwind grad(U);
    div(phi,k)                   Gauss linearUpwind grad(k);
    div(phi,omega)               Gauss linearUpwind grad(omega);
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
```

### 5.4 `system/fvSolution`
```
p     GAMG / GaussSeidel     tol 1e-7  relTol 0.01
U     smoothSolver            tol 1e-8  relTol 0.1
k     smoothSolver            tol 1e-8  relTol 0.1
omega smoothSolver            tol 1e-8  relTol 0.1

SIMPLE relaxationFactors: p 0.3, U/k/omega 0.7
       nNonOrthogonalCorrectors 2
```

### 5.5 Convergence criteria
Stop when **all** residuals drop ≥ 4 orders of magnitude from their initial value, OR Cl/Cd change < 0.1% over 200 consecutive iterations — whichever comes first. Maximum 5000 iterations; flag anything that hits the limit without meeting the criterion.

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
mesh_version: v1
openfoam_version: v13
solver_settings_hash: "51bc03cd..."   # md5(fvSchemes + fvSolution)
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
| Residuals dropped | < 4 orders | reject |
| Negative k in field | any | reject |
| Negative omega in field | any | reject |
| y+ at wall | > 5 | reject |
| Iterations hit limit | ≥ 5000 | flag |
| Not converged | — | flag |

Target acceptance rate: ~80–90%. Higher rejection rates indicate a mesh or BC problem.

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

### 8.3 Usage

```bash
# Default: .npz output to ./ML_dataset/
python dataset/scripts/build_ml_dataset.py

# HDF5 output
python dataset/scripts/build_ml_dataset.py --fmt h5

# Custom paths
python dataset/scripts/build_ml_dataset.py \
    --cases-dir dataset/cases \
    --output-dir ML_dataset
```

Loading a sample in Python:
```python
import numpy as np
data = np.load("ML_dataset/NACA2412_p5.0_3.0e5.npz")
# data['x'], data['u'], data['sdf'], data['reynolds'], ...
```

---

## 9. Running the Full Pipeline

```bash
# Full run: 50 profiles, 200 cases, 10 OOD
python dataset/scripts/generate_dataset.py \
    --n-profiles 50 --n-cases 200 --n-ood 10 --seed 0

# Manifest + splits only (no meshing/solving)
python dataset/scripts/generate_dataset.py \
    --n-profiles 50 --n-cases 200 --skip-of

# Run specific cases
python dataset/scripts/generate_dataset.py \
    --n-profiles 50 --n-cases 200 \
    --cases NACA2412_p5.0_3.0e5 NACA0012_p0.0_2.0e5

# After cases are done, build the ML dataset
python dataset/scripts/build_ml_dataset.py
```

---

## 10. Full Directory Layout

```
cfd_data_generator/
├── ML_dataset/                             # flat ML-ready point clouds (one .npz per case)
│   ├── NACA2412_p5.0_3.0e5.npz
│   └── ...
├── dataset/
│   ├── manifest.yaml                       # dataset-level metadata, seeds, envelope
│   ├── rejection_log.csv                   # QC rejections: case_id, reason, timestamp
│   ├── splits/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   └── ood_probe.txt
│   ├── cases/
│   │   └── NACA2412_p5.0_3.0e5/
│   │       ├── meta.yaml
│   │       ├── fields.h5
│   │       ├── mesh.h5
│   │       ├── geometry.h5
│   │       ├── convergence.h5
│   │       └── of_case/
│   │           ├── 0/           (U, p, k, omega, nut)
│   │           ├── constant/    (polyMesh/, momentumTransport, physicalProperties)
│   │           └── system/      (controlDict, fvSchemes, fvSolution)
│   └── scripts/
│       ├── generate_dataset.py             # main orchestrator — runs the full pipeline
│       ├── generate_geometry.py            # NACA LHS profile sampling
│       ├── generate_mesh.py                # Gmsh C-grid mesh generation
│       ├── setup_openfoam_case.py          # writes 0/, constant/, system/ from templates
│       ├── run_openfoam.py                 # runs simpleFoam, parses residual log
│       ├── extract_fields.py               # OF output → fields.h5, mesh.h5, geometry.h5, convergence.h5
│       ├── quality_check.py                # QC checks, writes rejection_log.csv
│       ├── build_ml_dataset.py             # crops + exports ML_dataset/*.npz
│       └── common.py                       # shared constants, NACA geometry, case naming
└── prototype/                              # exploratory notebooks / scratch work
```

---

## 11. Reproducibility Checklist

- [ ] OpenFOAM v13 Foundation installed and sourced (`/opt/openfoam13/etc/bashrc`)
- [ ] `manifest.yaml` records `openfoam_version`, `mesh_version`, all LHS seeds
- [ ] Mesh generation is deterministic: same NACA code → byte-identical mesh
- [ ] `solver_settings_hash` (md5 of fvSchemes + fvSolution) is identical across all cases
- [ ] Split lists in `splits/` are committed — no random re-splitting at load time
- [ ] At least 5 cases re-run end-to-end and produce identical `fields.h5` (use `repro_hashes.json`)
- [ ] `rejection_log.csv` preserved and non-empty after any full run
