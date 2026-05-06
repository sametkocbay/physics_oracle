#!/usr/bin/env bash

set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$case_dir"

resume=false

usage() {
    echo "Usage: $0 [--resume]"
    echo "  --resume  Continue from the latest written iteration without rebuilding the mesh"
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 1
fi

case "${1:-}" in
    "")
        ;;
    --resume)
        resume=true
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ -z "${WM_PROJECT_DIR:-}" ]]; then
    if [[ -f /opt/openfoam13/etc/bashrc ]]; then
        # shellcheck disable=SC1091
        source /opt/openfoam13/etc/bashrc
    else
        echo "OpenFOAM 13 environment not found at /opt/openfoam13/etc/bashrc" >&2
        exit 1
    fi
fi

required_commands=(foamRun)
if [[ "$resume" == false ]]; then
    required_commands=(python3 gmsh gmshToFoam checkMesh foamRun)
fi

for cmd in "${required_commands[@]}"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Required command not found: $cmd" >&2
        exit 1
    fi
done

mesh_geo="system/naca_airfoil.geo"
mesh_msh="$case_dir/naca_airfoil.msh"
boundary_file="constant/polyMesh/boundary"
control_dict="system/controlDict"
geo_generator="$case_dir/../utils/generate_naca_geo.py"

if [[ "$resume" == true ]]; then
    if [[ ! -d constant/polyMesh ]]; then
        echo "Cannot resume: constant/polyMesh is missing. Run a fresh case first." >&2
        exit 1
    fi

    time_dirs=()
    shopt -s nullglob
    for time_dir in [0-9]*; do
        [[ -d "$time_dir" ]] || continue
        time_dirs+=("$time_dir")
    done
    shopt -u nullglob

    if [[ ${#time_dirs[@]} -eq 0 ]]; then
        echo "Cannot resume: no written time directories were found." >&2
        exit 1
    fi

    latest_time="$(printf '%s\n' "${time_dirs[@]}" | sort -V | tail -n 1)"
    sed -i -E 's/^[[:space:]]*startFrom[[:space:]]+.*/startFrom       latestTime;/' "$control_dict"

    echo "[resume] Continuing from latest written iteration: $latest_time"
    foamRun -solver incompressibleFluid | tee -a simpleFoam.log
    exit 0
fi

shopt -s nullglob
rm -f "$mesh_msh" gmsh.log gmshToFoam.log checkMesh.log simpleFoam.log
rm -rf constant/polyMesh postProcessing processor*
for time_dir in [1-9]* [0-9].*; do
    [[ -d "$time_dir" ]] && [[ "$time_dir" != *.orig ]] && rm -rf "$time_dir"
done
shopt -u nullglob

echo "[1/5] Generating airfoil .geo via $(basename "$geo_generator")"
python3 "$geo_generator" --geo-out "$case_dir/$mesh_geo" \
    --polygon-out "$case_dir/constant/airfoil_polygon.csv" | tee gmsh.log

echo "[2/5] Generating Gmsh mesh from $mesh_geo"
gmsh -3 "$mesh_geo" -format msh2 -o "$mesh_msh" | tee -a gmsh.log

echo "[3/5] Importing mesh into OpenFOAM"
gmshToFoam "$mesh_msh" | tee gmshToFoam.log
rm -f "$mesh_msh"

echo "[4/5] Applying OpenFOAM patch types"
sed -i \
    -e '/frontAndBack/,/}/{s/type            patch;/type            empty;/; s/physicalType    patch;/physicalType    empty;/}' \
    -e '/bottom/,/}/{s/type            patch;/type            symmetryPlane;/; s/physicalType    patch;/physicalType    symmetryPlane;/}' \
    -e '/top/,/}/{s/type            patch;/type            symmetryPlane;/; s/physicalType    patch;/physicalType    symmetryPlane;/}' \
    -e '/airfoilWalls/,/}/{s/type            patch;/type            wall;/; s/physicalType    patch;/physicalType    wall;/}' \
    "$boundary_file"

checkMesh | tee checkMesh.log

echo "[5/5] Starting steady-state incompressible solver"
foamRun -solver incompressibleFluid | tee simpleFoam.log
