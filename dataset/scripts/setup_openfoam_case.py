"""§5 — Build the OpenFOAM case directory tree for one case.

Writes 0/{U,p,k,omega,nut,phi}, constant/{momentumTransport,physicalProperties},
and system/{controlDict,fvSchemes,fvSolution} with values substituted from the
case spec.

The 2D simulation uses a 1-cell-thick extruded mesh with frontAndBack=empty
and top/bottom=symmetryPlane (consistent with the project's case_template/
and equivalent to §5.3's "slip" wall in 2D).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    CHORD,
    NU,
    CaseSpec,
    setup_logging,
)

LOG = setup_logging()

# ---------------------------------------------------------------------------
# Boundary-condition templates  (formatted with case-specific numerical values)
# ---------------------------------------------------------------------------

FOAM_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM dataset case (auto-generated)
   \\\\    /   O peration     |
    \\\\  /    A nd           |
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


U_TEMPLATE = (FOAM_HEADER + """
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({U_x:.10g} {U_y:.10g} 0);

boundaryField
{{
    inlet        {{ type fixedValue; value uniform ({U_x:.10g} {U_y:.10g} 0); }}
    outlet       {{ type zeroGradient; }}
    top          {{ type freestream; freestreamValue uniform ({U_x:.10g} {U_y:.10g} 0); }}
    bottom       {{ type freestream; freestreamValue uniform ({U_x:.10g} {U_y:.10g} 0); }}
    airfoilWalls {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
""")


P_TEMPLATE = (FOAM_HEADER + """
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet        {{ type zeroGradient; }}
    outlet       {{ type fixedValue; value uniform 0; }}
    top          {{ type freestream; freestreamValue uniform 0; }}
    bottom       {{ type freestream; freestreamValue uniform 0; }}
    airfoilWalls {{ type zeroGradient; }}
    frontAndBack {{ type empty; }}
}}
""")


K_TEMPLATE = (FOAM_HEADER + """
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k:.10g};

boundaryField
{{
    inlet        {{ type fixedValue; value uniform {k:.10g}; }}
    outlet       {{ type zeroGradient; }}
    top          {{ type inletOutlet; inletValue uniform {k:.10g}; value uniform {k:.10g}; }}
    bottom       {{ type inletOutlet; inletValue uniform {k:.10g}; value uniform {k:.10g}; }}
    airfoilWalls {{ type fixedValue; value uniform 0; }}
    frontAndBack {{ type empty; }}
}}
""")


OMEGA_TEMPLATE = (FOAM_HEADER + """
dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omega:.10g};

boundaryField
{{
    inlet        {{ type fixedValue; value uniform {omega:.10g}; }}
    outlet       {{ type zeroGradient; }}
    top          {{ type inletOutlet; inletValue uniform {omega:.10g}; value uniform {omega:.10g}; }}
    bottom       {{ type inletOutlet; inletValue uniform {omega:.10g}; value uniform {omega:.10g}; }}
    airfoilWalls {{ type omegaWallFunction; value uniform {omega:.10g}; }}
    frontAndBack {{ type empty; }}
}}
""")


NUT_TEMPLATE = (FOAM_HEADER + """
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet        {{ type calculated; value uniform 0; }}
    outlet       {{ type calculated; value uniform 0; }}
    top          {{ type calculated; value uniform 0; }}
    bottom       {{ type calculated; value uniform 0; }}
    airfoilWalls {{ type nutLowReWallFunction; value uniform 0; }}
    frontAndBack {{ type empty; }}
}}
""")


PHI_TEMPLATE = (FOAM_HEADER + """
dimensions      [0 3 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet        {{ type calculated; value uniform 0; }}
    outlet       {{ type calculated; value uniform 0; }}
    top          {{ type calculated; value uniform 0; }}
    bottom       {{ type calculated; value uniform 0; }}
    airfoilWalls {{ type calculated; value uniform 0; }}
    frontAndBack {{ type empty; }}
}}
""")


# ---------------------------------------------------------------------------
# constant/
# ---------------------------------------------------------------------------

MOMENTUM_TRANSPORT = (FOAM_HEADER + """
simulationType  RAS;

RAS
{{
    model           kOmegaSST;
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}}
""")


PHYSICAL_PROPERTIES = (FOAM_HEADER + """
viscosityModel  constant;

nu              {nu:.6e};
""")


# ---------------------------------------------------------------------------
# system/
# ---------------------------------------------------------------------------

CONTROL_DICT = (FOAM_HEADER + """
application     foamRun;
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;

writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      1;

writeFormat     ascii;
writePrecision  12;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

functions
{{
    forceCoeffs1
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (airfoilWalls);
        rho             rhoInf;
        rhoInf          1;
        liftDir         ({lift_x:.10g} {lift_y:.10g} 0);
        dragDir         ({drag_x:.10g} {drag_y:.10g} 0);
        CofR            (0.25 0 0);
        pitchAxis       (0 0 1);
        magUInf         {U_mag:.10g};
        lRef            1.0;
        Aref            1.0;
    }}

    residuals
    {{
        type            residuals;
        libs            ("libutilityFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          (U p k omega);
    }}

    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         (airfoilWalls);
    }}

    writePhi
    {{
        type            writeObjects;
        libs            ("libutilityFunctionObjects.so");
        writeControl    writeTime;
        writeOption     anyWrite;
        objects         (phi);
    }}
}}
""")


FV_SCHEMES = (FOAM_HEADER + """
ddtSchemes
{{
    default         steadyState;
}}

gradSchemes
{{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(p)         Gauss linear;
}}

divSchemes
{{
    default                         none;
    div(phi,U)                      bounded Gauss linearUpwindV grad(U);
    div(phi,k)                      bounded Gauss upwind;
    div(phi,omega)                  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U)))))   Gauss linear;
    div((nuEff*dev(T(grad(U)))))    Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}

fluxRequired
{{
    default         no;
    p;
    phi;
}}

wallDist
{{
    method          meshWave;
}}
""")


FV_SOLUTION = (FOAM_HEADER + """
solvers
{{
    p
    {{
        solver              GAMG;
        tolerance           1e-7;
        relTol              0.01;
        smoother            DICGaussSeidel;
        cacheAgglomeration  true;
        agglomerator        faceAreaPair;
        nCellsInCoarsestLevel 10;
        mergeLevels         1;
    }}

    "(U|k|omega)"
    {{
        solver              smoothSolver;
        smoother            symGaussSeidel;
        tolerance           1e-8;
        relTol              0.1;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 2;
    consistent              yes;

    residualControl
    {{
        p               1e-4;
        U               1e-5;
        "(k|omega)"     1e-5;
    }}
}}

relaxationFactors
{{
    fields
    {{
        p               0.3;
    }}
    equations
    {{
        U               0.7;
        "(k|omega)"     0.7;
    }}
}}

cache
{{
    grad(U);
}}
""")


# ---------------------------------------------------------------------------
# Per-file writers
# ---------------------------------------------------------------------------

def _h(cls: str, obj: str) -> str:
    return FOAM_HEADER.format(cls=cls, obj=obj)


def write_zero_dir(zero_dir: Path, spec: CaseSpec) -> None:
    inl = spec.inlet()
    zero_dir.mkdir(parents=True, exist_ok=True)
    (zero_dir / "U").write_text(U_TEMPLATE.format(
        cls="volVectorField", obj="U", U_x=inl.U_x, U_y=inl.U_y))
    (zero_dir / "p").write_text(P_TEMPLATE.format(
        cls="volScalarField", obj="p"))
    (zero_dir / "k").write_text(K_TEMPLATE.format(
        cls="volScalarField", obj="k", k=inl.k_inlet))
    (zero_dir / "omega").write_text(OMEGA_TEMPLATE.format(
        cls="volScalarField", obj="omega", omega=inl.omega_inlet))
    (zero_dir / "nut").write_text(NUT_TEMPLATE.format(
        cls="volScalarField", obj="nut"))
    (zero_dir / "phi").write_text(PHI_TEMPLATE.format(
        cls="surfaceScalarField", obj="phi"))


def write_constant_dir(constant_dir: Path, spec: CaseSpec) -> None:
    constant_dir.mkdir(parents=True, exist_ok=True)
    (constant_dir / "momentumTransport").write_text(
        MOMENTUM_TRANSPORT.format(cls="dictionary", obj="momentumTransport"))
    (constant_dir / "physicalProperties").write_text(
        PHYSICAL_PROPERTIES.format(cls="dictionary", obj="physicalProperties", nu=NU))


def write_system_dir(system_dir: Path, spec: CaseSpec, end_time: int = 5000,
                     write_interval: int = 500) -> None:
    inl = spec.inlet()
    aoa_rad = spec.aoa_deg * 3.14159265358979 / 180.0
    import math
    lift_x = -math.sin(aoa_rad)
    lift_y = math.cos(aoa_rad)
    drag_x = math.cos(aoa_rad)
    drag_y = math.sin(aoa_rad)

    # Guarantee at least one solution write at end_time, even for short runs.
    effective_interval = min(write_interval, end_time)
    if end_time % effective_interval != 0:
        effective_interval = end_time

    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "controlDict").write_text(CONTROL_DICT.format(
        cls="dictionary", obj="controlDict",
        end_time=end_time, write_interval=effective_interval,
        lift_x=lift_x, lift_y=lift_y, drag_x=drag_x, drag_y=drag_y,
        U_mag=inl.U_mag,
    ))
    (system_dir / "fvSchemes").write_text(
        FV_SCHEMES.format(cls="dictionary", obj="fvSchemes"))
    (system_dir / "fvSolution").write_text(
        FV_SOLUTION.format(cls="dictionary", obj="fvSolution"))


def setup_openfoam_case(of_case_dir: Path, spec: CaseSpec, end_time: int = 5000) -> None:
    of_case_dir.mkdir(parents=True, exist_ok=True)
    write_zero_dir(of_case_dir / "0", spec)
    write_constant_dir(of_case_dir / "constant", spec)
    write_system_dir(of_case_dir / "system", spec, end_time=end_time)
    LOG.info("[%s] OpenFOAM case files written under %s", spec.case_id, of_case_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write OpenFOAM case files for one case.")
    p.add_argument("case_id")
    p.add_argument("--of-case", required=True, type=Path)
    p.add_argument("--end-time", type=int, default=5000)
    p.add_argument("--split", default="train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from common import parse_case_id
    parsed = parse_case_id(args.case_id)
    spec = CaseSpec.build(parsed["naca_code"], parsed["aoa_deg"], parsed["Re"], args.split)
    setup_openfoam_case(args.of_case, spec, end_time=args.end_time)


if __name__ == "__main__":
    main()
