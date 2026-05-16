"""Coded function object that dumps the mesh geometry OpenFOAM uses.

`mesh.h5` and the polyMesh store only topology — face-area vectors, cell
volumes, interpolation weights and non-orthogonal correction data are computed
by `fvMesh` at runtime and never persisted.  The differentiable residual
(`residual_diff/`) needs them to build FVM operators that exactly match OF's.

`MESH_GEOMETRY_FO` is a `coded` function object (same dict pattern as
`RESIDUAL_FO_BLOCK` in `_residual_functions.py`) whose `codeWrite` writes, into
the processed time directory:

  Sf, magSf, meshWeights, nonOrthDeltaCoeffs, nonOrthCorrectionVectors, Cf
      — surface fields (internal faces + per-patch boundary faces)
  Cv          — cell centres (volVectorField)
  cellVolume  — cell volumes (volScalarField)
  wallY       — wall distance (volScalarField)

It is run once per mesh by `residual_diff.geometry.export_mesh_geometry`.
"""
from __future__ import annotations


_GEOMETRY_CODE = r"""#{
    const fvMesh& m = mesh();
    const word& t = m.time().name();

    surfaceVectorField
    (
        IOobject("Sf", t, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        m.Sf()
    ).write();
    surfaceScalarField
    (
        IOobject("magSf", t, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        m.magSf()
    ).write();
    surfaceScalarField
    (
        IOobject("meshWeights", t, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        m.weights()
    ).write();
    surfaceScalarField
    (
        IOobject("nonOrthDeltaCoeffs", t, m, IOobject::NO_READ,
                 IOobject::NO_WRITE, false),
        m.nonOrthDeltaCoeffs()
    ).write();
    surfaceVectorField
    (
        IOobject("nonOrthCorrectionVectors", t, m, IOobject::NO_READ,
                 IOobject::NO_WRITE, false),
        m.nonOrthCorrectionVectors()
    ).write();
    surfaceVectorField
    (
        IOobject("Cf", t, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        m.Cf()
    ).write();
    volVectorField
    (
        IOobject("Cv", t, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        m.C()
    ).write();
    volScalarField cellVolume
    (
        IOobject("cellVolume", t, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        m,
        dimensionedScalar(dimVolume, 0.0)
    );
    cellVolume.primitiveFieldRef() = m.V();
    cellVolume.write();
    volScalarField
    (
        IOobject("wallY", t, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        wallDist::New(m).y()
    ).write();

    Info<< "[meshGeometry] wrote Sf/magSf/meshWeights/nonOrthDeltaCoeffs/"
        << "nonOrthCorrectionVectors/Cf/Cv/cellVolume/wallY" << endl;
#}"""


# Filenames written by the FO above.
GEOMETRY_FIELDS = (
    "Sf", "magSf", "meshWeights", "nonOrthDeltaCoeffs",
    "nonOrthCorrectionVectors", "Cf", "Cv", "cellVolume", "wallY",
)


MESH_GEOMETRY_FO: dict = {
    "type": "coded",
    "libs": ['"libutilityFunctionObjects.so"'],
    "writeControl": "writeTime",
    "codeInclude": (
        "#{\n"
        '#include "surfaceFields.H"\n'
        '#include "wallDist.H"\n'
        "#}"
    ),
    "codeOptions": (
        "#{\n"
        "-I$(LIB_SRC)/finiteVolume/lnInclude "
        "-I$(LIB_SRC)/meshTools/lnInclude\n"
        "#}"
    ),
    "codeLibs": "#{\n-lfiniteVolume -lmeshTools\n#}",
    "codeWrite": _GEOMETRY_CODE,
}
