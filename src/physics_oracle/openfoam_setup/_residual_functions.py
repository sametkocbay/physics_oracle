"""Per-cell residual capture for ML-warm-started OpenFOAM cases.

OpenFOAM 13 Foundation has no ``solverInfo writeResidualFields`` (an ESI-fork
feature) — its built-in ``residuals`` function object only writes one *scalar*
per field per iteration.  To get *per-cell* (spatial) residuals we ship a
``coded`` function object: a C++ snippet embedded in ``controlDict`` that
OpenFOAM compiles on the fly.  It evaluates the **explicit** residual of each
discretised governing equation at the field it is given and writes the result
as a ``volScalarField`` into the processed time directory.

``RESIDUAL_FO_BLOCK`` is spliced into ``controlDict.functions`` by
``case_setup.setup_case_with_initial_fields(enable_residual_capture=True)`` and
evaluated by ``runner.run_residual_postprocess`` (foamPostProcess, no solve) so
the residual is measured at the ML prediction itself.

Equations (steady, incompressible; freestream/RAS kOmegaSST):

* ``momentumResidual``  — |div(phi,U) + grad(p) - laplacian(nuEff,U)
                            - div(nuEff*dev2(T(grad(U))))|
* ``continuityResidual`` — div(phi)
* ``kResidual``     — explicit kOmegaSST k-equation imbalance
* ``omegaResidual`` — explicit kOmegaSST omega-equation imbalance

``phi`` is computed inside the snippet as the face flux implied by the velocity
field (``interpolate(U) & Sf``) — the case's stored ``0/phi`` is uniform zero.
Note this is the *collocated* flux: ``continuityResidual`` therefore measures the
divergence of the interpolated velocity field, which is small but not at solver
tolerance even for a converged field (that is the Rhie-Chow correction OF adds).

The k/omega blocks reimplement OF13's ``kOmegaSST`` (see
``MomentumTransportModels/.../RAS/kOmegaSST``); they assume the model's default
coefficients (the project's cases do not override them).

Cells adjacent to the ``airfoilWalls`` patch carry wall functions
(``omegaWallFunction``, ``nutLowReWallFunction``) that override omega/nut and the
solved equation matrices — the explicit transport-equation residual is not
meaningful there, so all four fields are masked to zero in those cells.

``__NU__`` is substituted with the per-case kinematic viscosity by
``case_setup._per_case_mapping``.
"""
from __future__ import annotations


# C++ executed on functionObject::write().  Verbatim OpenFOAM `#{ ... #}` code.
_CODE_WRITE = r"""#{
    // ---- fields at the evaluation point (the ML prediction) ----
    const volVectorField& U = mesh().lookupObject<volVectorField>("U");
    const volScalarField& p = mesh().lookupObject<volScalarField>("p");
    const volScalarField& k = mesh().lookupObject<volScalarField>("k");
    const volScalarField& omega = mesh().lookupObject<volScalarField>("omega");
    const volScalarField& nut = mesh().lookupObject<volScalarField>("nut");
    const fvMesh& m = U.mesh();
    const word& tname = m.time().name();

    // Laminar kinematic viscosity (per-case literal).
    const dimensionedScalar nu
    (
        "nu", dimensionSet(0, 2, -1, 0, 0, 0, 0), scalar(__NU__)
    );
    // Wall distance.
    const volScalarField& y = wallDist::New(m).y();

    // Face flux implied by the velocity field (0/phi is uniform zero).
    surfaceScalarField phi
    (
        IOobject("phi", tname, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        fvc::interpolate(U) & m.Sf()
    );

    // ---- momentum + continuity ----
    volTensorField gradU
    (
        IOobject("grad(U)", tname, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        fvc::grad(U)
    );
    volScalarField nuEff
    (
        IOobject("nuEff", tname, m, IOobject::NO_READ, IOobject::NO_WRITE, false),
        nu + nut
    );

    volVectorField momRes
    (
        fvc::div(phi, U)
      + fvc::grad(p)
      - fvc::laplacian(nuEff, U)
      - fvc::div(nuEff*dev2(T(gradU)))
    );
    volScalarField divU(fvc::div(phi));

    // ---- kOmegaSST coefficients (OF13 Foundation defaults) ----
    const scalar alphaK1 = 0.85,    alphaK2 = 1.0;
    const scalar alphaOmega1 = 0.5, alphaOmega2 = 0.856;
    const scalar gamma1 = 5.0/9.0,  gamma2 = 0.44;
    const scalar beta1 = 0.075,     beta2 = 0.0828;
    const scalar betaStar = 0.09, a1 = 0.31, b1 = 1.0, c1 = 10.0;

    volScalarField S2(2.0*magSqr(symm(gradU)));
    volScalarField GbyNu(dev(twoSymm(gradU)) && gradU);
    volScalarField G(nut*GbyNu);

    // Blending function F1 / F2 (F3 disabled, as in the default model).
    volScalarField CDkOmega
    (
        (2.0*alphaOmega2)*(fvc::grad(k) & fvc::grad(omega))/omega
    );
    volScalarField CDkOmegaPlus
    (
        max(CDkOmega, dimensionedScalar(CDkOmega.dimensions(), 1.0e-10))
    );
    volScalarField arg1
    (
        min
        (
            min
            (
                max
                (
                    (1.0/betaStar)*sqrt(k)/(omega*y),
                    scalar(500)*nu/(sqr(y)*omega)
                ),
                (4.0*alphaOmega2)*k/(CDkOmegaPlus*sqr(y))
            ),
            scalar(10)
        )
    );
    volScalarField F1(tanh(pow4(arg1)));
    volScalarField arg2
    (
        min
        (
            max
            (
                (2.0/betaStar)*sqrt(k)/(omega*y),
                scalar(500)*nu/(sqr(y)*omega)
            ),
            scalar(100)
        )
    );
    volScalarField F23(tanh(sqr(arg2)));

    // Blended coefficients: blend(F1, c1, c2) = F1*(c1 - c2) + c2.
    volScalarField alphaK(F1*(alphaK1 - alphaK2) + alphaK2);
    volScalarField alphaOmega(F1*(alphaOmega1 - alphaOmega2) + alphaOmega2);
    volScalarField gamma(F1*(gamma1 - gamma2) + gamma2);
    volScalarField beta(F1*(beta1 - beta2) + beta2);
    volScalarField DkEff(alphaK*nut + nu);
    volScalarField DomegaEff(alphaOmega*nut + nu);

    // k-equation residual (steady): div(phi,k) - laplacian(DkEff,k)
    //   - Pk + (2/3)*divU*k + betaStar*omega*k
    volScalarField Pk(min(G, (c1*betaStar)*k*omega));
    volScalarField kRes
    (
        fvc::div(phi, k)
      - fvc::laplacian(DkEff, k)
      - Pk
      + (2.0/3.0)*divU*k
      + betaStar*omega*k
    );

    // omega-equation residual (steady): div(phi,omega)
    //   - laplacian(DomegaEff,omega) - production
    //   + (2/3)*gamma*divU*omega + beta*omega^2 + (F1-1)*CDkOmega
    volScalarField omegaProd
    (
        gamma*min
        (
            GbyNu,
            (c1/a1)*betaStar*omega*max(a1*omega, b1*F23*sqrt(S2))
        )
    );
    volScalarField omegaRes
    (
        fvc::div(phi, omega)
      - fvc::laplacian(DomegaEff, omega)
      - omegaProd
      + (2.0/3.0)*gamma*divU*omega
      + beta*sqr(omega)
      + (F1 - 1.0)*CDkOmega
    );

    // ---- assemble per-cell residual volScalarFields ----
    volScalarField momentumResidual
    (
        IOobject("momentumResidual", tname, m, IOobject::NO_READ,
                 IOobject::NO_WRITE, false),
        mag(momRes)
    );
    volScalarField continuityResidual
    (
        IOobject("continuityResidual", tname, m, IOobject::NO_READ,
                 IOobject::NO_WRITE, false),
        divU
    );
    volScalarField kResidual
    (
        IOobject("kResidual", tname, m, IOobject::NO_READ,
                 IOobject::NO_WRITE, false),
        mag(kRes)
    );
    volScalarField omegaResidual
    (
        IOobject("omegaResidual", tname, m, IOobject::NO_READ,
                 IOobject::NO_WRITE, false),
        mag(omegaRes)
    );

    // Cells adjacent to the airfoil carry wall functions (omegaWallFunction,
    // nutLowReWallFunction): they override omega/nut and the solver manipulates
    // the equation matrices there, so the explicit transport-equation residual
    // is not meaningful — it can be enormous (omega ~ 1e7 => beta*omega^2 ~ 1e13).
    // Mask those cells to zero in every residual field.
    forAll(m.boundary(), patchi)
    {
        if (m.boundary()[patchi].name() != "airfoilWalls")
        {
            continue;
        }
        const labelUList& wc = m.boundary()[patchi].faceCells();
        forAll(wc, i)
        {
            const label c = wc[i];
            momentumResidual[c] = 0.0;
            continuityResidual[c] = 0.0;
            kResidual[c] = 0.0;
            omegaResidual[c] = 0.0;
        }
    }

    momentumResidual.write();
    continuityResidual.write();
    kResidual.write();
    omegaResidual.write();

    Info<< "[residualFields] wrote momentum/continuity/k/omega "
        << "residual fields at t=" << tname
        << " (wall-function cells masked)" << endl;
#}"""


# Names of the volScalarFields written by the coded FO above.
RESIDUAL_FIELDS = (
    "momentumResidual",
    "continuityResidual",
    "kResidual",
    "omegaResidual",
)


# Spliced into controlDict.functions by setup_case_with_initial_fields().
RESIDUAL_FO_BLOCK: dict = {
    "type": "coded",
    "libs": ['"libutilityFunctionObjects.so"'],
    "writeControl": "writeTime",
    "codeInclude": (
        "#{\n"
        '#include "fvc.H"\n'
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
    "codeWrite": _CODE_WRITE,
}
