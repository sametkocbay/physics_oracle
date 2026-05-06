// Gmsh geometry for quasi-2D flow over a single cylinder.
// The surface mesh is extruded to one recombined layer in z so gmshToFoam
// imports a 2D-compatible prism/hex mesh for an OpenFOAM case with empty
// front/back boundaries.

SetFactory("OpenCASCADE");

// -----------------------------------------------------------------------------
// Global geometry parameters
// -----------------------------------------------------------------------------
D  = 0.1;
R  = 0.5 * D;

cx = 0.0;
cy = 0.0;

xMin = -10.0 * D;
xMax =  25.0 * D;
yMin = -10.0 * D;
yMax =  10.0 * D;

dz  = 0.01 * D;
eps = 1e-6;

// -----------------------------------------------------------------------------
// Mesh sizing parameters
// -----------------------------------------------------------------------------
lcFar  = 0.50 * D;
lcNear = 0.03 * D;
lcWake = 0.05 * D;

// Boundary-layer sizing chosen for the current Re ~= 15000 operating point.
nBLayers     = 14;
blGrowth     = 1.20;
firstLayer   = 1.0e-4;
blThickness  = firstLayer * (blGrowth^nBLayers - 1.0) / (blGrowth - 1.0);
blFarSize    = lcNear;

// Wake refinement downstream of the single cylinder.
wakeHalfHeight = 2.5 * D;
wakeLength     = 10.0 * D;
wakeStart      = cx + R;
wakeEnd        = wakeStart + wakeLength;

// -----------------------------------------------------------------------------
// Mesh algorithm and output settings for OpenFOAM compatibility
// -----------------------------------------------------------------------------
Mesh.Algorithm               = 6;   // Frontal-Delaunay for robust CFD triangles
Mesh.MeshSizeFromPoints      = 0;
Mesh.MeshSizeFromCurvature   = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.Optimize                = 1;
Mesh.OptimizeNetgen          = 1;
Mesh.Smoothing               = 10;
Mesh.MshFileVersion          = 2.2; // gmshToFoam is most reliable with MSH2

// -----------------------------------------------------------------------------
// Base 2D geometry: outer rectangle minus one circular cylinder
// -----------------------------------------------------------------------------
Rectangle(1) = {xMin, yMin, 0.0, xMax - xMin, yMax - yMin};
Disk(2)      = {cx, cy, 0.0, R, R};

fluid2D[] = BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; };

// -----------------------------------------------------------------------------
// Recover curves by location so the script stays robust after boolean operations
// -----------------------------------------------------------------------------
inletCurves[]  = Curve In BoundingBox {xMin - eps, yMin - eps, -eps, xMin + eps, yMax + eps,  eps};
outletCurves[] = Curve In BoundingBox {xMax - eps, yMin - eps, -eps, xMax + eps, yMax + eps,  eps};
bottomCurves[] = Curve In BoundingBox {xMin - eps, yMin - eps, -eps, xMax + eps, yMin + eps,  eps};
topCurves[]    = Curve In BoundingBox {xMin - eps, yMax - eps, -eps, xMax + eps, yMax + eps,  eps};

cylinderCurves[] = Curve In BoundingBox {cx - R - eps, cy - R - eps, -eps, cx + R + eps, cy + R + eps, eps};

// Use transfinite spacing on the cylinder curves to keep the near-wall spacing controlled.
Transfinite Curve {cylinderCurves[]} = 128 Using Progression 1.0;

// -----------------------------------------------------------------------------
// Size fields: boundary layer, near-cylinder refinement, and wake boxes
// -----------------------------------------------------------------------------
Field[1] = BoundaryLayer;
Field[1].CurvesList       = {cylinderCurves[]};
Field[1].Size             = firstLayer;
Field[1].SizeFar          = blFarSize;
Field[1].Thickness        = blThickness;
Field[1].Ratio            = blGrowth;
Field[1].NbLayers         = nBLayers;
Field[1].IntersectMetrics = 1;
Field[1].Quads            = 1;

Field[2] = Distance;
Field[2].CurvesList = {cylinderCurves[]};
Field[2].Sampling   = 250;

Field[3] = Threshold;
Field[3].InField  = 2;
Field[3].SizeMin  = lcNear;
Field[3].SizeMax  = lcFar;
Field[3].DistMin  = 1.0 * D;
Field[3].DistMax  = 4.0 * D;

Field[4] = Box;
Field[4].VIn      = lcWake;
Field[4].VOut     = lcFar;
Field[4].XMin     = wakeStart;
Field[4].XMax     = wakeEnd;
Field[4].YMin     = -wakeHalfHeight;
Field[4].YMax     =  wakeHalfHeight;
Field[4].ZMin     = -eps;
Field[4].ZMax     =  eps;
Field[4].Thickness = 1.5 * D;

Field[5] = Min;
Field[5].FieldsList = {1, 3, 4};
Background Field = 5;

// -----------------------------------------------------------------------------
// Extrude the 2D mesh to one cell in z for a quasi-2D OpenFOAM mesh
// -----------------------------------------------------------------------------
extruded[] = Extrude {0.0, 0.0, dz}
{
    Surface{fluid2D[0]};
    Layers{1};
    Recombine;
};

// -----------------------------------------------------------------------------
// Recover 3D boundary surfaces for physical groups used by gmshToFoam/OpenFOAM
// -----------------------------------------------------------------------------
inletSurfaces[]  = Surface In BoundingBox {xMin - eps, yMin - eps, -eps, xMin + eps, yMax + eps, dz + eps};
outletSurfaces[] = Surface In BoundingBox {xMax - eps, yMin - eps, -eps, xMax + eps, yMax + eps, dz + eps};
bottomSurfaces[] = Surface In BoundingBox {xMin - eps, yMin - eps, -eps, xMax + eps, yMin + eps, dz + eps};
topSurfaces[]    = Surface In BoundingBox {xMin - eps, yMax - eps, -eps, xMax + eps, yMax + eps, dz + eps};

cylinderSurfaces[] = Surface In BoundingBox {cx - R - eps, cy - R - eps, -eps, cx + R + eps, cy + R + eps, dz + eps};

frontSurfaces[]  = Surface In BoundingBox {xMin - eps, yMin - eps, -eps,      xMax + eps, yMax + eps,  eps};
backSurfaces[]   = Surface In BoundingBox {xMin - eps, yMin - eps, dz - eps,   xMax + eps, yMax + eps, dz + eps};

// -----------------------------------------------------------------------------
// Physical groups expected by OpenFOAM plus front/back and fluid volume
// -----------------------------------------------------------------------------
Physical Surface("inlet")         = {inletSurfaces[]};
Physical Surface("outlet")        = {outletSurfaces[]};
Physical Surface("top")           = {topSurfaces[]};
Physical Surface("bottom")        = {bottomSurfaces[]};
Physical Surface("cylinderWalls") = {cylinderSurfaces[]};
Physical Surface("frontAndBack")  = {frontSurfaces[], backSurfaces[]};
Physical Volume("fluid")          = {extruded[1]};

// -----------------------------------------------------------------------------
// Generate the final quasi-2D volume mesh
// -----------------------------------------------------------------------------
Mesh 3;