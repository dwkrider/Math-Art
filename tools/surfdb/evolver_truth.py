"""Surface Evolver's own framed adjoint, measured per datafile.

GROUND TRUTH, not a derived quantity.  Each row is what Evolver
itself reports after running the datafile's `adj` and `frame`
commands on a mesh refined three times -- the area of the framed
conjugate patch and its bounding box.  It exists because the
topological checks cannot tell a right cell from a wrong one: a
wrong subgroup, a wrong family member and a wrong Gauss map have
each produced one clean sheet in this project and been wrong.

Reproduce with, per datafile,

    evolver64.exe -f script.txt <file>.fe

where the script is `r r "g 40" r "g 60" adj [frame] "g 60" u V
"g 60"` followed by a printf of `total_area` and the six vertex
extrema.  Nine of the forty-three did not finish inside the
three-minute limit and are simply absent; for those the bake
falls back to the boundary-to-plane residual.

Evolver is Ken Brakke's, and the datafiles are his:
  K. A. Brakke, "The Surface Evolver", Experimental Mathematics
  1(2) (1992) 141-165; https://kenbrakke.com/evolver/
"""

EVOLVER_ADJOINT = {
    'FRDadj.fe': {'facets': 256, 'area': 2.524345000,
        'bbox': (2.518000, 1.373000, 1.750000)},
    'GW5adj.fe': {'facets': 384, 'area': 1.535103000,
        'bbox': (0.964000, 1.068000, 1.890000)},
    'HRHTadj.fe': {'facets': 384, 'area': 0.970879000,
        'bbox': (0.448000, 1.220000, 0.877000)},
    'HRadj.fe': {'facets': 320, 'area': 0.982086000,
        'bbox': (0.505000, 1.265000, 0.866000)},
    'HTTRadj.fe': {'facets': 384, 'area': 0.706947000,
        'bbox': (1.704000, 0.470000, 0.510000)},
    'HTadj.fe': {'facets': 320, 'area': 1.093460000,
        'bbox': (0.743000, 0.960000, 1.000000)},
    'TRHTadj.fe': {'facets': 384, 'area': 0.601121000,
        'bbox': (1.500000, 0.866000, 0.436000)},
    'TRadj.fe': {'facets': 320, 'area': 0.713895000,
        'bbox': (1.756000, 0.475000, 0.500000)},
    'batwing41adj.fe': {'facets': 320, 'area': 0.872728000,
        'bbox': (1.000000, 1.410000, 1.000000)},
    'batwing57adj.fe': {'facets': 384, 'area': 0.994612000,
        'bbox': (1.000000, 1.625000, 1.055000)},
    'batwingadj.fe': {'facets': 256, 'area': 0.209222000,
        'bbox': (0.500000, 0.551000, 0.676000)},
    'disphenoid31adj.fe': {'facets': 320, 'area': 1.393189000,
        'bbox': (1.702000, 1.298000, 1.167000)},
    'disphenoid35adj.fe': {'facets': 320, 'area': 1.300827000,
        'bbox': (1.399000, 1.371000, 0.785000)},
    'disphenoid43adj.fe': {'facets': 384, 'area': 1.152553000,
        'bbox': (1.057000, 1.447000, 1.087000)},
    'disphenoid51adj.fe': {'facets': 384, 'area': 1.623843000,
        'bbox': (1.629000, 1.345000, 1.091000)},
    'disphenoid55adj.fe': {'facets': 448, 'area': 0.981677000,
        'bbox': (0.878000, 1.300000, 0.996000)},
    'disphenoid67adj.fe': {'facets': 448, 'area': 1.555077000,
        'bbox': (1.687000, 1.175000, 0.881000)},
    'hexplane1adj.fe': {'facets': 256, 'area': 0.397991000,
        'bbox': (1.000000, 0.546000, 0.694000)},
    'hexplane2adj.fe': {'facets': 320, 'area': 0.447743000,
        'bbox': (1.000000, 0.284000, 0.844000)},
    'hexplane3adj.fe': {'facets': 384, 'area': 0.464385000,
        'bbox': (1.000000, 0.210000, 0.880000)},
    'hexplane4adj.fe': {'facets': 448, 'area': 0.473830000,
        'bbox': (1.000000, 0.161000, 0.911000)},
    'hexplane5adj.fe': {'facets': 512, 'area': 0.477756000,
        'bbox': (1.000000, 0.155000, 0.905000)},
    'hybrid-1adj.fe': {'facets': 320, 'area': 0.482722000,
        'bbox': (0.969000, 0.809000, 0.617000)},
    'manta35adj.fe': {'facets': 320, 'area': 0.232394000,
        'bbox': (0.500000, 0.404000, 0.826000)},
    'manta51adj.fe': {'facets': 384, 'area': 0.237875000,
        'bbox': (0.500000, 0.419000, 0.857000)},
    'pbatadj.fe': {'facets': 512, 'area': 0.418776000,
        'bbox': (1.000000, 0.674000, 0.674000)},
    'pssadj.fe': {'facets': 384, 'area': 1.299512000,
        'bbox': (0.639000, 1.530000, 1.053000)},
    's12adj.fe': {'facets': 512, 'area': 14.906045000,
        'bbox': (2.393000, 2.594000, 5.015000)},
    'triplane0adj.fe': {'facets': 64, 'area': 0.320298000,
        'bbox': (1.000000, 0.496000, 0.504000)},
    'triplane1adj.fe': {'facets': 256, 'area': 0.431840000,
        'bbox': (1.000000, 0.339000, 0.801000)},
    'triplane2adj.fe': {'facets': 320, 'area': 0.457402000,
        'bbox': (1.000000, 0.215000, 0.901000)},
    'triplane3adj.fe': {'facets': 384, 'area': 0.458841000,
        'bbox': (1.000000, 0.186000, 0.871000)},
    'triplane4adj.fe': {'facets': 448, 'area': 0.461405000,
        'bbox': (1.000000, 0.221000, 0.890000)},
    'triplane5adj.fe': {'facets': 512, 'area': 0.455314000,
        'bbox': (1.000000, 0.146000, 0.887000)},
}


# The same measurement for the datafiles that are NOT adjoints.
# These define their surface directly, but most of them still have a
# free boundary sliding on constraint planes, so "span the polygon"
# is just as wrong for them -- and just as invisible to a
# topological check.
EVOLVER_PINNED = {
    '3.fe': {'facets': 4992, 'area': 38.204141337},
    '4.fe': {'facets': 7168, 'area': 49.745128798},
    '4d.fe': {'facets': 256, 'area': 9.268268575},
    'CLP.fe': {'facets': 384, 'area': 1.783975018},
    'CScell.fe': {'facets': 512, 'area': 0.196224335},
    'I-6.fe': {'facets': 2048, 'area': 5.267984013},
    'I-8.fe': {'facets': 2048, 'area': 2.601626872},
    'I-9.fe': {'facets': 3072, 'area': 5.050949183},
    'IWP.fe': {'facets': 256, 'area': 0.289406534},
    'RII.fe': {'facets': 768, 'area': 0.610187610},
    'RIII.fe': {'facets': 768, 'area': 3.589169840},
    'cd.fe': {'facets': 896, 'area': 1.326588515},
    'dcell.fe': {'facets': 64, 'area': 0.320266526},
    'disphenoid19.fe': {'facets': 256, 'area': 1.319952882},
    'hcell.fe': {'facets': 768, 'area': 8.846655518},
    'neovius.fe': {'facets': 256, 'area': 0.293167067},
    'pcell.fe': {'facets': 256, 'area': 0.195588650},
    'ycell.fe': {'facets': 384, 'area': 0.160953365},
}
