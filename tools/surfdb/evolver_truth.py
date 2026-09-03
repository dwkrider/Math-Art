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
    'FRDadj.fe': {'facets': 512, 'area': 2.521516664},
    'GW5adj.fe': {'facets': 800, 'area': 1.551646267},
    'HRHTadj.fe': {'facets': 1056, 'area': 0.965668985},
    'HRadj.fe': {'facets': 848, 'area': 0.981237898},
    'HTTRadj.fe': {'facets': 928, 'area': 0.662341950},
    'HTadj.fe': {'facets': 976, 'area': 1.091228928},
    'TRHTadj.fe': {'facets': 928, 'area': 0.589273118},
    'TRadj.fe': {'facets': 848, 'area': 0.669505940},
    'batwing41adj.fe': {'facets': 2592, 'area': 0.939701973},
    'batwing57adj.fe': {'facets': 3072, 'area': 0.992967684},
    'batwingadj.fe': {'facets': 2112, 'area': 0.208619534},
    'disphenoid31adj.fe': {'facets': 640, 'area': 1.464030193},
    'disphenoid35adj.fe': {'facets': 640, 'area': 1.566759848},
    'disphenoid43adj.fe': {'facets': 800, 'area': 1.623732361},
    'disphenoid51adj.fe': {'facets': 832, 'area': 1.694654686},
    'disphenoid55adj.fe': {'facets': 912, 'area': 1.655730970},
    'disphenoid67adj.fe': {'facets': 976, 'area': 1.754760669},
    'hexplane1adj.fe': {'facets': 2048, 'area': 0.397544287},
    'hexplane2adj.fe': {'facets': 2560, 'area': 0.447080749},
    'hexplane3adj.fe': {'facets': 3136, 'area': 0.464275580},
    'hexplane4adj.fe': {'facets': 3648, 'area': 0.473043942},
    'hexplane5adj.fe': {'facets': 4160, 'area': 0.478361476},
    'hybrid-1adj.fe': {'facets': 2592, 'area': 0.481570300},
    'manta35adj.fe': {'facets': 2560, 'area': 0.233817764},
    'manta51adj.fe': {'facets': 3072, 'area': 0.247569809},
    'pbatadj.fe': {'facets': 4128, 'area': 0.417358030},
    'pssadj.fe': {'facets': 1648, 'area': 1.286916649},
    's12adj.fe': {'facets': 3072, 'area': 14.211656469},
    'triplane0adj.fe': {'facets': 1024, 'area': 0.319847348},
    'triplane1adj.fe': {'facets': 2048, 'area': 0.430359797},
    'triplane2adj.fe': {'facets': 2560, 'area': 0.457359502},
    'triplane3adj.fe': {'facets': 3072, 'area': 0.469293028},
    'triplane4adj.fe': {'facets': 3584, 'area': 0.476015048},
    'triplane5adj.fe': {'facets': 4096, 'area': 0.480324313},

# Measured only after EVOLVERPATH was set.  These eleven were the
# "Evolver did not finish" rows, and it had not: each pulls in a stock
# script from the distribution's `fe/` directory (`multiplicate.cmd`
# and friends), Evolver could not find it without a search path, and
# it fell back to its interactive prompt and printed there until the
# harness killed it.  With the path set they finish in a second or two.
    'N14.fe': {'facets': 706, 'area': 0.361810283},
    'N26.fe': {'facets': 3584, 'area': 0.433438248},
    'N38.fe': {'facets': 4744, 'area': 0.425011610},
    'mantaadj.fe': {'facets': 2122, 'area': 0.205925589},
    'octoadj.fe': {'facets': 2631, 'area': 0.305585753},
    's14adj.fe': {'facets': 2304, 'area': 0.749870651},
    'c21padj.fe': {'facets': 2490, 'area': 0.612368191},
    'c27padj.fe': {'facets': 2889, 'area': 0.616609526},
    'c33padj.fe': {'facets': 2810, 'area': 0.621486010},
    'c39padj.fe': {'facets': 3851, 'area': 0.623528729},
    'c45padj.fe': {'facets': 3432, 'area': 0.626047633},
}


# The same measurement for the datafiles that are NOT adjoints.
# These define their surface directly, but most of them still have a
# free boundary sliding on constraint planes, so "span the polygon"
# is just as wrong for them -- and just as invisible to a
# topological check.
EVOLVER_PINNED = {
    '3.fe': {'facets': 4992, 'area': 38.204141337},
    '4.fe': {'facets': 1756, 'area': 46.656045814},
    '4d.fe': {'facets': 256, 'area': 9.268268575},
    'CLP.fe': {'facets': 1536, 'area': 1.782231949},
    'CScell.fe': {'facets': 1792, 'area': 0.195620693},
    'I-6.fe': {'facets': 8192, 'area': 5.241999990},
    'I-8.fe': {'facets': 7808, 'area': 2.477692022},
    'I-9.fe': {'facets': 4608, 'area': 4.951994494},
    'IWP.fe': {'facets': 384, 'area': 0.288945123},
    'RII.fe': {'facets': 1780, 'area': 0.607840297},
    'RIII.fe': {'facets': 1152, 'area': 3.552560315},
    'cd.fe': {'facets': 976, 'area': 1.318050169},
    'dcell.fe': {'facets': 256, 'area': 0.319926025},
    'disphenoid19.fe': {'facets': 320, 'area': 1.319376584},
    'hcell.fe': {'facets': 1536, 'area': 8.832897498},
    'neovius.fe': {'facets': 384, 'area': 0.292791713},
    'pcell.fe': {'facets': 272, 'area': 0.195509583},
    'ycell.fe': {'facets': 768, 'area': 0.160155976},
    'CYcell.fe': {'facets': 2094, 'area': 0.275650087},
    'Scell.fe': {'facets': 2880, 'area': 0.338854698},
    'bubble2.fe': {'facets': 704, 'area': 9.167415701},
    'cat1disk.fe': {'facets': 1536, 'area': 7.475469253},
    'cat1diskpart.fe': {'facets': 6256, 'area': 2.438107791},
    'cat2disk.fe': {'facets': 1792, 'area': 10.177594500},
    'catdisk.fe': {'facets': 7680, 'area': 14.297287019},
    'cbga1.fe': {'facets': 504, 'area': 0.024170933},
    'cbga2.fe': {'facets': 600, 'area': 0.030439167},
    'crosscat.fe': {'facets': 1024, 'area': 4.826912474},
    'cubble.fe': {'facets': 2304, 'area': 74.375909710},
    'cube.fe': {'facets': 384, 'area': 4.852986458},
    'hncusp.fe': {'facets': 2320, 'area': 6.150046932},
    'loops.fe': {'facets': 1536, 'area': 3.190203024},
    'mound.fe': {'facets': 388, 'area': 12.846221872},
    'octacone.fe': {'facets': 1408, 'area': 22.879133038},
    'octahex.fe': {'facets': 2688, 'area': 5.714265816},
    'octakites.fe': {'facets': 3328, 'area': 5.794916036},
    'octapent.fe': {'facets': 2816, 'area': 5.726269645},
    'octaquad.fe': {'facets': 2816, 'area': 5.718907865},
    'octasq.fe': {'facets': 2944, 'area': 5.752752579},
    'quad.fe': {'facets': 256, 'area': 4.318201496},
    'ringblob.fe': {'facets': 512, 'area': 37.003574191},
    'spiral.fe': {'facets': 9302, 'area': 1.727411798},
    'tankex.fe': {'facets': 448, 'area': 3.498090612},
    'twointor.fe': {'facets': 1152, 'area': 3.343153075},
}
