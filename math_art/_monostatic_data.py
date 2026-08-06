
# Spherical-harmonic model of the "real" (fabricated) Gomboc.
#
# The genuine mono-monostatic Gomboc has no single closed-form
# equation: as its makers describe (gomboc.eu), it is a "tennis-ball"
# assembly of segments of simple surfaces (cylinder, ellipsoid, cone)
# and planes, engineered so the class {1,1} equilibrium is robust.
# (The Domokos-Varkonyi 2006 existence proof and the Sloan analytic
# gombocs are DIFFERENT, essentially spherical bodies -- see
# monostatic_body_generator.py.)
#
# To reproduce the RECOGNISABLE fabricated shape, its convex,
# star-shaped boundary was digitised from a reference model and fit as
# a real spherical-harmonic expansion of the radial function about the
# centre of mass,  R = sum_l sum_m c[l,m] Y_lm , using the
# (unnormalised real) basis
#     Y_lm = P_l^{|m|}(cos phi) * (cos(m th) if m>=0 else sin(|m| th))
# with phi the polar angle and th the azimuth.  Degree L = 12 (169
# coefficients) fits the reference to ~0.8%% RMS of the mean radius.
# Coefficients are normalised to a mean radius of 1; c[k] is in the
# canonical order l = 0..L, m = -l..+l.
#
# Reference shape: the fabricated Gomboc (Domokos & Varkonyi;
# gomboc.eu). See also Wolfram MathWorld, "Gomboc".

GOMBOC_SH_L = 12

GOMBOC_SH_COEF = (
    0.9633825, -0.1503896, -3.836056e-06, 3.30984e-06, -4.718482e-08,
    2.411058e-07, 0.109803, -5.349407e-07, -0.01473761, -0.002848746,
    -1.22788e-08, -0.0335369, -4.753487e-07, 2.220853e-06, -1.211156e-08,
    7.13418e-08, -2.457514e-09, -2.120947e-08, -1.748196e-08,
    -2.428328e-07, -0.04629666, 3.330422e-07, -0.002353506, -8.684977e-09,
    0.0001853746, 1.192479e-05, 1.081174e-09, -0.0005550292,
    -1.142609e-08, 0.007219142, 7.130669e-08, -1.330837e-07, -1.34536e-09,
    8.084174e-12, 4.49274e-11, -6.981906e-12, 9.088668e-13, 1.870125e-12,
    8.599693e-13, -1.095352e-10, -1.629972e-10, 1.904562e-08, 0.03607643,
    6.854903e-09, -0.0005142483, 6.084998e-10, 2.356098e-06,
    -5.434475e-11, -1.614879e-06, -1.143998e-07, 5.151082e-12,
    -1.007343e-06, 3.183169e-11, -0.0001102598, 1.926168e-09, 0.001435896,
    1.66792e-07, -1.036409e-10, -3.725163e-10, -4.161329e-11,
    -1.287151e-11, 6.514248e-13, -2.096421e-13, 1.828161e-14,
    -4.942344e-15, 8.383172e-15, -1.174889e-13, 1.459367e-12,
    -8.753358e-12, 8.224602e-11, 2.259197e-10, -1.768654e-08, 0.0432773,
    -5.651637e-09, -0.0003384191, 1.068425e-10, 2.105824e-06, 3.18388e-12,
    -1.115271e-07, 4.837101e-13, 4.280795e-09, 1.367014e-10, 1.456435e-14,
    -9.390165e-09, -6.22551e-13, 6.4545e-08, 4.21723e-11, -2.432963e-05,
    -2.192916e-09, -0.002186121, -1.226518e-07, 1.242819e-09,
    -3.119621e-10, -2.079982e-11, -1.095896e-12, -6.288513e-14,
    -1.032905e-13, 1.151322e-14, 8.107558e-16, -5.68869e-15,
    -5.526907e-16, -1.672567e-15, 1.195887e-14, 9.148022e-15,
    -1.036168e-14, -7.391433e-14, 1.689978e-12, -7.427391e-11,
    2.10036e-11, 1.558282e-08, -0.01061525, -4.281593e-09, -5.374899e-05,
    3.863446e-11, 5.965222e-07, 5.663974e-14, -2.259094e-08, 3.035407e-14,
    6.651518e-11, 4.275682e-16, -5.913242e-12, -3.23228e-13, 1.996381e-16,
    -6.333372e-12, -3.215229e-16, -6.975483e-10, 2.238909e-13,
    -2.329984e-08, -3.472572e-12, -1.583533e-05, 7.759859e-10, 0.0034319,
    5.084693e-08, -1.824039e-10, 2.576814e-10, 5.910513e-13,
    -4.519219e-12, -9.634043e-14, 2.928428e-14, -2.429568e-15,
    -2.973452e-15, -1.02967e-15, -8.766543e-18, -5.100534e-17,
    1.394677e-18, -5.937341e-17, 1.434525e-17, 4.781751e-16,
    -5.142175e-15, -3.152967e-16, -2.376322e-17, -2.912028e-13,
    -1.788714e-13, 2.824179e-11, -1.932287e-11, -7.979966e-09,
    -0.008718949, 2.743857e-09, 0.0001167077, 3.23005e-12, -7.198161e-08,
    -4.125008e-13, -1.120952e-09, 1.294592e-14, -2.768278e-11,
    2.94389e-16, -6.712096e-13, 1.275134e-17, 2.223833e-14,
)
