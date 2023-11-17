import cupy as cp
import numpy as np

def chebyshev_t(c, x):
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(cp.double)
    if isinstance(x, (tuple, list)):
        x = cp.asarray(x)
    if isinstance(x, cp.ndarray):
        c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x    

def popov_hopping(dR):
    ang_per_bohr = 0.529177  # [Angstroms/Bohr radius]
    eV_per_hart = 27.2114  # [eV/Hartree]

    dR = dR / ang_per_bohr
    dRn = cp.linalg.norm(dR, axis=1)
    dRn[dRn == 0] = 1  # Avoid division by zero

    dRn = dR / dRn[:, None]

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = cp.linalg.norm(dR, axis=1)
    r = cp.clip(r, 1, 10)

    aa = 1.0  # [Bohr radii]
    b = 10.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    #Css_sigma = cp.array([-0.5286482, 0.4368816, -0.2390807, 0.0701587, 0.0106355, -0.0258943, 0.0169584, -0.0070929, 0.0019797, -0.000304])
    #Csp_sigma = cp.array([0.3865122, -0.2909735, 0.1005869, 0.0340820, -0.0705311, 0.0528565, -0.0270332, 0.0103844, -0.0028724, 0.0004584])
    Cpp_sigma = cp.array([0.1727212, -0.0937225, -0.0445544, 0.1114266, -0.0978079, 0.0577363, -0.0262833, 0.0094388, -0.0024695, 0.0003863])
    Cpp_pi = cp.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478, -0.0535682, 0.0181983, -0.0046855, 0.0007303, 0.0000225, -0.0000393])

    #Vss_sigma = cp.array([chebyshev_t(Css_sigma, yi) for yi in y])
    #Vsp_sigma = cp.array([chebyshev_t(Csp_sigma, yi) for yi in y])
    Vpp_sigma = cp.array([chebyshev_t(Cpp_sigma, yi) for yi in y])
    Vpp_pi = cp.array([chebyshev_t(Cpp_pi, yi) for yi in y])

    #Vss_sigma -= Css_sigma[0] / 2
    #Vsp_sigma -= Csp_sigma[0] / 2
    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat * eV_per_hart

def porezag_hopping(dR):
    ang_per_bohr = 0.529177  # [Angstroms/Bohr radius]
    eV_per_hart = 27.2114  # [eV/Hartree]

    dR = dR / ang_per_bohr
    dRn = cp.linalg.norm(dR, axis=1)
    dRn[dRn == 0] = 1  # Avoid division by zero

    dRn = dR / dRn[:, None]

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = cp.linalg.norm(dR, axis=1)
    r = cp.clip(r, 1, 7)

    aa = 1.0  # [Bohr radii]
    b = 7.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    #Css_sigma = cp.array([-0.4663805, 0.3528951, -0.1402985, 0.0050519, 0.0269723, -0.0158810, 0.0036716, 0.0010301, -0.0015546, 0.0008601])
    #Csp_sigma = cp.array([0.3395418, -0.2250358, 0.0298224, 0.0653476, -0.0605786, 0.0298962, -0.0099609, 0.0020609, 0.0001264, -0.0003381])
    Cpp_sigma = cp.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
    Cpp_pi = cp.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855])

    #Vss_sigma = cp.array([chebyshev_t(Css_sigma, yi) for yi in y])
    #Vsp_sigma = cp.array([chebyshev_t(Csp_sigma, yi) for yi in y])
    Vpp_sigma = cp.array([chebyshev_t(Cpp_sigma, yi) for yi in y])
    Vpp_pi = cp.array([chebyshev_t(Cpp_pi, yi) for yi in y])

    #Vss_sigma -= Css_sigma[0] / 2
    #Vsp_sigma -= Csp_sigma[0] / 2
    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat * eV_per_hart

def nnhop(dR):
    dist = cp.linalg.norm(dR, axis=1) - 2
    return 1 - dist

def nnhop_intra(dR):
    return 2.7 * cp.ones(len(dR), dtype=cp.float64)

def nnhop_inter(dR):
    return 0.3 * cp.ones(len(dR), dtype=cp.float64)
