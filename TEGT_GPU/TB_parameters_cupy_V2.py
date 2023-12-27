use_cupy=False
if use_cupy:
    #import autograd.cupy as lp  # Thinly-wrapped numpy
    from autograd import grad
    #from cupyx.scipy.spatial.distance import cdist
else:
    import autograd.numpy as lp  # Thinly-wrapped numpy
    from autograd import grad
    from scipy.spatial.distance import cdist

import numpy as np
#import TEGT_GPU.descriptors as descriptors
import descriptors
def exponential(x, a, b):
    return a * lp.exp(-b * (x - 6.33))

def moon(r, a, b, c): 
    """
    Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012)
    """
    d, dz = r 
    return a * lp.exp(-b * (d - 2.68))*(1 - (dz/d)**2) + c * lp.exp(-b * (d - 6.33)) * (dz/d)**2

def fang(rvec, a0, b0, c0, a3, b3, c3, a6, b6, c6, d6):
    """
    Parameterization from Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)
    """
    r, theta12, theta21 = rvec
    r = r / 4.649 

    def v0(x, a, b, c): 
        return a * lp.exp(-b * x ** 2) * lp.cos(c * x)

    def v3(x, a, b, c): 
        return a * (x ** 2) * lp.exp(-b * (x - c) ** 2)  

    def v6(x, a, b, c, d): 
        return a * lp.exp(-b * (x - c)**2) * lp.sin(d * x)

    f =  v0(r, a0, b0, c0) 
    f += v3(r, a3, b3, c3) * (lp.cos(3 * theta12) + lp.cos(3 * theta21))
    f += v6(r, a6, b6, c6, d6) * (lp.cos(6 * theta12) + lp.cos(6 * theta21))
    return f

def chebyshev_t(c, x):
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(lp.double)
    if isinstance(x, (tuple, list)):
        x = lp.asarray(x)
    if isinstance(x, lp.ndarray):
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
    dRn = lp.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,lp.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = lp.linalg.norm(dR, axis=1)
    r = lp.clip(r, 1, 10)

    aa = 1.0  # [Bohr radii]
    b = 10.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    lpp_sigma = lp.array([0.1727212, -0.0937225, -0.0445544, 0.1114266, -0.0978079, 0.0577363, -0.0262833, 0.0094388, -0.0024695, 0.0003863])
    lpp_pi = lp.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478, -0.0535682, 0.0181983, -0.0046855, 0.0007303, 0.0000225, -0.0000393])

    Vpp_sigma = lp.array([chebyshev_t(lpp_sigma, yi) for yi in y])
    Vpp_pi = lp.array([chebyshev_t(lpp_pi, yi) for yi in y])

    Vpp_sigma -= lpp_sigma[0] / 2
    Vpp_pi -= lpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    #return valmat*eV_per_hart
    return lp.linalg.norm(dR, axis=1)

def porezag_hopping(dR):
    dRn = lp.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,lp.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = lp.linalg.norm(dR, axis=1)
    r = lp.clip(r, 1, 7)

    aa = 1.0  # [Bohr radii]
    b = 7.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    lpp_sigma = lp.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
    lpp_pi = lp.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855])

    Vpp_sigma = lp.array([chebyshev_t(lpp_sigma, yi) for yi in y])
    Vpp_pi = lp.array([chebyshev_t(lpp_pi, yi) for yi in y])
    Vpp_sigma -= lpp_sigma[0] / 2
    Vpp_pi -= lpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz

    #return valmat*eV_per_hart
    return lp.linalg.norm(dR, axis=1)

def popov(lattice_vectors, atomic_basis, i, j, di, dj,layer_types=None):
    """
    Moon model for bilayer graphene - Moon and Koshino, PRB 85 (2012)
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for intralayer in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    lattice_vectors = lp.array(lattice_vectors)
    atomic_basis = lp.array(atomic_basis)
    i = lp.array(i)
    j = lp.array(j)
    di = lp.array(di)
    dj = lp.array(dj)
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = lp.zeros_like(i,dtype=lp.float64)
    ntypes = [1,2]
    for i_type in ntypes:
        for j_type in ntypes:
            valid_indices = layer_types[i] == i_type
            valid_indices &= layer_types[j] == j_type
            if i_type==j_type:
                hoppings[valid_indices] = porezag_hopping(disp[valid_indices])
            else:
                hop = popov_hopping(disp[valid_indices])
                hoppings[valid_indices] = hop
    return hoppings

def mk(lattice_vectors, atomic_basis, i, j, di, dj,layer_types=None):
    """
    Moon model for bilayer graphene - Moon and Koshino, PRB 85 (2012)
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for intralayer in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    lattice_vectors = lp.array(lattice_vectors)
    atomic_basis = lp.array(atomic_basis)
    i = lp.array(i)
    j = lp.array(j)
    di = lp.array(di)
    dj = lp.array(dj)
    dxy, dz = descriptors.ix_to_dist(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = moon([lp.sqrt(dz**2 + dxy**2), dz], -2.7, 1.17, 0.48)
    return hoppings

def nn_hop(lattice_vectors, atomic_basis, i, j, di, dj,layer_types=None):
    lattice_vectors = lp.array(lattice_vectors)
    atomic_basis = lp.array(atomic_basis)
    i = lp.array(i)
    j = lp.array(j)
    di = lp.array(di)
    dj = lp.array(dj)
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    dist = lp.linalg.norm(disp,axis=1)
    nn_dist = 1.42
    hoppings = -(dist-nn_dist)+2
    return hoppings

def letb(lattice_vectors, atomic_basis, i, j, di, dj, layer_types=None):
    return None
