use_cupy=False
use_autograd=False
if use_cupy:
    #import autograd.cupy as lp  # Thinly-wrapped numpy
    from autograd import grad
    #from cupyx.scipy.spatial.distance import cdist
elif use_autograd:
    import autograd.numpy as lp  # Thinly-wrapped numpy
    from autograd import grad
    from scipy.spatial.distance import cdist
else:
    import numpy as lp
    from scipy.spatial.distance import cdist
import numpy as np
from numba import njit
import numba
#import TEGT_GPU.descriptors as descriptors
import descriptors

@njit
def exponential(x, a, b):
    return a * lp.exp(-b * (x - 6.33))
@njit
def moon(r, a, b, c): 
    """
    Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012)
    """
    d, dz = r 
    return a * lp.exp(-b * (d - 2.68))*(1 - (dz/d)**2) + c * lp.exp(-b * (d - 6.33)) * (dz/d)**2
@njit
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
@njit
def chebyshev_t(c, x):
    #if c.dtype.char in '?bBhHiIlLqQpP':
    #    c = c.astype(lp.double)
    #if isinstance(x, (tuple, list)):
    #    x = lp.asarray(x)
    #if isinstance(x, lp.ndarray):
    #    c = c.reshape(c.shape + (1,)*x.ndim)

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

@njit
def norm(a):
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in np.arange(a.shape[0]):
        sum=0
        for j in np.arange(a.shape[1]):
            sum += a[i,j]*a[i,j]
        norms[i] = np.sqrt(sum)
    return norms

@njit
def popov_hopping(dR):
    #dRn = lp.linalg.norm(dR, axis=1)
    dRn = norm(dR)
    dRn = dR / dRn[:,lp.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = norm(dR)
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
    return valmat*eV_per_hart
@njit
def popov_hopping_grad(dR):
    #dRn = lp.linalg.norm(dR, axis=1)
    dRn = norm(dR)
    dRn = dR / dRn[:,lp.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = norm(dR)
    r = lp.clip(r, 1, 10)

    aa = 1.0  # [Bohr radii]
    b = 10.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)
    dr = 1e-3

    lpp_sigma = lp.array([0.1727212, -0.0937225, -0.0445544, 0.1114266, -0.0978079, 0.0577363, -0.0262833, 0.0094388, -0.0024695, 0.0003863])
    lpp_pi = lp.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478, -0.0535682, 0.0181983, -0.0046855, 0.0007303, 0.0000225, -0.0000393])

    Vpp_sigma = lp.array([chebyshev_t(lpp_sigma, yi) for yi in y])
    Vpp_pi = lp.array([chebyshev_t(lpp_pi, yi) for yi in y])

    d_Vpp_sigma = lp.array([(chebyshev_t(lpp_sigma, yi+dr)-chebyshev_t(lpp_sigma, yi-dr))/2/dr for yi in y])
    d_Vpp_pi = lp.array([(chebyshev_t(lpp_pi, yi+dr)-chebyshev_t(lpp_pi, yi-dr))/2/dr for yi in y])

    Vpp_sigma -= lpp_sigma[0] / 2
    Vpp_pi -= lpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz

    x_ = dR[:,0]
    y_ = dR[:,1]
    z_ = dR[:,2]
    gradt = np.zeros_like(dR)
    gradt[:,0] = (-2 * x_ * lp.power(n,2) * Vpp_sigma + lp.power(n,2) * d_Vpp_sigma \
            + 2 * x_ * lp.power(n,2) * Vpp_pi + (1 - lp.power(n,2)) * d_Vpp_pi)*dRn[:,0]
    
    gradt[:,1] = (-2 * y_ * lp.power(n,2) * Vpp_sigma + lp.power(n,2) * d_Vpp_sigma \
            + 2 * y_ * lp.power(n,2) * Vpp_pi + (1 - lp.power(n,2)) * d_Vpp_pi)*dRn[:,1]
    
    gradt[:,2] = (-2 * z_ * lp.power(n,2) * Vpp_sigma + lp.power(n,2) * d_Vpp_sigma \
            + 2 * z_ * lp.power(n,2) * Vpp_pi + (1 - lp.power(n,2)) * d_Vpp_pi)*dRn[:,2]
    
    return gradt * eV_per_hart

@njit
def porezag_hopping(dR):

    #dRn = lp.linalg.norm(dR, axis=1)
    dRn = norm(dR)
    dRn = dR / dRn[:,lp.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = norm(dR)
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

    return valmat*eV_per_hart
#@njit
def porezag_hopping_grad(dR):
    #dRn = lp.linalg.norm(dR, axis=1)
    dRn = norm(dR)
    dRn = dR / dRn[:,lp.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = norm(dR)
    r = lp.clip(r, 1, 10)

    aa = 1.0  # [Bohr radii]
    b = 7.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)
    dr = 1e-3

    lpp_sigma = lp.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
    lpp_pi = lp.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855])

    Vpp_sigma = lp.array([chebyshev_t(lpp_sigma, yi) for yi in y])
    Vpp_pi = lp.array([chebyshev_t(lpp_pi, yi) for yi in y])

    d_Vpp_sigma = lp.array([(chebyshev_t(lpp_sigma, yi+dr)-chebyshev_t(lpp_sigma, yi-dr))/2/dr for yi in y])
    d_Vpp_pi = lp.array([(chebyshev_t(lpp_pi, yi+dr)-chebyshev_t(lpp_pi, yi-dr))/2/dr for yi in y])

    Vpp_sigma -= lpp_sigma[0] / 2
    Vpp_pi -= lpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz

    x_ = dR[:,0]
    y_ = dR[:,1]
    z_ = dR[:,2]
    gradt = np.zeros_like(dR)
    gradt[:,0] = (-2 * x_ * lp.power(n,2) * Vpp_sigma + lp.power(n,2) * d_Vpp_sigma \
            + 2 * x_ * lp.power(n,2) * Vpp_pi + (1 - lp.power(n,2)) * d_Vpp_pi)*dRn[:,0]
    
    gradt[:,1] = (-2 * y_ * lp.power(n,2) * Vpp_sigma + lp.power(n,2) * d_Vpp_sigma \
            + 2 * y_ * lp.power(n,2) * Vpp_pi + (1 - lp.power(n,2)) * d_Vpp_pi)*dRn[:,1]
    
    gradt[:,2] = (-2 * z_ * lp.power(n,2) * Vpp_sigma + lp.power(n,2) * d_Vpp_sigma \
            + 2 * z_ * lp.power(n,2) * Vpp_pi + (1 - lp.power(n,2)) * d_Vpp_pi)*dRn[:,2]
    
    return gradt * eV_per_hart

#@njit
def popov(lattice_vectors, atomic_basis, i, j, di, dj):
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
    #lattice_vectors = lp.array(lattice_vectors)
    #atomic_basis = lp.array(atomic_basis)
    #i = lp.array(i)
    #j = lp.array(j)
    #di = lp.array(di)
    #dj = lp.array(dj)
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = popov_hopping(disp)
                
    return hoppings
#@njit
def popov_grad(lattice_vectors, atomic_basis, i, j, di, dj):
    #lattice_vectors = lp.array(lattice_vectors)
    #tomic_basis = lp.array(atomic_basis)
    #i = lp.array(i)
    #j = lp.array(j)
    #di = lp.array(di)
    #dj = lp.array(dj)
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hopping_grad = popov_hopping_grad(disp)
                
    return hopping_grad
#@njit
def porezag(lattice_vectors, atomic_basis, i, j, di, dj):
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
    #lattice_vectors = lp.array(lattice_vectors)
    #atomic_basis = lp.array(atomic_basis)
    #i = lp.array(i)
    #j = lp.array(j)
    #di = lp.array(di)
    #dj = lp.array(dj)
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = porezag_hopping(disp)
    return hoppings
#@njit
def porezag_grad(lattice_vectors, atomic_basis, i, j, di, dj):
    #lattice_vectors = lp.array(lattice_vectors)
    #atomic_basis = lp.array(atomic_basis)
    #i = lp.array(i)
    #j = lp.array(j)
    #di = lp.array(di)
    #dj = lp.array(dj)
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hopping_grad = porezag_hopping_grad(disp)
                
    return hopping_grad
@njit
def mk(lattice_vectors, atomic_basis, i, j, di, dj):
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
@njit
def nn_hop(lattice_vectors, atomic_basis, i, j, di, dj):
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
@njit
def letb_interlayer(lattice_vectors, atomic_basis, i, j, di, dj):
    return None
@njit
def letb_intralayer(lattice_vectors, atomic_basis, i, j, di, dj):
    return None
