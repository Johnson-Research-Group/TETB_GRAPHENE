from scipy.spatial.distance import cdist
import numpy as np
#from numba import njit
import matplotlib.pyplot as plt
import TETB_GRAPHENE.descriptors as descriptors
#import descriptors


#########################################################################################

# UTILS

########################################################################################
#@njit
def exponential(x, a, b):
    return a * np.exp(-b * (x - 6.33))
#@njit
def moon(r, a, b, c): 
    """
    Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012)
    """
    d, dz = r 
    return a * np.exp(-b * (d - 2.68))*(1 - (dz/d)**2) + c * np.exp(-b * (d - 6.33)) * (dz/d)**2
#@njit
def fang(rvec, a0, b0, c0, a3, b3, c3, a6, b6, c6, d6):
    """
    Parameterization from Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)
    """
    r, theta12, theta21 = rvec
    r = r / 4.649 

    def v0(x, a, b, c): 
        return a * np.exp(-b * x ** 2) * np.cos(c * x)

    def v3(x, a, b, c): 
        return a * (x ** 2) * np.exp(-b * (x - c) ** 2)  

    def v6(x, a, b, c, d): 
        return a * np.exp(-b * (x - c)**2) * np.sin(d * x)

    f =  v0(r, a0, b0, c0) 
    f += v3(r, a3, b3, c3) * (np.cos(3 * theta12) + np.cos(3 * theta21))
    f += v6(r, a6, b6, c6, d6) * (np.cos(6 * theta12) + np.cos(6 * theta21))
    return f
#@njit
def chebyshev_t(c, x):
    """chebyshev polynomials
     
    :params c: (np.ndarray) coefficients
     
    :params x: (np.ndarray) inputs
    
    :returns: (np.ndarray) """

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

#@njit
def norm(a):
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in np.arange(a.shape[0]):
        sum=0
        for j in np.arange(a.shape[1]):
            sum += a[i,j]*a[i,j]
        norms[i] = np.sqrt(sum)
    return norms

###############################################################################

# POPOV

###############################################################################
#@njit
def popov_hopping(dR):
    """pairwise Slater Koster Interlayer hopping parameters for pz orbitals of carbon as parameterized by Popov, Van Alsenoy in
     "Low-frequency phonons of few-layer graphene within a tight-binding model". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian matrix elements [eV]
    """
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 10)
    aa = 1.0  # [Bohr radii]
    b = 10.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Cpp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,
                           -0.0978079, 0.0577363, -0.0262833, 0.0094388,
                             -0.0024695, 0.0003863])
    Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,
                        -0.0535682, 0.0181983, -0.0046855, 0.0007303,
                          0.0000225, -0.0000393])
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 


    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat*eV_per_hart

def popov_overlap(dR):
    """pairwise Slater Koster Interlayer overlap parameters for pz orbitals of carbon as parameterized by Popov, Van Alsenoy in
     "Low-frequency phonons of few-layer graphene within a tight-binding model". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Overlap matrix elements [eV]
    """
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 10)
  
    #%boundaries for polynomial

    aa = 1 #; %Angstrom
    b = 10 #; %Angstrom
    y = (2*r-(b+aa))/(b-aa)
    #orignally sigma
    Cpp_sigma=np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997,
                        0.0921727, -0.0268106, 0.0002240, 0.0040319,
                        -0.0022450, 0.0005596])
    Cpp_pi=   np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,
                        0.0156376, 0.0025976, -0.0039498, 0.0020581,
                        -0.0007114, 0.0001427])

    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    Vpp_sigma  -= Cpp_sigma[0]/2
    Vpp_pi -= Cpp_pi[0]/2
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi  #; %Changing only this as only
    return Ezz #*eV_per_hart

#@njit
def popov_hopping_grad(dR):
    """pairwise Slater Koster Interlayer hopping gradient parameters for pz orbitals of carbon as parameterized by Popov, Van Alsenoy in
     "Low-frequency phonons of few-layer graphene within a tight-binding model". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian Gradient matrix elements [eV/bohr]
    """
    dRn = norm(dR)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    #r =norm(dR)
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 10)
    aa = 1.0  # [Bohr radii]
    b = 10.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)
    dr = 1e-3

    Cpp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,
                           -0.0978079, 0.0577363, -0.0262833, 0.0094388,
                             -0.0024695, 0.0003863])
    Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,
                        -0.0535682, 0.0181983, -0.0046855, 0.0007303,
                          0.0000225, -0.0000393])
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    d_Vpp_sigma = np.array([(chebyshev_t(Cpp_sigma, yi+dr)-chebyshev_t(Cpp_sigma, yi-dr))/2/dr for yi in y])
    d_Vpp_pi = np.array([(chebyshev_t(Cpp_pi, yi+dr)-chebyshev_t(Cpp_pi, yi-dr))/2/dr for yi in y])

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz

    xsq = np.power(dR[:,0],2)
    ysq = np.power(dR[:,1],2)
    zsq = np.power(dR[:,2],2)
    
    gradt = np.zeros_like(dR)
    rsq = np.power(r,2)
    nsq = np.power(n,2)
    gradt[:,0] = (-2 * Vpp_sigma * nsq/r  \
                + d_Vpp_sigma * nsq \
                + 2 * Vpp_pi * nsq /r \
                + d_Vpp_pi * ((xsq**2+ysq**2 + xsq*ysq)/rsq**2 + nsq * (xsq + ysq)/rsq)) * l
    
    gradt[:,1] = (-2 * Vpp_sigma * nsq /r \
                + d_Vpp_sigma * nsq \
                + 2 * Vpp_pi * nsq / r\
                + d_Vpp_pi * ((xsq**2+ysq**2 + xsq*ysq)/rsq**2 + nsq * (xsq + ysq)/rsq)) * m
                
    gradt[:,2] = (2 * Vpp_sigma * n * (xsq + ysq)/rsq \
                + d_Vpp_sigma * nsq \
                - 2 * Vpp_pi * n * (xsq + ysq)/rsq \
                + d_Vpp_pi * (xsq**2 + ysq**2)/rsq**2) * n
    #gradients verified against finite difference
    return gradt * eV_per_hart * 2/(b-aa)

def popov_overlap_grad(dR):
    """pairwise Slater Koster Interlayer overlap gradient parameters for pz orbitals of carbon as parameterized by Popov, Van Alsenoy in
     "Low-frequency phonons of few-layer graphene within a tight-binding model". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) overlap gradient matrix elements [eV/bohr]
    """
    dRn = norm(dR)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    #r =norm(dR)
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 10)
    aa = 1.0  # [Bohr radii]
    b = 10.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)
    dr = 1e-3

    Cpp_sigma=np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997,
                        0.0921727, -0.0268106, 0.0002240, 0.0040319,
                        -0.0022450, 0.0005596])
    Cpp_pi=   np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,
                        0.0156376, 0.0025976, -0.0039498, 0.0020581,
                        -0.0007114, 0.0001427])
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    d_Vpp_sigma = np.array([(chebyshev_t(Cpp_sigma, yi+dr)-chebyshev_t(Cpp_sigma, yi-dr))/2/dr for yi in y])
    d_Vpp_pi = np.array([(chebyshev_t(Cpp_pi, yi+dr)-chebyshev_t(Cpp_pi, yi-dr))/2/dr for yi in y])

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi

    xsq = np.power(dR[:,0],2)
    ysq = np.power(dR[:,1],2)
    zsq = np.power(dR[:,2],2)
    
    grads = np.zeros_like(dR)
    rsq = np.power(r,2)
    nsq = np.power(n,2)
    grads[:,0] = (-2 * Vpp_sigma * nsq/r  \
                + d_Vpp_sigma * nsq \
                + 2 * Vpp_pi * nsq /r \
                + d_Vpp_pi * ((xsq**2+ysq**2 + xsq*ysq)/rsq**2 + nsq * (xsq + ysq)/rsq)) * l
    
    grads[:,1] = (-2 * Vpp_sigma * nsq /r \
                + d_Vpp_sigma * nsq \
                + 2 * Vpp_pi * nsq / r\
                + d_Vpp_pi * ((xsq**2+ysq**2 + xsq*ysq)/rsq**2 + nsq * (xsq + ysq)/rsq)) * m
                
    grads[:,2] = (2 * Vpp_sigma * n * (xsq + ysq)/rsq \
                + d_Vpp_sigma * nsq \
                - 2 * Vpp_pi * n * (xsq + ysq)/rsq \
                + d_Vpp_pi * (xsq**2 + ysq**2)/rsq**2) * n
    
    return grads * 2/(b-aa)

#@njit
def popov(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    popov parameter wrapper
    """
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = popov_hopping(disp)
                
    return hoppings
#@njit
def popov_grad(lattice_vectors, atomic_basis, i, j, di, dj):
    """popov grad parameter wrapper """
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hopping_grad = popov_hopping_grad(disp)
                
    return hopping_grad

####################################################################################################

# POREZAG

####################################################################################################
#@njit
def porezag_hopping(dR):
    """pairwise Slater Koster hopping parameters for pz orbitals of carbon as parameterized by Porezag in
     "Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian matrix elements [eV]
    """
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 7)

    aa = 1.0  # [Bohr radii]
    b = 7.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Cpp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,
                           -0.0673216, 0.0316900, -0.0117293, 0.0033519, 
                           -0.0004838, -0.0000906])
    Cpp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, 
                       -0.0300733, 0.0074465, -0.0008563, -0.0004453, 
                       0.0003842, -0.0001855])
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat*eV_per_hart

def porezag_overlap(dR):
    """pairwise Slater Koster overlap parameters for pz orbitals of carbon as parameterized by Porezag in
     "Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Overlap matrix elements [eV]
    """
    eV_per_hart=27.2114
    dRn = np.linalg.norm(dR, axis=1)
    #dRn = norm(dR)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    #r = norm(dR)
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 7)
    aa = 1 #; %Angstrom

    b =7 #; %Angstrom

    y = (2*r-(b+aa))/(b-aa)

    
    #overlap matrix coefficient (No units mentioned)
    #originally sigma
    Cpp_sigma=np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,
                            0.0753818, -0.0108677, -0.0075444, 0.0051533,
                            -0.0013747, 0.0000751])
    Cpp_pi=   np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,
                            0.0061645, 0.0051460, -0.0032776, 0.0009119,
                            -0.0001265, -0.000227])
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    Vpp_sigma  -= Cpp_sigma[0]/2
    Vpp_pi -= Cpp_pi[0]/2
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi
    return Ezz #*eV_per_hart

def porezag_hopping_grad(dR):
    """pairwise Slater Koster hopping gradient parameters for pz orbitals of carbon as parameterized by Porezag in
     "Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian gradient matrix elements [eV/bohr]
    """
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 7)

    aa = 1.0  # [Bohr radii]
    b = 7.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)
    dr = 1e-3

    Cpp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
    Cpp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855])

    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    d_Vpp_sigma = np.array([(chebyshev_t(Cpp_sigma, yi+dr)-chebyshev_t(Cpp_sigma, yi-dr))/2/dr for yi in y])
    d_Vpp_pi = np.array([(chebyshev_t(Cpp_pi, yi+dr)-chebyshev_t(Cpp_pi, yi-dr))/2/dr for yi in y])

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi

    xsq = np.power(dR[:,0],2)
    ysq = np.power(dR[:,1],2)
    zsq = np.power(dR[:,2],2)
    
    gradt = np.zeros_like(dR)
    rsq = np.power(r,2)
    nsq = np.power(n,2)
    gradt[:,0] = (-2 * Vpp_sigma * nsq/r  \
                + d_Vpp_sigma * nsq \
                + 2 * Vpp_pi * nsq /r \
                + d_Vpp_pi * ((xsq**2+ysq**2 + xsq*ysq)/rsq**2 + nsq * (xsq + ysq)/rsq)) * l
    
    gradt[:,1] = (-2 * Vpp_sigma * nsq /r \
                + d_Vpp_sigma * nsq \
                + 2 * Vpp_pi * nsq / r\
                + d_Vpp_pi * ((xsq**2+ysq**2 + xsq*ysq)/rsq**2 + nsq * (xsq + ysq)/rsq)) * m
                
    gradt[:,2] = (2 * Vpp_sigma * n * (xsq + ysq)/rsq \
                + d_Vpp_sigma * nsq \
                - 2 * Vpp_pi * n * (xsq + ysq)/rsq \
                + d_Vpp_pi * (xsq**2 + ysq**2)/rsq**2) * n
    #gradients verified against finite difference
    return gradt * eV_per_hart * 2/(b-aa)

def porezag_overlap_grad(dR):
    """pairwise Slater Koster overlap gradient parameters for pz orbitals of carbon as parameterized by Porezag in
     "Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Overlap gradient matrix elements [eV/bohr]
    """
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 7)

    aa = 1.0  # [Bohr radii]
    b = 7.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)
    dr = 1e-3

    Cpp_sigma=np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,
                            0.0753818, -0.0108677, -0.0075444, 0.0051533,
                            -0.0013747, 0.0000751])
    Cpp_pi=   np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,
                            0.0061645, 0.0051460, -0.0032776, 0.0009119,
                            -0.0001265, -0.000227])
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    d_Vpp_sigma = np.array([(chebyshev_t(Cpp_sigma, yi+dr)-chebyshev_t(Cpp_sigma, yi-dr))/2/dr for yi in y])
    d_Vpp_pi = np.array([(chebyshev_t(Cpp_pi, yi+dr)-chebyshev_t(Cpp_pi, yi-dr))/2/dr for yi in y])

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi

    xsq = np.power(dR[:,0],2)
    ysq = np.power(dR[:,1],2)
    zsq = np.power(dR[:,2],2)
    
    grads = np.zeros_like(dR)
    rsq = np.power(r,2)
    nsq = np.power(n,2)
    grads[:,0] = (-2 * Vpp_sigma * nsq/r  \
                + d_Vpp_sigma * nsq \
                + 2 * Vpp_pi * nsq /r \
                + d_Vpp_pi * ((xsq**2+ysq**2 + xsq*ysq)/rsq**2 + nsq * (xsq + ysq)/rsq)) * l
    
    grads[:,1] = (-2 * Vpp_sigma * nsq /r \
                + d_Vpp_sigma * nsq \
                + 2 * Vpp_pi * nsq / r\
                + d_Vpp_pi * ((xsq**2+ysq**2 + xsq*ysq)/rsq**2 + nsq * (xsq + ysq)/rsq)) * m
                
    grads[:,2] = (2 * Vpp_sigma * n * (xsq + ysq)/rsq \
                + d_Vpp_sigma * nsq \
                - 2 * Vpp_pi * n * (xsq + ysq)/rsq \
                + d_Vpp_pi * (xsq**2 + ysq**2)/rsq**2) * n
    return grads * 2/(b-aa)
#@njit
def porezag(lattice_vectors, atomic_basis, i, j, di, dj):
    """
    porezag parameter wrapper
    """
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = porezag_hopping(disp)
    return hoppings
#@njit
def porezag_grad(lattice_vectors, atomic_basis, i, j, di, dj):
    """porezag gradient parameter wrapper """
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hopping_grad = porezag_hopping_grad(disp)
                
    return hopping_grad

############################################################################################

# Extras

#############################################################################################
#@njit
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
    lattice_vectors = np.array(lattice_vectors)
    atomic_basis = np.array(atomic_basis)
    i = np.array(i)
    j = np.array(j)
    di = np.array(di)
    dj = np.array(dj)
    dxy, dz = descriptors.ix_to_dist(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = moon([np.sqrt(dz**2 + dxy**2), dz], -2.7, 1.17, 0.48)
    return hoppings
#@njit
def nn_hop(lattice_vectors, atomic_basis, i, j, di, dj) :
    """nearest neighbor hopping. atomic basis in bohr. hoppings in eV"""
    conversion = 1.0/.529177 #[bohr/angstrom]
    lattice_vectors = np.array(lattice_vectors)
    atomic_basis = np.array(atomic_basis)
    i = np.array(i)
    j = np.array(j)
    di = np.array(di)
    dj = np.array(dj)
    disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    dist = np.linalg.norm(disp,axis=1)
    nn_dist = 1.42*conversion
    nn_layer_sep = 3.35*conversion
    slope = (2.7-0.3)/(nn_dist - nn_layer_sep)
    inter = 2.7/slope/nn_dist
    hoppings = slope * dist + inter
    return hoppings
#@njit
def letb_interlayer(lattice_vectors, atomic_basis, i, j, di, dj):
    return None
#@njit
def letb_intralayer(lattice_vectors, atomic_basis, i, j, di, dj):
    return None

if __name__=="__main__":
    import matplotlib.pyplot as plt
    n = 15
    r = np.linspace(1.01,6.99,n)
    hop = np.zeros(n)
    for i in range(n):
        hop[i] = porezag_hopping(np.array([[r[i],0,0]]))
    plt.plot(r,hop,label="porezag")

    for i in range(n):
        hop[i] = popov_hopping(np.array([[0,0,r[i]]]))
    plt.plot(r,hop,label="popov")
    plt.legend()
    plt.savefig("hoppings.png")

    plt.clf()

    bond_int_pi = np.zeros(n)
    bond_int_sigma = np.zeros(n)
    #popov
    Cpp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266, -0.0978079, 0.0577363, -0.0262833, 0.0094388, -0.0024695, 0.0003863])
    Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478, -0.0535682, 0.0181983, -0.0046855, 0.0007303, 0.0000225, -0.0000393])
    #porezag
    Cpp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
    Cpp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855])

    
    #Vpp_sigma = np.array([np.polynomial.chebyshev.chebval(yi,Cpp_sigma) for yi in r])
    #Vpp_pi = np.array([np.polynomial.chebyshev.chebval(yi,Cpp_pi) for yi in r])
    aa = 1 #; %Angstrom
    b = 10 #; %Angstrom
    y = (2*r-(b+aa))/(b-aa)
    Vpp_sigma = np.polynomial.chebyshev.chebval(y,Cpp_sigma)
    Vpp_pi = np.polynomial.chebyshev.chebval(y,Cpp_pi)
    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2

    plt.plot(r,Vpp_sigma,label="porezag sigma")
    plt.plot(r,Vpp_pi,label="porezag pi")
    plt.legend()
    plt.savefig("bond_ints_porezag.png")

    plt.clf()

    #overlap
    #popov
    model = "popov"
    Cpp_sigma=np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997,
                        0.0921727, -0.0268106, 0.0002240, 0.0040319,
                        -0.0022450, 0.0005596])
    Cpp_pi=   np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,
                        0.0156376, 0.0025976, -0.0039498, 0.0020581,
                        -0.0007114, 0.0001427])
    #porezag
    model = "porezag"
    Cpp_sigma=np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,
                            0.0753818, -0.0108677, -0.0075444, 0.0051533,
                            -0.0013747, 0.0000751])
    Cpp_pi=   np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,
                            0.0061645, 0.0051460, -0.0032776, 0.0009119,
                            -0.0001265, -0.000227])
    aa = 1 #; %Angstrom
    b = 7 #; %Angstrom
    y = (2*r-(b+aa))/(b-aa)
    n = r
    Vpp_sigma = np.polynomial.chebyshev.chebval(y,Cpp_sigma)
    Vpp_pi = np.polynomial.chebyshev.chebval(y,Cpp_pi)
    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    elem = n**2*Vpp_sigma + (1-n**2)*Vpp_pi

    plt.plot(r,Vpp_sigma,label=" sigma overlap")
    plt.plot(r,Vpp_pi,label=" pi overlap")
    #plt.plot(r,elem,label="overlap matrix element")
    plt.title(model+" overlap")
    plt.legend()
    plt.savefig("overlap_ints_"+model+".png")

    plt.clf()

    X,Y = np.meshgrid(r,r)
    X = X.flatten()
    Y = Y.flatten()
    n2 = np.power(X,2) / (np.power(X,2) + np.power(Y,2))
    plt.scatter(X,Y,color = n2 + (1-n2))
    plt.savefig("directional_cosines.png")
    plt.clf()
