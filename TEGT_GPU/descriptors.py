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
import h5py
import pandas as pd
import numpy as np
from numba import njit

#@njit
def nnmat(lattice_vectors, atomic_basis):
    """
    Build matrix which tells you relative coordinates
    of nearest neighbors to an atom i in the supercell

    Returns: nnmat [natom x 3 x 3]
    """
    nnmat = lp.zeros((len(atomic_basis), 3, 3))

    # Extend atom list
    atoms = []
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            displaced_atoms = atomic_basis + lattice_vectors[lp.newaxis, 0] * i + lattice_vectors[lp.newaxis, 1] * j
            atoms += [list(x) for x in displaced_atoms]
    atoms = lp.array(atoms)
    atomic_basis = lp.array(atomic_basis)

    # Loop
    for i in range(len(atomic_basis)):
        displacements = atoms - atomic_basis[i]
        distances = lp.linalg.norm(displacements,axis=1)
        ind = lp.argsort(distances)
        nnmat[i] = displacements[ind[1:4]]

    return nnmat
#@njit
def ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """ 
    Converts displacement indices to physical distances
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    dxy - Distance in Bohr, projected in the x/y plane
    dz  - Distance in Bohr, projected onto the z axis
    """

    displacement_vector = di[:, lp.newaxis] * lattice_vectors[0] +\
                          dj[:, lp.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]

    displacement_vector_xy = displacement_vector[:, :2] 
    displacement_vector_z =  displacement_vector[:, -1] 

    dxy = lp.linalg.norm(displacement_vector_xy, axis = 1)
    dz = lp.abs(displacement_vector_z)
    return dxy, dz
#@njit
def ix_to_disp(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """ 
    Converts displacement indices to physical distances
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    dxy - Distance in Bohr, projected in the x/y plane
    dz  - Distance in Bohr, projected onto the z axis
    """
    displacement_vector = di[:, lp.newaxis] * lattice_vectors[0] +\
                          dj[:, lp.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]
    return displacement_vector
#@njit
def partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Given displacement indices and geometry,
    get indices for partitioning the data
    """
    # First find the smallest distance in the lattice -> reference for NN 
    distances = ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj)
    distances = lp.sqrt(distances[0]**2 + distances[1]**2)
    min_distance = min(distances)

    # NN should be within 5% of min_distance
    t01_ix = (distances >= 0.95 * min_distance) & (distances <= 1.05 * min_distance)

    # NNN should be withing 5% of sqrt(3)x of min_distance
    t02_ix = (distances >= 0.95 * lp.sqrt(3) * min_distance) & (distances <= 1.05 * lp.sqrt(3) * min_distance)

    # NNNN should be within 5% of 2x of min_distance
    t03_ix = (distances >= 0.95 * 2 * min_distance) & (distances <= 1.05 * 2 * min_distance)
   
    # Anything else, we zero out
    t00 = (distances < 0.95 * min_distance) | (distances > 1.05 * 2 * min_distance)

    return t01_ix, t02_ix, t03_ix, t00
@njit
def triangle_height(a, base):
    """
    Give area of a triangle given two displacement vectors for 2 sides
    """
    area = lp.linalg.det(
            lp.array([a, base, [1, 1, 1]])
    )
    area = lp.abs(area)/2
    height = 2 * area / lp.linalg.norm(base)
    return height
#@njit
def t01_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    # Compute NN distances
    r = di[:, lp.newaxis] * lattice_vectors[0] + dj[:, lp.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai] # Relative coordinates
    #r[:, -1] = 0 # Project into xy-plane
    a = lp.linalg.norm(r, axis = 1)
    return pd.DataFrame({'a': a})
#@njit
def t02_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    # Compute NNN distances
    r = di[:, lp.newaxis] * lattice_vectors[0] + dj[:, lp.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai] # Relative coordinates
    #r[:, -1] = 0 # Project into xy-plane
    b = lp.linalg.norm(r, axis = 1)

    # Compute h
    h1 = []
    h2 = []
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in range(len(r)):
        nn = mat[aj[i]] + r[i]
        nn[:, -1] = 0 # Project into xy-plane
        nndist = lp.linalg.norm(nn, axis = 1)
        ind = lp.argsort(nndist)
        h1.append(triangle_height(nn[ind[0]], r[i]))
        h2.append(triangle_height(nn[ind[1]], r[i]))
    return pd.DataFrame({'h1': h1, 'h2': h2, 'b': b})
#@njit
def t03_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Compute t03 descriptors
    """
    # Compute NNNN distances
    r = di[:, lp.newaxis] * lattice_vectors[0] + dj[:, lp.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai] # Relative coordinates
    c = lp.linalg.norm(r, axis = 1)
    #r[:, -1] = 0 # Project into xy-plane

    # All other hexagon descriptors
    l = []
    h = []
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in range(len(r)):
        nn = mat[aj[i]] + r[i]
        nn[:, -1] = 0 # Project into xy-plane
        nndist = lp.linalg.norm(nn, axis = 1)
        ind = lp.argsort(nndist)
        b = nndist[ind[0]]
        d = nndist[ind[1]]
        h3 = triangle_height(nn[ind[0]], r[i])
        h4 = triangle_height(nn[ind[1]], r[i])

        nn = r[i] - mat[ai[i]]
        nn[:, -1] = 0 # Project into xy-plane
        nndist = lp.linalg.norm(nn, axis = 1)
        ind = lp.argsort(nndist)
        a = nndist[ind[0]]
        e = nndist[ind[1]]
        h1 = triangle_height(nn[ind[0]], r[i])
        h2 = triangle_height(nn[ind[1]], r[i])

        l.append((a + b + d + e)/4)
        h.append((h1 + h2 + h3 + h4)/4)
    return pd.DataFrame({'c': c, 'h': h, 'l': l})
#@njit
def descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """ 
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    # Partition 
    partition = partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj)
    
    # Compute descriptors
    t01 = t01_descriptors(lattice_vectors, atomic_basis, di[partition[0]], dj[partition[0]], ai[partition[0]], aj[partition[0]])
    t02 = t02_descriptors(lattice_vectors, atomic_basis, di[partition[1]], dj[partition[1]], ai[partition[1]], aj[partition[1]])
    t03 = t03_descriptors(lattice_vectors, atomic_basis, di[partition[2]], dj[partition[2]], ai[partition[2]], aj[partition[2]])
    return t01, t02, t03
#@njit
def ix_to_orientation(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Converts displacement indices to orientations of the 
    nearest neighbor environments using definitions in 
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    theta_12 - Orientation of upper-layer relative to bond length
    theta_21 - Orientation of lower-layer relative to bond length
    """
    displacement_vector = di[:, lp.newaxis] * lattice_vectors[0] +\
                          dj[:, lp.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]
    mat = nnmat(lattice_vectors, atomic_basis)

    # Compute distances and angles
    theta_12 = []
    theta_21 = []
    for disp, i, j, inn, jnn in zip(displacement_vector, ai, aj, mat[ai], mat[aj]):
        sin_jnn = lp.cross(jnn[:,:2], disp[:2]) 
        sin_inn = lp.cross(inn[:,:2], disp[:2]) 
        cos_jnn = lp.dot(jnn[:,:2], disp[:2]) 
        cos_inn = lp.dot(inn[:,:2], disp[:2]) 
        theta_jnn = lp.arctan2(sin_jnn, cos_jnn)
        theta_inn = lp.arctan2(sin_inn, cos_inn)

        theta_12.append(lp.pi - theta_jnn[0])
        theta_21.append(theta_inn[0])
    return theta_12, theta_21
#@njit
def descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    
    output = {
        'dxy': [], # Distance in Bohr, xy plane
        'dz': [],  # Distance in Bohr, z
        'd': [],   # Distance in Bohr 
        'theta_12': [], # Orientation of upper layer NN environment
        'theta_21': [], # Orientation of lower layer NN environment
    }

    # 1-body terms
    dist_xy, dist_z = ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj)
    dist = lp.sqrt(dist_z ** 2 + dist_xy ** 2)
    output['dxy'] = list(dist_xy)
    output['dz'] = list(dist_z)
    output['d'] = list(dist)

    # Many-body terms
    theta_12, theta_21 = ix_to_orientation(lattice_vectors, atomic_basis, di, dj, ai, aj)
    output['theta_12'] += list(theta_12)
    output['theta_21'] += list(theta_21)
   
    # Return pandas DataFrame
    df = pd.DataFrame(output)
    return df
