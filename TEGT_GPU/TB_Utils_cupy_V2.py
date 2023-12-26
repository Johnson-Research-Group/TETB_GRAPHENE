import numpy as np 
import cupy as cp
from cupyx.scipy.spatial.distance import cdist
#from TEGT_GPU.TB_parameters_cupy_V2 import *
from TB_parameters_cupy_V2 import *
import matplotlib.pyplot as plt
def compute_hoppings(lattice_vectors, atomic_basis, hopping_model,kpoint,layer_types=None):
    """
    Compute hoppings in a hexagonal environment of the computation cell 
    Adequate for large unit cells (> 100 atoms)
    Input:
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for graphene in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        layer_types     - int   (natoms) layer index of atom i
        hopping_model   - model for computing hoppings

    Output:
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """

    natom = len(atomic_basis)
    di = []
    dj = []
    extended_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, cp.newaxis] * dx + lattice_vectors[1, cp.newaxis] * dy)
            di += [dx] * natom
            dj += [dy] * natom
    distances = cdist(atomic_basis, extended_coords)
    indi, indj = cp.where((distances > 0.1) & (distances < 10)) # 10 Bohr cutoff
    di = cp.array(di)[indj]
    dj = cp.array(dj)[indj]
    i  = cp.array(indi)
    j  = cp.array(indj % natom)
    hoppings = hopping_model(lattice_vectors, atomic_basis,i, j, di, dj, layer_types=layer_types) / 2 # Divide by 2 since we are double counting every pair
    disp = di[:, cp.newaxis] * lattice_vectors[0] +\
                          dj[:, cp.newaxis] * lattice_vectors[1] +\
                          atomic_basis[j] - atomic_basis[i]
    kpoint = kpoint #@lattice_vectors.T
    #phase = cp.exp(1j * cp.dot(kpoint, disp.T))
    Ham = cp.zeros((natom,natom),dtype=cp.complex64)
    #Ham_elem = hoppings * phase #multiply element wise
    #Ham[i,j] = Ham_elem
    ind_R_ = cp.stack((di,dj,cp.zeros_like(di)),axis=1)@lattice_vectors.T
    rv = -atomic_basis[i,:]+atomic_basis[j,:]+ind_R_
    phases = cp.exp((1.0j)*cp.dot(kpoint,rv.T))
    #Ham[i,j] += hoppings * phases
    #Ham[j,i] += cp.conj(hoppings*phases)
    for index,hopping in enumerate(hoppings):
        ind_R= ind_R_[index,:]
        # vector from one site to another
        rv=-atomic_basis[i[index],:]+atomic_basis[j[index],:]+ind_R
        # Calculate the hopping, see details in info/tb/tb.pdf
        phase=cp.exp((1.0j)*cp.dot(kpoint,rv))
        amp=hopping*phase
        # add this hopping into a matrix and also its conjugate
        Ham[i[index],j[index]]+=amp
        Ham[j[index],i[index]]+=amp.conjugate()
    return Ham, i,j, di, dj, phases

def gen_ham_ovrlp(atom_positions, layer_types, cell, kpoint, model_type):
    """
    Returns a pythtb model object for a given ASE atomic configuration 
    Input:
        ase_atoms - ASE object for the periodic system
        model_type - 'letb' or 'mk'
    Output:
        gra - PythTB model describing hoppings between atoms using model_type        
    """
    models_functions = {'letb':letb,
                         'mk':mk,
                         'popov':popov}
    if model_type not in ['letb','mk','popov']:
        print("Invalid function {}".format(models_functions))
        return None
    
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = cp.asarray(cell)*conversion
    atomic_basis = cp.asarray(atom_positions)*conversion
    kpoint = cp.asarray(kpoint)/conversion
    
    #optional stricter descriptor between layers, necesssary for high corrugation
    layer_types = cp.asarray(layer_types)
    Ham, i,j, di, dj, phase = compute_hoppings(lattice_vectors, atomic_basis, 
                                     models_functions[model_type],kpoint,layer_types=layer_types)
    return Ham, i,j, di, dj, phase

def get_hellman_feynman(atomic_basis, layer_types, lattice_vectors, eigvec, model_type, indi,indj, di, dj, phases):
    #construct density matrix
    natoms = len(layer_types)
    nocc = natoms//2
    fd_dist = 2*cp.eye(natoms)
    fd_dist[nocc:,nocc:] = 0
    density_matrix = cp.conj(eigvec).T @ fd_dist @ eigvec
    ind_R_ = cp.stack((di,dj,cp.zeros_like(di)),axis=1)@lattice_vectors.T
    disp = -atomic_basis[indi,:]+atomic_basis[indj,:]+ind_R_ 
    Forces = cp.zeros((natoms,3))
    dr = 1e-3
    models_functions = {'letb':letb,
                         'mk':mk,
                         'popov':popov}
    hopping_model = models_functions[model_type]
    for i in range(natoms):
        #find atom i neighbors indices
        neigh_ind_i =  indi==i
        for j in range(3):
            #perturb atom i position in j direction
            perturb_pos = cp.copy(atomic_basis)
            perturb_pos[i,j] +=dr
            #compute hoppings of its neighbors
            h_up = hopping_model(lattice_vectors, perturb_pos,indi[neigh_ind_i], indj[neigh_ind_i], di[neigh_ind_i], dj[neigh_ind_i], layer_types=layer_types) # this should be 1d array size(neigh_ind_i)
            #repeat in opposite direction
            perturb_pos = cp.copy(atomic_basis)
            perturb_pos[i,j] -=dr
            #compute hoppings of its neighbors
            h_d = hopping_model(lattice_vectors, perturb_pos,indi[neigh_ind_i], indj[neigh_ind_i], di[neigh_ind_i], dj[neigh_ind_i], layer_types=layer_types)
            #take finite difference
            H_diff = (h_up - h_d)/2/dr * phases[indi[neigh_ind_i]]             
            #take expectation value using neighbor indices and density matrix
            ave_gradH = cp.sum(H_diff * density_matrix[indi[neigh_ind_i],indj[neigh_ind_i]]).real
            Forces[i,j] = ave_gradH
    return Forces

def get_hellman_feynman_fd(atom_positions, layer_types, cell, eigvec, model_type,kpoint):
    dr = 1e-3
    natoms, _ = atom_positions.shape
    nocc = natoms // 2
    Forces = cp.zeros((natoms, 3), dtype=cp.float64)
    for dir_ind in range(3):
        for i in range(natoms):
            atom_positions_pert = cp.copy(atom_positions)
            atom_positions_pert[i, dir_ind] += dr
            Ham,_ ,_, _, _, _ = gen_ham_ovrlp(atom_positions_pert, layer_types, cell, kpoint, model_type)
            eigvalues, eigvectors = cp.linalg.eigh(Ham)
            Energy_up = 2 * cp.sum(eigvalues[:nocc])
            
            atom_positions_pert = cp.copy(atom_positions)
            atom_positions_pert[i, dir_ind] -= dr
            Ham, _ ,_, _, _, _ = gen_ham_ovrlp(atom_positions_pert, layer_types, cell, kpoint, model_type)
            eigvalues, eigvectors = cp.linalg.eigh(Ham)
            Energy_dwn = 2 * cp.sum(eigvalues[:nocc])

            Forces[i, dir_ind] = -(Energy_up - Energy_dwn) / (2 * dr)

    return Forces
