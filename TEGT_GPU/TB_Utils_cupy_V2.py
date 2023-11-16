import numpy as np 
import cupy as cp
from cupyx.scipy.spatial.distance import cdist
from TEGT_GPU.TB_parameters_cupy import *

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
    indi, indj = cp.where((distances > 0) & (distances < 10)) # 10 Bohr cutoff
    di = cp.array(di)[indj]
    dj = cp.array(dj)[indj]
    i  = indi
    j  = indj % natom
    hoppings = hopping_model(lattice_vectors, atomic_basis,i, j, di, dj, layer_types=layer_types) / 2 # Divide by 2 since we are double counting every pair
    phase = cp.exp(1j * cp.dot(kpoint, extended_coords.T))
    Ham = cp.zeros((natom,natom))
    Ham_elem = hoppings*phase #multiply element wise
    Ham[i,j] = Ham_elem
    return Ham, i,j, di, dj, phase

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
    
    #optional stricter descriptor between layers, necesssary for high corrugation
    layer_types = cp.asarray(layer_types)
    Ham, i,j, di, dj, phase = compute_hoppings(lattice_vectors, atomic_basis, 
                                     models_functions[model_type],kpoint,layer_types=layer_types)
    
    return Ham, i,j, di, dj, phase

def get_hellman_feynman(atomic_basis, layer_types, lattice_vectors, eigvec, hopping_model, indi,indj, di, dj, phases):
    #construct density matrix
    natoms = len(layer_types)
    nocc = natoms//2
    density_matrix = cp.sum(cp.conj(eigvec[:, :nocc]) * eigvec[:, :nocc], axis=1)
    Forces = cp.zeros((natoms,3))
    dr = 1e-3
    for i in range(natoms):
        #find atom i neighbors indices
        neigh_ind_i =  indi[i]
        neigh_ind_j = indj[i]
        for j in range(3):
            #perturb atom i position in j direction
            perturb_pos = atomic_basis
            perturb_pos[i,j] +=dr
            #compute hoppings of its neighbors
            h_up = hopping_model(lattice_vectors, atomic_basis,neigh_ind_i, neigh_ind_j, di[neigh_ind_i], dj[neigh_ind_i], layer_types=layer_types) / 2 # this should be 1d array size(neigh_ind_i)
            #repeat in opposite direction
            perturb_pos = atomic_basis
            perturb_pos[i,j] -=dr
            #compute hoppings of its neighbors
            h_d = hopping_model(lattice_vectors, atomic_basis,neigh_ind_i, neigh_ind_j, di[neigh_ind_i], dj[neigh_ind_i], layer_types=layer_types) / 2
            #take finite difference
            H_diff = (h_up - h_d)/2/dr * phases[neigh_ind_j]
            
            #take expectation value using neighbor indices and density matrix
            ave_gradH = H_diff * density_matrix[neigh_ind_i,neigh_ind_j]
            Forces[i,j] = ave_gradH
    return Forces
