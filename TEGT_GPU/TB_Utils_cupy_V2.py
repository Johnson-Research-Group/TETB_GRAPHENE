use_cupy=False
if use_cupy:
    #import autograd.cupy as lp  # Thinly-wrapped numpy
    from autograd import jacobian
    #from cupyx.scipy.spatial.distance import cdist
else:
    import autograd.numpy as lp  # Thinly-wrapped numpy
    from autograd import jacobian
    from scipy.spatial.distance import cdist

import numpy as np

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
            extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, lp.newaxis] * dx + lattice_vectors[1, lp.newaxis] * dy)
            di += [dx] * natom
            dj += [dy] * natom
    distances = cdist(atomic_basis, extended_coords)
    indi, indj = lp.where((distances > 0.1) & (distances < 10)) # 10 Bohr cutoff
    di = lp.array(di)[indj]
    dj = lp.array(dj)[indj]
    i  = lp.array(indi)
    j  = lp.array(indj % natom)
    hoppings = hopping_model(lattice_vectors, atomic_basis,i, j, di, dj, layer_types=layer_types) / 2 # Divide by 2 since we are double counting every pair
    disp = di[:, lp.newaxis] * lattice_vectors[0] +\
                          dj[:, lp.newaxis] * lattice_vectors[1] +\
                          atomic_basis[j] - atomic_basis[i]
    kpoint = kpoint #@lattice_vectors.T
    #phase = lp.exp(1j * lp.dot(kpoint, disp.T))
    Ham = lp.zeros((natom,natom),dtype=lp.complex64)
    #Ham_elem = hoppings * phase #multiply element wise
    #Ham[i,j] = Ham_elem
    ind_R_ = lp.stack((di,dj,lp.zeros_like(di)),axis=1)@lattice_vectors.T
    rv = -atomic_basis[i,:]+atomic_basis[j,:]+ind_R_
    phases = lp.exp((1.0j)*lp.dot(kpoint,rv.T))
    #Ham[i,j] += hoppings * phases
    #Ham[j,i] += lp.conj(hoppings*phases)
    for index,hopping in enumerate(hoppings):
        ind_R= ind_R_[index,:]
        # vector from one site to another
        rv=-atomic_basis[i[index],:]+atomic_basis[j[index],:]+ind_R
        # Calculate the hopping, see details in info/tb/tb.pdf
        phase=lp.exp((1.0j)*lp.dot(kpoint,rv))
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
                         'popov':popov,
                         "nn":nn_hop}
    if model_type not in ['letb','mk','popov',"nn"]:
        print("Invalid function {}".format(models_functions))
        return None
    
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = lp.asarray(cell)*conversion
    atomic_basis = lp.asarray(atom_positions)*conversion
    kpoint = lp.asarray(kpoint)/conversion
    
    #optional stricter descriptor between layers, necesssary for high corrugation
    layer_types = lp.asarray(layer_types)
    Ham, i,j, di, dj, phase = compute_hoppings(lattice_vectors, atomic_basis, 
                                     models_functions[model_type],kpoint,layer_types=layer_types)
    return Ham, i,j, di, dj, phase

def get_helem(lattice_vectors, hopping_model,indi, indj, di, dj, layer_types):
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = lp.array(lattice_vectors)*conversion
    def fxn(pos):
        pos = pos*conversion
        hopping = hopping_model(lattice_vectors, pos,indi, indj, di, dj, layer_types=layer_types)
        return hopping
    return fxn

def get_hellman_feynman(atomic_basis, layer_types, lattice_vectors, eigvec, model_type, indi,indj, di, dj, phases):
    #construct density matrix
    natoms = len(layer_types)
    nocc = natoms//2
    fd_dist = 2*lp.eye(natoms)
    fd_dist[nocc:,nocc:] = 0
    density_matrix =  (lp.conj(eigvec).T @ fd_dist  @ eigvec).T
    #ind_R_ = lp.stack((di,dj,lp.zeros_like(di)),axis=1)@lattice_vectors.T
    #disp = -atomic_basis[indi,:]+atomic_basis[indj,:]+ind_R_ 
    Forces = lp.zeros((natoms,3))
    models_functions = {'letb':letb,
                         'mk':mk,
                         'popov':popov,
                         "nn":nn_hop}
    hopping_model = models_functions[model_type]
    helem_fxn = get_helem(lattice_vectors, hopping_model,indi, indj, di, dj, layer_types)
    gradH_fxn = jacobian(helem_fxn)
    gradH = gradH_fxn((atomic_basis))
    gradH = lp.nan_to_num(gradH)
    #Forces = -lp.sum(gradH * density_matrix[indi,indj][:,lp.newaxis,lp.newaxis]  * phases[:,lp.newaxis,lp.newaxis],axis=0).real
    density_mat_ji = lp.sum(eigvec[:nocc,indj],axis=0)
    density_mat_ij = lp.sum(eigvec[:nocc,indi],axis=0)
    Forces = -lp.sum(lp.conj(density_mat_ji)[:,lp.newaxis,lp.newaxis] * gradH * phases[:,lp.newaxis,lp.newaxis]* density_mat_ij[:,lp.newaxis,lp.newaxis])
    return Forces

def get_hellman_feynman_fd(atom_positions, layer_types, cell, eigvec, model_type,kpoint):
    dr = 1e-4
    natoms, _ = atom_positions.shape
    nocc = natoms // 2
    Forces = lp.zeros((natoms, 3), dtype=lp.float64)
    for dir_ind in range(3):
        for i in range(natoms):
            atom_positions_pert = lp.copy(atom_positions)
            atom_positions_pert[i, dir_ind] += dr
            Ham,_ ,_, _, _, _ = gen_ham_ovrlp(atom_positions_pert, layer_types, cell, kpoint, model_type)
            eigvalues, eigvectors = lp.linalg.eigh(Ham)
            Energy_up = 2 * lp.sum(eigvalues[:nocc])
            
            atom_positions_pert = lp.copy(atom_positions)
            atom_positions_pert[i, dir_ind] -= dr
            Ham, _ ,_, _, _, _ = gen_ham_ovrlp(atom_positions_pert, layer_types, cell, kpoint, model_type)
            eigvalues, eigvectors = lp.linalg.eigh(Ham)
            Energy_dwn = 2 * lp.sum(eigvalues[:nocc])

            Forces[i, dir_ind] = -(Energy_up - Energy_dwn) / (2 * dr)

    return Forces

if __name__=="__main__":
    import ase.io
    from ase.lattice.hexagonal import Graphite
    from ase import Atoms
    def get_atom_pairs(n,a):
        L=n*a+10
        sym=""
        pos=lp.zeros((int(2*n),3))
        mol_id = lp.zeros(int(2*n))
        for i in range(n):
            sym+="BTi"
            pos[i,:] = lp.array([0,0,0])
            pos[i+n,:] = lp.array([0,0,(i+1)*a])
            mol_id[i] = 1
            mol_id[i+n]=2
        #'BBBBTiTiTiTi'(0,a,0),(a,2*a,0),(2*a,3*a,0),(3*a,4*a,0)
        atoms = Atoms(sym,positions=pos, #,(2*a,0,0),(a,a,0)],
                    cell=[L,L,L])
        atoms.set_array("mol-id",mol_id)
        return atoms

    n = 1
    a = 2.46
    atoms = get_atom_pairs(n,a)
    atom_positions = atoms.positions
    cell = atoms.get_cell()
    mol_id = atoms.get_array("mol-id")
    layer_types = atoms.get_chemical_symbols()
    kpoint = lp.array([0,0,0])
    params_str = "mk" #"popov"
    Ham,i,j, di, dj, phase = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoint, params_str)
    eigvals,eigvec = lp.linalg.eigh(Ham)
    hf_forces = get_hellman_feynman(atom_positions, mol_id, cell, eigvec, params_str, i,j, di, dj, phase)
    print(hf_forces)