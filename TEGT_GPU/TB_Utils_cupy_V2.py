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

models_functions_interlayer = {'letb':letb_interlayer,
                                    'mk':mk,
                                    'popov':popov,
                                    "nn":nn_hop}
models_cutoff_interlayer={'letb':10,
                        'mk':10,
                        'popov':5.29,
                        "nn":3}
models_self_energy = {'letb':0,
                    'mk':0,
                    'popov':-5.2887,
                    "nn":0}
models_functions_intralayer = {'letb':letb_intralayer,
                                'mk':mk,
                                'porezag':porezag,
                                "nn":nn_hop}
models_cutoff_intralayer={'letb':10,
                        'mk':10,
                        'porezag':3.7,
                        "nn":3}

"""def compute_hoppings(lattice_vectors, atomic_basis, hopping_model,kpoint,layer_types=None):
    
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
    return Ham, i,j, di, dj, phases"""

def gen_ham_ovrlp(atom_positions, layer_types, cell, kpoint, model_type):
    """
    Returns a pythtb model object for a given ASE atomic configuration 
    Input:
        ase_atoms - ASE object for the periodic system
        model_type - 'letb' or 'mk'
    Output:
        gra - PythTB model describing hoppings between atoms using model_type        
    """
    
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = lp.asarray(cell)*conversion
    atomic_basis = lp.asarray(atom_positions)*conversion
    kpoint = lp.asarray(kpoint)/conversion

    layer_types = lp.asarray(layer_types)
    layer_type_set = set(layer_types)

    natom = len(atomic_basis)
    diFull = []
    djFull = []
    extended_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, lp.newaxis] * dx + lattice_vectors[1, lp.newaxis] * dy)
            diFull += [dx] * natom
            djFull += [dy] * natom
    distances = cdist(atomic_basis, extended_coords)
    
    Ham = models_self_energy[model_type["interlayer"]]*lp.eye(natom,dtype=lp.complex64)
    for i_int,i_type in enumerate(layer_type_set):
        for j_int,j_type in enumerate(layer_type_set):
            if i_type==j_type:
                hopping_model = models_functions_intralayer[model_type["intralayer"]]
                cutoff = models_cutoff_intralayer[model_type["intralayer"]] * conversion
            else:
                hopping_model = models_functions_interlayer[model_type["interlayer"]]
                cutoff = models_cutoff_interlayer[model_type["interlayer"]] * conversion

            indi, indj = lp.where((distances > 0.1) & (distances < cutoff))
            di = lp.array(diFull)[indj]
            dj = lp.array(djFull)[indj]
            i  = lp.array(indi)
            j  = lp.array(indj % natom)
            valid_indices = layer_types[i] == i_type
            valid_indices &= layer_types[j] == j_type
            ind_R_ = lp.stack((di[valid_indices],dj[valid_indices],lp.zeros_like(di[valid_indices])),axis=1)@lattice_vectors.T
            rv = -atomic_basis[i[valid_indices],:]+atomic_basis[j[valid_indices],:]+ind_R_
            phases = lp.exp((1.0j)*lp.dot(kpoint,rv.T))

            hoppings = hopping_model(lattice_vectors, atomic_basis,i[valid_indices], 
                                  j[valid_indices], di[valid_indices], dj[valid_indices])/2  # Divide by 2 since we are double counting every pair
            Ham[i[valid_indices],j[valid_indices]] += hoppings * phases
            Ham[j[valid_indices],i[valid_indices]] += lp.conj(hoppings*phases)

    return Ham

def get_helem(lattice_vectors, hopping_model,indi, indj, di, dj):
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = lp.array(lattice_vectors)*conversion
    def fxn(pos):
        pos = pos*conversion
        hopping = hopping_model(lattice_vectors, pos,indi, indj, di, dj)
        return hopping
    return fxn

def get_hellman_feynman(atomic_basis, layer_types, lattice_vectors, eigvec, model_type,kpoint):
    #construct density matrix
    natoms = len(layer_types)
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    nocc = natoms//2
    fd_dist = 2*lp.eye(natoms)
    fd_dist[nocc:,nocc:] = 0
    density_matrix =  lp.conj(eigvec) @ fd_dist  @ eigvec.T

    Forces = lp.zeros((natoms,3))
    layer_type_set = set(layer_types)

    diFull = []
    djFull = []
    extended_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, lp.newaxis] * dx + lattice_vectors[1, lp.newaxis] * dy)
            diFull += [dx] * natoms
            djFull += [dy] * natoms
    distances = cdist(atomic_basis, extended_coords)

    gradH = lp.zeros((len(diFull),natoms,3))
    for i_int,i_type in enumerate(layer_type_set):
        for j_int,j_type in enumerate(layer_type_set):

            if i_type==j_type:
                hopping_model = models_functions_intralayer[model_type["intralayer"]]
                cutoff = models_cutoff_intralayer[model_type["intralayer"]] * conversion
            else:
                hopping_model = models_functions_interlayer[model_type["interlayer"]]
                cutoff = models_cutoff_interlayer[model_type["interlayer"]] * conversion

            indi, indj = lp.where((distances > 0.1) & (distances < cutoff))
            di = lp.array(diFull)[indj]
            dj = lp.array(djFull)[indj]
            i  = lp.array(indi)
            j  = lp.array(indj % natoms)
            valid_indices = layer_types[i] == i_type
            valid_indices &= layer_types[j] == j_type
            ind_R_ = lp.stack((di[valid_indices],dj[valid_indices],lp.zeros_like(di[valid_indices])),axis=1)@lattice_vectors.T
            rv = -atomic_basis[i[valid_indices],:]+atomic_basis[j[valid_indices],:]+ind_R_
            phases = lp.exp((1.0j)*lp.dot(kpoint,rv.T))
    
            helem_fxn = get_helem(lattice_vectors, hopping_model,i[valid_indices], 
                                  j[valid_indices], di[valid_indices], dj[valid_indices])
            gradH_fxn = jacobian(helem_fxn)
            gradH_tmp = gradH_fxn(atomic_basis)
            rho =  density_matrix[i[valid_indices],j[valid_indices]][:,lp.newaxis,lp.newaxis]
            Forces += -lp.sum(lp.nan_to_num(gradH_tmp)*phases[:,lp.newaxis,lp.newaxis]*rho,axis=0).real
            #gradH[valid_indices,:,:] += lp.nan_to_num(gradH_tmp)*phases[:,lp.newaxis,lp.newaxis]
    
    #Forces = -lp.sum(gradH   ,axis=0).real #* phases[:,lp.newaxis,lp.newaxis]
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
            Ham = gen_ham_ovrlp(atom_positions_pert, layer_types, cell, kpoint, model_type)
            eigvalues, eigvectors = lp.linalg.eigh(Ham)
            Energy_up = 2 * lp.sum(eigvalues[:nocc])
            
            atom_positions_pert = lp.copy(atom_positions)
            atom_positions_pert[i, dir_ind] -= dr
            Ham = gen_ham_ovrlp(atom_positions_pert, layer_types, cell, kpoint, model_type)
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