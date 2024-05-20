from scipy.spatial.distance import cdist
import numpy as np
from TETB_GRAPHENE.TB_parameters import *
#from TB_parameters import *
import matplotlib.pyplot as plt
import glob
import scipy.linalg as spla
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
                        "nn":4.4}

def gen_ham_ovrlp(atom_positions, layer_types, cell, kpoint, model_type):
    """
    builds a hamiltonian and overlap matrix using distance dependent tight binding parameters

    :params atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :params layer_types: (np.ndarray [Natoms,]) atom types expressed as integers

    :params cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :params kpoint: (np.ndarray [3,]) kpoint to build hamiltonian and overlap with

    :params model_type: (str) specify which tight binding model to use. Options: [popov, mk]

    :returns: tuple(np.ndarray [Norbs,Norbs], np.ndarray [Norbs,Norbs]) Hamiltonian, Overlap        
    """
    
    conversion = 1.0/.529177 #[bohr/angstrom] ASE is always in angstrom, while our package wants bohr
    lattice_vectors = np.asarray(cell)*conversion
    atomic_basis = np.asarray(atom_positions)*conversion
    kpoint = np.asarray(kpoint)/conversion

    layer_types = np.asarray(layer_types)
    layer_type_set = set(layer_types)

    natom = len(atomic_basis)
    diFull = []
    djFull = []
    extended_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, np.newaxis] * dx + lattice_vectors[1, np.newaxis] * dy)
            diFull += [dx] * natom
            djFull += [dy] * natom
    distances = cdist(atomic_basis, extended_coords)
    
    Ham = models_self_energy[model_type["interlayer"]]*np.eye(natom,dtype=np.complex64)
    Overlap = np.eye(natom,dtype=np.complex64)
    for i_int,i_type in enumerate(layer_type_set):
        for j_int,j_type in enumerate(layer_type_set):
            if i_type==j_type:
                hopping_model = models_functions_intralayer[model_type["intralayer"]]
                overlap_model = porezag_overlap
                cutoff = models_cutoff_intralayer[model_type["intralayer"]] * conversion
            else:
                hopping_model = models_functions_interlayer[model_type["interlayer"]]
                
                overlap_model = popov_overlap
                cutoff = models_cutoff_interlayer[model_type["interlayer"]] * conversion
            i, j = np.where((distances > 0.1)  & (distances < cutoff))
            di = np.array(diFull)[j]
            dj = np.array(djFull)[j]
            i  = np.array(i)
            j  = np.array(j % natom)
            valid_indices = layer_types[i] == i_type
            valid_indices &= layer_types[j] == j_type
            valid_indices &= i!=j

            disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di[valid_indices], dj[valid_indices],
                                           i[valid_indices], j[valid_indices])
            phases = np.exp((1.0j)*np.dot(kpoint,disp.T))

            hoppings = hopping_model(lattice_vectors, atomic_basis,i[valid_indices], 
                                  j[valid_indices], di[valid_indices], dj[valid_indices])/2  # Divide by 2 since we are double counting every pair
            overlap_elem = overlap_model(disp)/2
            Ham[i[valid_indices],j[valid_indices]] += hoppings * phases
            Ham[j[valid_indices],i[valid_indices]] += np.conj(hoppings*phases)
            Overlap[i[valid_indices],j[valid_indices]] +=   overlap_elem  * phases
            Overlap[j[valid_indices],i[valid_indices]] +=  np.conj(overlap_elem * phases) 

    return Ham, Overlap

def get_hellman_feynman(atomic_basis, layer_types, lattice_vectors, eigvals,eigvec, model_type,kpoint):
    """Calculate Hellman-feynman forces for a given system. Uses finite differences to calculate matrix elements derivatives 
    
    :params atomic_basis: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :params layer_types: (np.ndarray [Natoms,]) atom types expressed as integers

    :params lattice_vectors: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :params eigvals: (np.ndarray [natoms,]) band structure eigenvalues of system

    :params eigvec: (np.ndarray [natoms,natoms]) eigenvectors of system

    :params model_type: (str) specify which tight binding model to use. Options: [popov, mk]

    :params kpoint: (np.ndarray [3,]) kpoint to build hamiltonian and overlap with

    :returns: (np.ndarray [natoms,3]) tight binding forces on each atom"""
    #get hellman_feynman forces at single kpoint. 
    #dE/dR_i =  - Tr_i(rho_e *dS/dR_i + rho * dH/dR_i)
    #construct density matrix
    natoms = len(layer_types)
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = np.array(lattice_vectors)*conversion
    atomic_basis = atomic_basis*conversion
    nocc = natoms//2
    fd_dist = 2*np.eye(natoms)
    fd_dist[nocc:,nocc:] = 0
    occ_eigvals = 2*np.diag(eigvals)
    occ_eigvals[nocc:,nocc:] = 0
    density_matrix =  eigvec @ fd_dist  @ np.conj(eigvec).T
    energy_density_matrix = eigvec @ occ_eigvals @ np.conj(eigvec).T
    tot_eng = 2 * np.sum(eigvals[:nocc])

    Forces = np.zeros((natoms,3))
    layer_type_set = set(layer_types)

    diFull = []
    djFull = []
    extended_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, np.newaxis] * dx + lattice_vectors[1, np.newaxis] * dy)
            diFull += [dx] * natoms
            djFull += [dy] * natoms
    distances = cdist(atomic_basis, extended_coords)

    gradH = np.zeros((len(diFull),natoms,3))
    for i_int,i_type in enumerate(layer_type_set):
        for j_int,j_type in enumerate(layer_type_set):

            if i_type==j_type:
                hopping_model_grad = porezag_hopping_grad
                overlap_model_grad = porezag_overlap_grad
                cutoff = models_cutoff_intralayer[model_type["intralayer"]] * conversion
                hopping_model = porezag_hopping
                overlap_model = porezag_overlap
            else:
                hopping_model_grad = popov_hopping_grad
                overlap_model_grad = popov_overlap_grad
                hopping_model = popov_hopping
                overlap_model = popov_overlap
                cutoff = models_cutoff_interlayer[model_type["interlayer"]] * conversion

            indi, indj = np.where((distances > 0.1) & (distances < cutoff))
            di = np.array(diFull)[indj]
            dj = np.array(djFull)[indj]
            i  = np.array(indi)
            j  = np.array(indj % natoms)
            valid_indices = layer_types[i] == i_type
            valid_indices &= layer_types[j] == j_type
            disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di[valid_indices], dj[valid_indices],
                                           i[valid_indices], j[valid_indices])
            phases = np.exp((1.0j)*np.dot(kpoint,disp.T))

            #check gradients of hoppings via finite difference
            grad_hop = np.zeros_like(disp)
            grad_overlap = np.zeros_like(disp)

            delta = 1e-5
            for dir_ind in range(3):
                dr = np.zeros(3)
                dr[dir_ind] +=  delta
                hop_up = hopping_model(disp+dr[np.newaxis,:])
                hop_dwn = hopping_model(disp-dr[np.newaxis,:])
                grad_hop[:,dir_ind] = (hop_up - hop_dwn)/2/delta

                overlap_up = overlap_model(disp+dr[np.newaxis,:])
                overlap_dwn = overlap_model(disp-dr[np.newaxis,:])

                grad_overlap[:,dir_ind] = (overlap_up - overlap_dwn)/2/delta

            rho =  density_matrix[i[valid_indices],j[valid_indices]][:,np.newaxis] 
            energy_rho = energy_density_matrix[i[valid_indices],j[valid_indices]][:,np.newaxis]
            gradH = grad_hop * phases[:,np.newaxis] * rho
            gradH += np.conj(gradH)
            Pulay =  grad_overlap * phases[:,np.newaxis] * energy_rho
            Pulay += np.conj(Pulay)

            for atom_ind in range(natoms):
                use_ind = np.squeeze(np.where(i[valid_indices]==atom_ind))
                ave_gradH = gradH[use_ind,:]
                ave_gradS = Pulay[use_ind,:] 
                if ave_gradH.ndim!=2:
                    Forces[atom_ind,:] -= -ave_gradH.real 
                    Forces[atom_ind,:] -=   ave_gradS.real
                else:
                    Forces[atom_ind,:] -= -np.sum(ave_gradH,axis=0).real 
                    Forces[atom_ind,:] -=   np.sum(ave_gradS,axis=0).real
    return Forces * conversion

def get_hellman_feynman_fd(atom_positions, layer_types, cell, eigvec, model_type,kpoint):
    """Calculate Hellman-feynman forces for a given system. Uses finite differences to calculate matrix elements derivatives 
    
    :params atomic_basis: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :params layer_types: (np.ndarray [Natoms,]) atom types expressed as integers

    :params lattice_vectors: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :params eigvec: (np.ndarray [natoms,natoms]) eigenvectors of system

    :params model_type: (str) specify which tight binding model to use. Options: [popov, mk]

    :params kpoint: (np.ndarray [3,]) kpoint to build hamiltonian and overlap with

    :returns: (np.ndarray [natoms,3]) tight binding forces on each atom"""
    dr = 1e-3
    natoms, _ = atom_positions.shape
    nocc = natoms // 2
    Forces = np.zeros((natoms, 3), dtype=np.float64)
    for dir_ind in range(3):
        for i in range(natoms):
            atom_positions_pert = np.copy(atom_positions)
            atom_positions_pert[i, dir_ind] += dr
            Ham,Overlap = gen_ham_ovrlp(atom_positions_pert, layer_types, cell, kpoint, model_type)
            eigvalues, eigvectors = spla.eigh(Ham,b=Overlap)
            Energy_up = 2 * np.sum(eigvalues[:nocc])
            
            atom_positions_pert = np.copy(atom_positions)
            atom_positions_pert[i, dir_ind] -= dr
            Ham,Overlap = gen_ham_ovrlp(atom_positions_pert, layer_types, cell, kpoint, model_type)
            eigvalues, eigvectors = spla.eigh(Ham,b=Overlap)
            Energy_dwn = 2 * np.sum(eigvalues[:nocc])

            Forces[i, dir_ind] = -(Energy_up - Energy_dwn) / (2 * dr)

    return Forces

if __name__=="__main__":
    import ase.io
    from ase.lattice.hexagonal import Graphite
    from ase import Atoms
    def get_atom_pairs(n,a):
        L=n*a+10
        sym=""
        pos=np.zeros((int(2*n),3))
        mol_id = np.zeros(int(2*n))
        for i in range(n):
            sym+="BTi"
            pos[i,:] = np.array([0,0,0])
            pos[i+n,:] = np.array([0,0,(i+1)*a])
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
    kpoint = np.array([0,0,0])
    params_str = "mk" #"popov"
    Ham,i,j, di, dj, phase = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoint, params_str)
    eigvals,eigvec = np.linalg.eigh(Ham)
    hf_forces = get_hellman_feynman(atom_positions, mol_id, cell, eigvec, params_str, i,j, di, dj, phase)
    print(hf_forces)
