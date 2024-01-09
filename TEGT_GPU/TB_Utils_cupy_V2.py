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
import glob
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
    Returns a pythtb model object for a given ASE atomic configuration 
    Input:
        ase_atoms - ASE object for the periodic system
        model_type - 'letb' or 'mk'
    Output:
        gra - PythTB model describing hoppings between atoms using model_type        
    """
    
    conversion = 1.0/.529177 #[bohr/angstrom] ASE is always in angstrom, while our package wants bohr
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
    Overlap = np.eye(natom,dtype=lp.complex64)
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
            i, j = lp.where((distances > 0.1)  & (distances < cutoff))
            di = lp.array(diFull)[j]
            dj = lp.array(djFull)[j]
            i  = lp.array(i)
            j  = lp.array(j % natom)
            valid_indices = layer_types[i] == i_type
            valid_indices &= layer_types[j] == j_type
            valid_indices &= i!=j

            disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di[valid_indices], dj[valid_indices],
                                           i[valid_indices], j[valid_indices])
            phases = lp.exp((1.0j)*lp.dot(kpoint,disp.T))

            hoppings = hopping_model(lattice_vectors, atomic_basis,i[valid_indices], 
                                  j[valid_indices], di[valid_indices], dj[valid_indices])/2  # Divide by 2 since we are double counting every pair
            overlap_elem = overlap_model(disp)/2
            Ham[i[valid_indices],j[valid_indices]] += hoppings * phases
            Ham[j[valid_indices],i[valid_indices]] += lp.conj(hoppings*phases)
            Overlap[i[valid_indices],j[valid_indices]] +=   overlap_elem  * phases
            Overlap[j[valid_indices],i[valid_indices]] +=  lp.conj(overlap_elem * phases) 
    
    """mean_z = np.mean(atom_positions[:,2])
    top_ind = np.where(atom_positions[:,2]>mean_z)
    bot_ind = np.where(atom_positions[:,2]<mean_z)
    layer_sep = np.mean(atom_positions[top_ind,2]-atom_positions[bot_ind,2])
    ham_files = glob.glob("latte_hamiltonians/ham_*",recursive=True)
    smin=100
    s=""
    for hf in ham_files:
        sstr = hf.split("_")[-1]
        if np.abs(layer_sep-float(sstr))<smin:
            s = sstr
            smin = np.abs(layer_sep-float(sstr))

    print(s)
    latte_data = np.loadtxt("latte_hamiltonians/ham_"+s)
    natoms = 200
    latte_Ham = -5.29*np.eye(natoms)
    indexi=np.int64(latte_data[:,1])-1
    indexj = np.int64(latte_data[:,2])-1
    latte_Ham[indexi,indexj] += latte_data[:,0]
    diff = Ham.real-latte_Ham

    #plt.imshow(Ham.real-latte_Ham)
    #plt.colorbar()
    #plt.savefig("latte_python_ham_diff"+s+".png")
    #plt.clf()

    latte_data = np.loadtxt("latte_overlap/overlap_"+s)
    natoms = 200
    latte_overlap = np.eye(natoms)
    indexi=np.int64(latte_data[:,1])-1
    indexj = np.int64(latte_data[:,2])-1
    #latte_overlap[indexi,indexj] = latte_data[:,0]
    for o in range(len(latte_data[:,0])):
        latte_overlap[indexi[o],indexj[o]] = latte_data[o,0]

    plt.imshow(Overlap.real-latte_overlap)
    plt.colorbar()
    plt.savefig("latte_python_overlap_diff"+s+".png")
    plt.clf()"""

    return Ham, Overlap

def gen_ham_ovrlp_ref(atom_positions, layer_types, cell, kpoint, model_type):
    """
    Returns a pythtb model object for a given ASE atomic configuration 
    Input:
        ase_atoms - ASE object for the periodic system
        model_type - 'letb' or 'mk'
    Output:
        gra - PythTB model describing hoppings between atoms using model_type        
    """
    
    conversion = 1.0/.529177 #[bohr/angstrom] ASE is always in angstrom, while our package wants bohr
    lattice_vectors = lp.asarray(cell)*conversion
    atomic_basis = lp.asarray(atom_positions)*conversion
    kpoint = lp.asarray(kpoint)/conversion

    layer_types = lp.asarray(layer_types)
    layer_type_set = set(layer_types)

    natom = len(atomic_basis)
    
    #Ham = models_self_energy[model_type["interlayer"]]*lp.eye(natom,dtype=lp.complex64)
    Ham = lp.zeros((natom,natom),dtype=lp.complex64)
    Overlap = lp.eye(natom,dtype = lp.complex64)
    for indi in range(natom):
        for indj in range(natom):
            if indj<=indi:
                continue

            if layer_types[indi]==layer_types[indj]:
                hopping_model = models_functions_intralayer[model_type["intralayer"]]
                overlap_model = porezag_overlap
                cutoff = models_cutoff_intralayer[model_type["intralayer"]] * conversion
            else:
                hopping_model = models_functions_interlayer[model_type["interlayer"]]
                overlap_model = popov_overlap
                cutoff = models_cutoff_interlayer[model_type["interlayer"]] * conversion

            min_dist = 100
            n=0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1,0,1]:
                        ind_R_ = dx*lattice_vectors[0,:] + dy*lattice_vectors[1,:] + dz*lattice_vectors[2,:]
                        rv = -atomic_basis[indi,:]+atomic_basis[indj,:]+ind_R_
                        if np.linalg.norm(rv)<min_dist:
                            disp = rv
                            min_dist = np.linalg.norm(rv)
            if np.linalg.norm(disp)>cutoff:
                continue
            phases = lp.exp((1.0j)*lp.dot(kpoint,disp))
            hoppings = np.squeeze(hopping_model([disp])) * phases  
            overlap_elem = np.squeeze(overlap_model([disp])) * phases
            Overlap[indi,indj] =   overlap_elem 
            Overlap[indj,indi] =  lp.conj(overlap_elem) 
            Ham[indi,indj] = hoppings 
            Ham[indj,indi] = lp.conj(hoppings)
    print(np.round(Ham.real,decimals=2))
    return Ham,Overlap

def get_helem(lattice_vectors, hopping_model,indi, indj, di, dj):
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = lp.array(lattice_vectors)*conversion
    def fxn(pos):
        pos = pos*conversion
        hopping = hopping_model(lattice_vectors, pos,indi, indj, di, dj)
        return hopping
    return fxn

def get_hellman_feynman_autograd(atomic_basis, layer_types, lattice_vectors, eigvec, model_type,kpoint):
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
            disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di[valid_indices], dj[valid_indices],
                                           i[valid_indices], j[valid_indices])
            phases = lp.exp((1.0j)*lp.dot(kpoint,disp.T))
    
            helem_fxn = get_helem(lattice_vectors, hopping_model,i[valid_indices], 
                                  j[valid_indices], di[valid_indices], dj[valid_indices])
            gradH_fxn = jacobian(helem_fxn)
            gradH_tmp = gradH_fxn(atomic_basis)
            rho =  density_matrix[i[valid_indices],j[valid_indices]][:,lp.newaxis,lp.newaxis]
            Forces += -lp.sum(lp.nan_to_num(gradH_tmp)*phases[:,lp.newaxis,lp.newaxis]*rho,axis=0).real

    return Forces

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
                #if model_type["intralayer"] != "porezag":
                #    print("analytical gradients only implemented for porezag parameters, use autograd instead")
                #    exit()
                hopping_model_grad = porezag_grad
                cutoff = models_cutoff_intralayer[model_type["intralayer"]] * conversion
                hopping_model = models_functions_intralayer[model_type["intralayer"]]
            else:
                #if model_type["intralayer"] != "popov":
                #    print("analytical gradients only implemented for popov parameters, use autograd instead")
                    #exit()
                hopping_model_grad = popov_grad
                hopping_model = models_functions_interlayer[model_type["interlayer"]]
                cutoff = models_cutoff_interlayer[model_type["interlayer"]] * conversion

            indi, indj = lp.where((distances > 0.1) & (distances < cutoff))
            di = lp.array(diFull)[indj]
            dj = lp.array(djFull)[indj]
            i  = lp.array(indi)
            j  = lp.array(indj % natoms)
            valid_indices = layer_types[i] == i_type
            valid_indices &= layer_types[j] == j_type
            disp = descriptors.ix_to_disp(lattice_vectors, atomic_basis, di[valid_indices], dj[valid_indices],
                                           i[valid_indices], j[valid_indices])
            phases = lp.exp((1.0j)*lp.dot(kpoint,disp.T))
    
            #helem_fxn = get_helem(lattice_vectors, hopping_model,i[valid_indices], 
            #                      j[valid_indices], di[valid_indices], dj[valid_indices])
            #gradH_fxn = jacobian(helem_fxn)
            #gradH_tmp = gradH_fxn(atomic_basis)
            #dH_autograd = np.zeros((natoms,natoms))
            #for index in range(natoms):
            #dH_autograd[i[valid_indices],j[valid_indices]] += gradH_tmp[:,1,0]
            #print(np.round(dH_autograd,decimals=2),"\n")
            grad_hop = get_grad_hop(atomic_basis,lattice_vectors,hopping_model_grad,i[valid_indices], 
                                  j[valid_indices], di[valid_indices], dj[valid_indices])
            rho =  density_matrix[i[valid_indices],j[valid_indices]][:,lp.newaxis] #,lp.newaxis]
            dH_analytical = np.zeros((natoms,natoms))
            dH_analytical[i[valid_indices],j[valid_indices]] = grad_hop[:,0]
            #print(np.round(dH_analytical,2))
            gradH = grad_hop * phases[:,lp.newaxis] * rho
            for atom_ind in range(natoms):
                use_ind = np.squeeze(np.where(i[valid_indices]==atom_ind))
                ave_gradH = gradH[use_ind,:]
                if ave_gradH.ndim!=2:
                    Forces[atom_ind,:] += 2*ave_gradH.real
                else:
                    Forces[atom_ind,:] += lp.sum(2*ave_gradH,axis=0).real
    return Forces

#@njit
def get_grad_hop(pos,lattice_vectors,hopping_model_grad,ind_i,ind_j,di,dj):
    conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = lp.array(lattice_vectors)*conversion
    pos = pos*conversion
    natoms = np.shape(pos)[0]
    #gradH = np.zeros((len(i),natoms,3))
    #for i in range(natoms):
    #    #get gradients in hoppings for all neighbors of atom i
    #    sub_ind = np.where(ind_i==i)
    #    gradH[i,:] = hopping_model_grad(lattice_vectors, pos, ind_i[sub_ind], ind_j[sub_ind], di[sub_ind], dj[sub_ind])
    gradH = hopping_model_grad(lattice_vectors, pos, ind_i, ind_j, di, dj)
    return gradH

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