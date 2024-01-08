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
import scipy.linalg as spla
#from TEGT_GPU.TB_Utils_cupy_V2 import *
from TB_Utils_cupy_V2 import *
#from TEGT_GPU.TB_parameters_cupy_V2 import *
from TB_parameters_cupy_V2 import *

def get_recip_cell(cell):
    a1 = cell[:, 0]
    a2 = cell[:, 1]
    a3 = cell[:, 2]

    volume = np.dot(a1, np.cross(a2, a3))

    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume

    return np.array([b1, b2, b3])


def get_tb_forces_energy(atom_positions,mol_id,cell,kpoints,params_str,rcut = 10):
    atom_positions = np.asarray(atom_positions)
    cell = np.asarray(cell)
    kpoints = np.asarray(kpoints)
    mol_id = np.asarray(mol_id)

    #params = get_param_dict(params_str)
    recip_cell = get_recip_cell(cell)
    
    if kpoints.shape == (3,):
        kpoints = kpoints.reshape((1, 3))
    
    kpoints = kpoints @ recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]

    Energy = 0
    Forces = np.zeros((natoms, 3), dtype=np.complex64)
    for k in range(nkp):
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        #Ham = np.linalg.inv(Overlap) @ Ham #@ Overlap
        
        #Ham = get_Energy_spectrum(atom_positions, mol_id, cell,kval = kpoints[k,:])
        if use_cupy:
            Ham = np.asarray(Ham)
            eigvalues, eigvectors = np.linalg.eigh(Ham)
            eigvalues = np.asnumpy(eigvalues)
            eigvectors = np.asnumpy(eigvectors)
        else:
            eigvalues, eigvectors = np.linalg.eigh(Ham)
            #eigvalues, eigvectors = spla.eig(Ham,b=Overlap)
            #sort_ind = np.argsort(eigvalues)
            #eigvalues = eigvalues[sort_ind].real
            #eigvectors = eigvectors[sort_ind,:]
        nocc = int(natoms / 2)
        fd_dist = 2*lp.eye(natoms)
        fd_dist[nocc:,nocc:] = 0
        density_matrix =  eigvectors @ fd_dist  @ lp.conj(eigvectors).T
        Energy += 2 * np.sum(eigvalues[:nocc]) /nocc

        #Zrho = density_matrix@Ham
        #Energy = np.sum(Zrho )

        Forces += get_hellman_feynman(atom_positions,mol_id, cell, eigvectors, params_str,kpoints[k,:] )

    return Energy,Forces


def get_tb_forces_energy_fd(atom_positions, mol_id, cell, kpoints, params_str, rcut=10):
    atom_positions = np.asarray(atom_positions)
    cell = np.asarray(cell)
    mol_id = np.asarray(mol_id)
    kpoints = np.asarray(kpoints)

    recip_cell = get_recip_cell(cell)
    if kpoints.ndim == 1:
        kpoints = np.reshape(kpoints, (1, 3))
    kpoints = kpoints @ recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]
    Energy = 0
    Forces = np.zeros((natoms, 3), dtype=np.complex64)
    
    
    for k in range(nkp):
        Ham = gen_ham_ovrlp(atom_positions,  mol_id, cell, kpoints[k], params_str)
        eigvalues, eigvectors = np.linalg.eigh(Ham)
        nocc = int(natoms / 2)
        Energy += 2 * np.sum(np.sort(eigvalues)[:nocc])
        Forces += get_hellman_feynman_fd(atom_positions,mol_id, cell, eigvectors, params_str,kpoints[k]) 
    if use_cupy:
        return np.asnumpy(Energy), np.asnumpy(Forces)
    else:
        return Energy,Forces

def calc_band_structure(atom_positions, mol_id, cell, kpoints, params_str):
    atom_positions = np.asarray(atom_positions)
    cell = np.asarray(cell)
    kpoints = np.asarray(kpoints)
    mol_id = np.asarray(mol_id)

    recip_cell = get_recip_cell(cell)
    if kpoints.ndim == 1:
        kpoints = np.reshape(kpoints, (1,3))
    kpoints = kpoints @ recip_cell.T
    natoms = atom_positions.shape[0]
    nkp = kpoints.shape[0]
    evals = np.zeros((natoms, nkp))
    evecs = np.zeros((natoms, natoms, nkp), dtype=np.complex64)
    
    for k in range(nkp):
        #Ham = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k], params_str)
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        #Ham = np.linalg.inv(Overlap) @ Ham #@ Overlap
        eigvalues, eigvectors = np.linalg.eigh(Ham)
        #eigvalues, eigvectors = spla.eig(Ham,b=Overlap)
        #sort_ind = np.argsort(eigvalues)
        #eigvalues = eigvalues[sort_ind].real
        #eigvectors = eigvectors[sort_ind,:]
        evals[:, k] = eigvalues
        evecs[:, :, k] = eigvectors
    if use_cupy:
        return np.asnumpy(evals), np.asnumpy(evecs)
    else:
        return evals,evecs

def get_param_dict(params_str):
    if params_str == "popov":
        params = {
            "B": {
                "B": {
                    "hopping": popov_hopping,
                    "self_energy": -5.2887,
                    "rcut": 3.7,
                },
                "Ti": {
                    "hopping": porezag_hopping,
                    "rcut": 5.29,
                },
            },
            "Ti": {
                "B": {
                    "hopping": porezag_hopping,
                    "rcut": 5.29,
                },
                "Ti": {
                    "hopping": popov_hopping,
                    "self_energy": -5.2887,
                    "rcut": 3.7,
                },
            },
        }
    else:  # params_str == "nn"
        params = {
            "B": {
                "B": {
                    "hopping": nnhop,
                    "self_energy": 0,
                    "rcut": 3,
                },
                "Ti": {
                    "hopping": nnhop,
                    "rcut": 3,
                },
            },
            "Ti": {
                "B": {
                    "hopping": nnhop,
                    "rcut": 3,
                },
                "Ti": {
                    "hopping": nnhop,
                    "self_energy": 0,
                    "rcut": 3,
                },
            },
        }
    return params
