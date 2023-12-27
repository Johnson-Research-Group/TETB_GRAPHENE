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
#from TEGT_GPU.TB_Utils_cupy_V2 import *
from TB_Utils_cupy_V2 import *
#from TEGT_GPU.TB_parameters_cupy_V2 import *
from TB_parameters_cupy_V2 import *

def get_recip_cell(cell):
    a1 = cell[:, 0]
    a2 = cell[:, 1]
    a3 = cell[:, 2]

    volume = lp.dot(a1, lp.cross(a2, a3))

    b1 = 2 * lp.pi * lp.cross(a2, a3) / volume
    b2 = 2 * lp.pi * lp.cross(a3, a1) / volume
    b3 = 2 * lp.pi * lp.cross(a1, a2) / volume

    return lp.array([b1, b2, b3])

def get_tb_forces_energy(atom_positions,mol_id,cell,kpoints,params_str,rcut = 10):
    atom_positions = lp.asarray(atom_positions)
    cell = lp.asarray(cell)
    kpoints = lp.asarray(kpoints)
    mol_id = lp.asarray(mol_id)

    #params = get_param_dict(params_str)
    recip_cell = get_recip_cell(cell)
    
    if kpoints.shape == (3,):
        kpoints = kpoints.reshape((1, 3))
    
    kpoints = kpoints @ recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]

    Energy = 0
    Forces = lp.zeros((natoms, 3), dtype=lp.complex64)

    for k in range(nkp):
        Ham,i,j, di, dj, phase = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        eigvalues, eigvectors = lp.linalg.eigh(Ham)
        nocc = int(natoms / 2)
        Energy += 2 * lp.sum(eigvalues[:nocc])
        Forces += get_hellman_feynman(atom_positions,mol_id, cell, eigvectors, params_str ,i,j, di, dj, phase)
    if use_cupy:
        return lp.asnumpy(Energy), lp.asnumpy(Forces)
    else:
        return Energy,Forces


def get_tb_forces_energy_fd(atom_positions, mol_id, cell, kpoints, params_str, rcut=10):
    atom_positions = lp.asarray(atom_positions)
    cell = lp.asarray(cell)
    mol_id = lp.asarray(mol_id)
    kpoints = lp.asarray(kpoints)
    eV_per_hartree = 1 #27.211407953
    #params = get_param_dict(params_str)

    recip_cell = get_recip_cell(cell)
    if kpoints.ndim == 1:
        kpoints = lp.reshape(kpoints, (1, 3))
    kpoints = kpoints @ recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]
    Energy = 0
    Forces = lp.zeros((natoms, 3), dtype=lp.complex64)
    
    for k in range(nkp):
        Ham,_,_, _, _, _ = gen_ham_ovrlp(atom_positions,  mol_id, cell, kpoints[k], params_str)
        eigvalues, eigvectors = lp.linalg.eigh(Ham)
        nocc = int(natoms / 2)
        Energy += 2 * lp.sum(eigvalues[:nocc])*eV_per_hartree
        Forces += get_hellman_feynman_fd(atom_positions,mol_id, cell, eigvectors, params_str,kpoints[k]) 
    if use_cupy:
        return lp.asnumpy(Energy), lp.asnumpy(Forces)
    else:
        return Energy,Forces

def calc_band_structure(atom_positions, mol_id, cell, kpoints, params_str, rcut=10):
    atom_positions = lp.asarray(atom_positions)
    cell = lp.asarray(cell)
    kpoints = lp.asarray(kpoints)
    mol_id = lp.asarray(mol_id)
    eV_per_hartree = 1 # 27.211407953
    #params = get_param_dict(params_str)
    recip_cell = get_recip_cell(cell)
    if kpoints.ndim == 1:
        kpoints = lp.reshape(kpoints, (1,3))
    kpoints = kpoints @ recip_cell
    natoms = atom_positions.shape[0]
    nkp = kpoints.shape[0]
    evals = lp.zeros((natoms, nkp))
    evecs = lp.zeros((natoms, natoms, nkp), dtype=lp.complex64)
    
    for k in range(nkp):
        Ham,_,_, _, _, _ = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k], params_str)
        eigvalues, eigvectors = lp.linalg.eigh(Ham)
        evals[:, k] = eigvalues*eV_per_hartree
        evecs[:, :, k] = eigvectors
    if use_cupy:
        return lp.asnumpy(evals), lp.asnumpy(evecs)
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
