import cupy as cp
import numpy as np 
from TEGT_GPU.TB_Utils_cupy import *
from TEGT_GPU.TB_parameters_cupy import *

def get_recip_cell(cell):
    a1 = cell[:, 0]
    a2 = cell[:, 1]
    a3 = cell[:, 2]

    volume = cp.dot(a1, cp.cross(a2, a3))

    b1 = 2 * cp.pi * cp.cross(a2, a3) / volume
    b2 = 2 * cp.pi * cp.cross(a3, a1) / volume
    b3 = 2 * cp.pi * cp.cross(a1, a2) / volume

    return cp.array([b1, b2, b3])

def get_tb_forces_energy(atom_positions,atom_types,cell,kpoints,params_str,rcut = 10):
    atom_positions = cp.asarray(atom_positions)
    atom_types = np.array(atom_types)
    cell = cp.asarray(cell)
    kpoints = cp.asarray(kpoints)

    params = get_param_dict(params_str)
    recip_cell = get_recip_cell(cell)
    
    if kpoints.shape == (3,):
        kpoints = kpoints.reshape((1, 3))
    
    kpoints = kpoints * recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]

    Energy = 0
    Forces = cp.zeros((natoms, 3), dtype=cp.complex64)

    for k in range(nkp):
        Ham = gen_ham_ovrlp(atom_positions, atom_types, cell, kpoints[k,:], params)
        eigvalues, eigvectors = cp.linalg.eigh(Ham)
        nocc = int(natoms / 2)
        Energy += 2 * cp.sum(eigvalues[:nocc])
        Forces += get_hellman_feynman(atom_positions,  atom_types, cell, eigvectors, kpoints[k,:], params)

    return cp.asnumpy(Energy), cp.asnumpy(Forces)

def get_tb_forces_energy_fd(atom_positions, atom_types, cell, kpoints, params_str, rcut=10):
    atom_positions = cp.asarray(atom_positions)
    atom_types = np.array(atom_types)
    cell = cp.asarray(cell)
    kpoints = cp.asarray(kpoints)
    params = get_param_dict(params_str)

    recip_cell = get_recip_cell(cell)
    if kpoints.ndim == 1:
        kpoints = cp.reshape(kpoints, (1, 3))
    kpoints = kpoints * recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]
    Energy = 0
    Forces = cp.zeros((natoms, 3), dtype=cp.complex64)
    
    for k in range(nkp):
        Ham = gen_ham_ovrlp(atom_positions,  mol_id, cell, kpoints[k], params)
        eigvalues, eigvectors = cp.linalg.eigh(Ham)
        nocc = int(natoms / 2)
        Energy += 2 * cp.sum(eigvalues[:nocc])
        Forces += get_hellman_feynman_fd(atom_positions, mol_id, cell, eigvectors, kpoints[k], params)
    
    return cp.asnumpy(Energy), cp.asnumpy(Forces)

def calc_band_structure(atom_positions, atom_types, cell, kpoints, params_str, rcut=10):
    atom_positions = cp.asarray(atom_positions)
    cell = cp.asarray(cell)
    kpoints = cp.asarray(kpoints)

    params = get_param_dict(params_str)
    recip_cell = get_recip_cell(cell)
    if kpoints.ndim == 1:
        kpoints = cp.reshape(kpoints, (1,3))
    kpoints = kpoints @ recip_cell.T
    natoms = atom_positions.shape[0]
    nkp = kpoints.shape[0]
    evals = cp.zeros((natoms, nkp))
    evecs = cp.zeros((natoms, natoms, nkp), dtype=cp.complex64)
    
    for k in range(nkp):
        Ham = gen_ham_ovrlp(atom_positions, atom_types, cell, kpoints[k], params)
        eigvalues, eigvectors = cp.linalg.eigh(Ham)
        evals[:, k] = eigvalues
        evecs[:, :, k] = eigvectors
    return cp.asnumpy(evals), cp.asnumpy(evecs)

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
