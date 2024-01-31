#from cupyx.scipy.spatial.distance import cdist
#from cupyx.scipy.linalg import solve_triangular
#import cupy as cp
from scipy.linalg import solve_triangular
import numpy as cp
import numpy as np 
import scipy.linalg as spla
from TEGT_GPU.TB_Utils import *
#from TB_Utils import *
from TEGT_GPU.TB_parameters import *
#from TB_parameters import *

def get_recip_cell(cell):
    a1 = cell[:, 0]
    a2 = cell[:, 1]
    a3 = cell[:, 2]

    volume = cp.dot(a1, np.cross(a2, a3))

    b1 = 2 * cp.pi * cp.cross(a2, a3) / volume
    b2 = 2 * cp.pi * cp.cross(a3, a1) / volume
    b3 = 2 * cp.pi * cp.cross(a1, a2) / volume

    return cp.array([b1, b2, b3])

def generalized_eigen(A,B):
    lambda_B,Phi_B = cp.linalg.eigh(B)
    Lambda_B_squareRoot = cp.nan_to_num(lambda_B**0.5) #+0.000001
    del lambda_B
    Lambda_B_squareRoot = cp.diag(Lambda_B_squareRoot)
    Phi_B = Phi_B.dot(cp.linalg.inv(Lambda_B_squareRoot))
    A_hat = (Phi_B.T).dot(A).dot(Phi_B)
    Lambda_A, Phi_A = cp.linalg.eigh(A_hat)
    Phi = Phi_B.dot(Phi_A)
    return Lambda_A, Phi

def get_tb_forces_energy(atom_positions,mol_id,cell,kpoints,params_str,rcut = 10):
    atom_positions = cp.asarray(atom_positions)
    cell = cp.asarray(cell)
    kpoints = cp.asarray(kpoints)
    mol_id = cp.asarray(mol_id)
    recip_cell = get_recip_cell(cell)
    
    if kpoints.shape == (3,):
        kpoints = kpoints.reshape((1, 3))
    
    kpoints = kpoints @ recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]

    Energy = 0
    Forces = cp.zeros((natoms, 3), dtype=cp.complex64)
    for k in range(nkp):
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str) 
        eigvalues, eigvectors = generalized_eigen(Ham,Overlap) 
        nocc = int(natoms / 2)
        Energy += 2 * cp.sum(eigvalues[:nocc])

        Forces += get_hellman_feynman(atom_positions,mol_id, cell, eigvalues,eigvectors, params_str,kpoints[k,:] )

    #return cp.asnumpy(Energy),cp.asnumpy(Forces)
    return Energy, Forces

def get_interlayer_tb_forces(atom_positions,mol_id,cell,kpoints,params_str):
    atom_positions = np.asarray(atom_positions)
    cell = np.asarray(cell)
    kpoints = np.asarray(kpoints)
    mol_id = np.asarray(mol_id)
    recip_cell = get_recip_cell(cell)
    
    if kpoints.shape == (3,):
        kpoints = kpoints.reshape((1, 3))
    
    kpoints = kpoints @ recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]

    interlayer_Forces = 0
    for k in range(nkp):
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        eigvalues, eigvectors = spla.eigh(Ham,b=Overlap)
        interlayer_Forces += get_hellman_feynman_interlayer(atom_positions,mol_id, cell, eigvalues,eigvectors, params_str,kpoints[k,:] )
    return interlayer_Forces 


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
    #return np.asnumpy(Energy), np.asnumpy(Forces)
    return Energy,Forces

def calc_band_structure(atom_positions, mol_id, cell, kpoints, params_str):
    atom_positions = cp.asarray(atom_positions)
    cell = cp.asarray(cell)
    kpoints = cp.asarray(kpoints)
    mol_id = cp.asarray(mol_id)

    recip_cell = get_recip_cell(cell)
    if kpoints.ndim == 1:
        kpoints = np.reshape(kpoints, (1,3))
    kpoints = kpoints @ recip_cell.T
    natoms = atom_positions.shape[0]
    nkp = kpoints.shape[0]
    evals = cp.zeros((natoms, nkp))
    evecs = cp.zeros((natoms, natoms, nkp), dtype=cp.complex64)
    
    for k in range(nkp):
        #Ham = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k], params_str)
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        eigvalues,eigvectors = generalized_eigen(Ham,Overlap) 
        evals[:, k] = eigvalues
        evecs[:, :, k] = eigvectors
    #return np.asnumpy(evals), np.asnumpy(evecs)
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
