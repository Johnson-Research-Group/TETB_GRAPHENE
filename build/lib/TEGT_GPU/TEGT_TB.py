import cupy as cp
#import numpy as cp
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
    Binv = cp.linalg.inv(B)
    renorm_A  = Binv @ A
    eigvals,eigvecs = cp.linalg.eigh(renorm_A)
    #normalize eigenvectors s.t. eigvecs.conj().T @ B @ eigvecs = I
    Q = eigvecs.conj().T @ B @ eigvecs
    U = cp.linalg.cholesky(cp.linalg.inv(Q))
    eigvecs = eigvecs @ U 
    eigvals = cp.diag(eigvecs.conj().T @ A @ eigvecs).real
    #print("GPU version ",np.round((eigvecs.conj().T @ B @ eigvecs).real,decimals=2))

    return eigvals,eigvecs


def get_tb_forces_energy(atom_positions,mol_id,cell,kpoints,params_str):
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
        eigvalues,eigvectors = generalized_eigen(Ham,Overlap)
        del Ham
        del Overlap
        nocc = int(natoms / 2)
        Energy += 2 * cp.sum(eigvalues[:nocc])

        Forces += get_hellman_feynman(atom_positions,mol_id, cell, eigvalues,eigvectors, params_str,kpoints[k,:] )
        del eigvalues
        del eigvectors

    return cp.asnumpy(Energy),cp.asnumpy(Forces)

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
    atom_positions = cp.asarray(atom_positions)
    cell = cp.asarray(cell)
    mol_id = cp.asarray(mol_id)
    kpoints = cp.asarray(kpoints)

    recip_cell = get_recip_cell(cell)
    if kpoints.ndim == 1:
        kpoints = cp.reshape(kpoints, (1, 3))
    kpoints = kpoints @ recip_cell
    nkp = kpoints.shape[0]
    natoms = atom_positions.shape[0]
    Energy = 0
    Forces = cp.zeros((natoms, 3), dtype=cp.complex64)
    
    
    for k in range(nkp):
        Ham,Overlap = gen_ham_ovrlp(atom_positions,  mol_id, cell, kpoints[k], params_str)
        eigvalues, eigvectors = generalized_eigen(Ham,Overlap)
        nocc = int(natoms / 2)
        Energy += 2 * cp.sum(eigvalues[:nocc])
        Forces += get_hellman_feynman_fd(atom_positions,mol_id, cell, eigvectors, params_str,kpoints[k]) 
    return cp.asnumpy(Energy),cp.asnumpy(Forces)

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
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        eigvalues,eigvectors = generalized_eigen(Ham,Overlap)  
        evals[:, k] = eigvalues
        evecs[:, :, k] = eigvectors
    return cp.asnumpy(evals), cp.asnumpy(evecs)
