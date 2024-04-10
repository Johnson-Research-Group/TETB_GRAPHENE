import cupy as cp
import numpy as np 
import scipy.linalg as spla
from TETB_GRAPHENE_GPU.TB_Utils import *
from TETB_GRAPHENE_GPU.TB_parameters import *

def get_recip_cell(cell):
    """find reciprocal cell given real space cell
    :param cell: (np.ndarray [3,3]) real space cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector
    
    :returns: (np.ndarray [3,3]) reciprocal cell of system where recip_cell[i, j] is the jth Cartesian coordinate of the ith reciprocal cell vector"""
    a1 = cell[:, 0]
    a2 = cell[:, 1]
    a3 = cell[:, 2]

    volume = cp.dot(a1, np.cross(a2, a3))

    b1 = 2 * cp.pi * cp.cross(a2, a3) / volume
    b2 = 2 * cp.pi * cp.cross(a3, a1) / volume
    b3 = 2 * cp.pi * cp.cross(a1, a2) / volume

    return cp.array([b1, b2, b3])

def generalized_eigen(A,B):
    """generalized eigen value solver using cupy. equivalent to scipy.linalg.eigh(A,B=B) """
    Binv = cp.linalg.inv(B)
    renorm_A  = Binv @ A
    eigvals,eigvecs = cp.linalg.eigh(renorm_A)
    #normalize eigenvectors s.t. eigvecs.conj().T @ B @ eigvecs = I
    Q = eigvecs.conj().T @ B @ eigvecs
    U = cp.linalg.cholesky(cp.linalg.inv(Q))
    eigvecs = eigvecs @ U 
    eigvals = cp.diag(eigvecs.conj().T @ A @ eigvecs).real

    return eigvals,eigvecs


def get_tb_forces_energy(atom_positions,mol_id,cell,kpoints,params_str):
    """calculate tight binding forces using hellman feynman theorem and energy for a given array of kpoints, using cupy  
    
    :param atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :param mol_id: (np.ndarray [Natoms,]) atom types expressed as integers

    :param cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :param kpoints: (np.ndarray [number of kpoints,3]) array of kpoints to calculate tight binding calculation over

    :param params_str: (str) specify which tight binding model to use
    
    :returns: tuple(float, np.ndarray [number of atoms, 3]) tight binding energy, tight binding forces"""
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


def get_tb_forces_energy_fd(atom_positions, mol_id, cell, kpoints, params_str, rcut=10):
    """calculate tight binding forces using finite differences and energy for a given array of kpoints, using cupy. 
    **note this method is very slow since it has to solve a generalized eigenvalue problem for each atom. meant for testing purposes only
    
    :param atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :param mol_id: (np.ndarray [Natoms,]) atom types expressed as integers

    :param cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :param kpoints: (np.ndarray [number of kpoints,3]) array of kpoints to calculate tight binding calculation over

    :param params_str: (str) specify which tight binding model to use
    
    :returns: tuple(float, np.ndarray [number of atoms, 3]) tight binding energy, tight binding forces"""
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
    """get band structure for a given system and path in kspace.
         
    :param atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :param mol_id: (np.ndarray [Natoms,]) atom types expressed as integers

    :param cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :param kpoints: (np.ndarray [number of kpoints,3]) array of kpoints to calculate tight binding calculation over

    :param params_str: (str) specify which tight binding model to use
    
    :returns: (np.ndarray [Number of eigenvalues, number of kpoints])"""
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
