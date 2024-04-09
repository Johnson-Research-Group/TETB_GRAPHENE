from scipy.spatial.distance import cdist
import numpy as np 
import scipy.linalg as spla
from TETB_GRAPHENE.TB_Utils import *
from TETB_GRAPHENE.TB_parameters import *

def get_recip_cell(cell):
    """find reciprocal cell given real space cell
    :param cell: (np.ndarray [3,3]) real space cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector
    
    :returns: (np.ndarray [3,3]) reciprocal cell of system where recip_cell[i, j] is the jth Cartesian coordinate of the ith reciprocal cell vector"""
    a1 = cell[:, 0]
    a2 = cell[:, 1]
    a3 = cell[:, 2]

    volume = np.dot(a1, np.cross(a2, a3))

    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume

    return np.array([b1, b2, b3])


def get_tb_forces_energy(atom_positions,mol_id,cell,kpoints,params_str):
    """calculate tight binding forces using hellman feynman theorem and energy for a given array of kpoints  
    
    :param atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :param mol_id: (np.ndarray [Natoms,]) atom types expressed as integers

    :param cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :param kpoints: (np.ndarray [number of kpoints,3]) array of kpoints to calculate tight binding calculation over

    :param params_str: (str) specify which tight binding model to use
    
    :returns: tuple(float, np.ndarray [number of atoms, 3]) tight binding energy, tight binding forces"""
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

    Energy = 0
    Forces = np.zeros((natoms, 3), dtype=np.complex64)
    for k in range(nkp):
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        eigvalues, eigvectors = spla.eigh(Ham,b=Overlap)
        nocc = int(natoms / 2)
        Energy += 2 * np.sum(eigvalues[:nocc])

        Forces += get_hellman_feynman(atom_positions,mol_id, cell, eigvalues,eigvectors, params_str,kpoints[k,:] )

    return Energy,Forces


def get_tb_forces_energy_fd(atom_positions, mol_id, cell, kpoints, params_str):
    """calculate tight binding forces using finite differences and energy for a given array of kpoints, using cupy. 
    **note this method is very slow since it has to solve a generalized eigenvalue problem for each atom. meant for testing purposes only
    
    :param atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :param mol_id: (np.ndarray [Natoms,]) atom types expressed as integers

    :param cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :param kpoints: (np.ndarray [number of kpoints,3]) array of kpoints to calculate tight binding calculation over

    :param params_str: (str) specify which tight binding model to use
    
    :returns: tuple(float, np.ndarray [number of atoms, 3]) tight binding energy, tight binding forces"""
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
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        eigvalues, eigvectors = spla.eigh(Ham,b=Overlap)
        nocc = int(natoms / 2)
        Energy += 2 * np.sum(np.sort(eigvalues)[:nocc])
        Forces += get_hellman_feynman_fd(atom_positions,mol_id, cell, eigvectors, params_str,kpoints[k]) 

    return Energy,Forces

def calc_band_structure(atom_positions, mol_id, cell, kpoints, params_str):
    """get band structure for a given system and path in kspace.
         
    :param atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :param mol_id: (np.ndarray [Natoms,]) atom types expressed as integers

    :param cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :param kpoints: (np.ndarray [number of kpoints,3]) array of kpoints to calculate tight binding calculation over

    :param params_str: (str) specify which tight binding model to use
    
    :returns: (np.ndarray [Number of eigenvalues, number of kpoints])"""
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
        Ham,Overlap = gen_ham_ovrlp(atom_positions, mol_id, cell, kpoints[k,:], params_str)
        eigvalues, eigvectors = spla.eigh(Ham,b=Overlap)
        evals[:, k] = eigvalues
        evecs[:, :, k] = eigvectors
    return evals,evecs
