import cupy as cp
import smallpebble as sp

def wrap_disp(r1, r2, cell):
    """Wrap positions to unit cell. 3D"""
    pbc = cp.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1],])
    pbc = cp.transpose(pbc)
    r2_replicated = r2 + cp.dot(pbc, cell.T) - r1
    return r2_replicated

def gen_ham_ovrlp(atom_positions, atom_types, cell, kpoint, params):
    natoms = atom_positions.shape[0]
    Ham = cp.zeros((natoms, natoms), dtype=cp.complex64)
    Overlap = cp.zeros((natoms, natoms), dtype=cp.complex64)

    i_indices, j_indices = cp.triu_indices(natoms, k=0)
    disp = wrap_disp(atom_positions[i_indices], atom_positions[j_indices], cell)
    dist = cp.linalg.norm(disp, axis=1)
    
    valid_indices = (dist < params[atom_types[i_indices]][atom_types[j_indices]]["rcut"]) & (dist > 1)
    valid_indices &= (i_indices != j_indices)

    phase = cp.exp(1j * cp.dot(kpoint, disp[valid_indices]))
    Ham[i_indices[valid_indices], j_indices[valid_indices]] += params[atom_types[i_indices[valid_indices]]][atom_types[j_indices[valid_indices]]]["hopping"](disp[valid_indices]) * phase
    cp.fill_diagonal(Ham, [params[atom_types[i]][atom_types[i]]["self_energy"] for i in range(natoms)])

    return cp.hermitian(Ham)

def get_helem_fxn(r2, cell, typei, typen, params):
    def helem(r1):
        disp = wrap_disp(r1, r2, cell)
        dist = cp.linalg.norm(disp)
        if 0.5 < dist < params[typei][typen]["rcut"]:
            hop = params[typei][typen]["hopping"](disp)
            return hop
        else:
            return 0

    return helem

def get_hellman_feynman_fd(atom_positions, atom_types, cell, eigvec, kpoint, params):
    dr = 1e-4
    natoms, _ = atom_positions.shape
    nocc = len(eigvec) // 2
    Forces = cp.zeros((natoms, 3), dtype=cp.float64)
    atom_positions_pert = cp.copy(atom_positions)

    for dir_ind in range(3):
        atom_positions_pert[:, dir_ind] += dr
        Ham = gen_ham_ovrlp(atom_positions_pert,  atom_types, cell, kpoint, params)
        eigvalues, eigvectors = cp.linalg.eigh(Ham)
        Energy_up = 2 * cp.sum(eigvalues[:nocc])

        atom_positions_pert[:, dir_ind] -= 2 * dr
        Ham = gen_ham_ovrlp(atom_positions_pert, atom_types, cell, kpoint, params)
        eigvalues, eigvectors = cp.linalg.eigh(Ham)
        Energy_dwn = 2 * cp.sum(eigvalues[:nocc])

        Forces[:, dir_ind] = -(Energy_up - Energy_dwn) / (2 * dr)

        atom_positions_pert[:, dir_ind] = cp.copy(atom_positions[:, dir_ind])

    return Forces

def get_hellman_feynman(atom_positions, atom_types, cell, eigvec, kpoint, params):
    natoms = atom_positions.shape[0]
    nocc = cp.int(eigvec.shape[0] / 2)
    Force = cp.zeros((natoms, 3), dtype=cp.complex64)
    
    i_indices, j_indices = cp.triu_indices(natoms, k=1)
    r1 = atom_positions[i_indices]
    r2 = atom_positions[j_indices]
    typei = atom_types[i_indices]
    typen = atom_types[j_indices]
    disp = wrap_disp(r1, r2, cell)
    dist = cp.linalg.norm(disp, axis=1)
    
    valid_indices = (0.5 < dist) & (dist < params[typei][typen]["rcut"])
    i_indices = i_indices[valid_indices]
    j_indices = j_indices[valid_indices]

    helem_fxn = get_helem_fxn(r2[valid_indices], cell, typei, typen, params)

    #I need to make sure this works
    gradH = sp.get_gradients(helem_fxn)
    gradH = cp.asarray([gradH[r1[i]] for i in range(len(i_indices))])

    rho = cp.sum(cp.conj(eigvec[j_indices, :nocc]) * eigvec[i_indices, :nocc], axis=1)
    disp = wrap_disp(r1, r2[valid_indices], cell)
    phase = cp.exp(1j * cp.dot(kpoint, disp))

    ave_gradH = 4 * gradH * rho * phase[:, cp.newaxis]

    for i in range(natoms):
        i_mask = (i_indices == i)
        Force[i, 0] = -cp.sum(ave_gradH[i_mask, 0])
        Force[i, 1] = -cp.sum(ave_gradH[i_mask, 1])
        Force[i, 2] = -cp.sum(ave_gradH[i_mask, 2])

    return Force