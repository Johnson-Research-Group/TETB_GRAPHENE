import cupy as cp
import numpy as np
import smallpebble as sp
import matplotlib.pyplot as plt
def wrap_disp(r1, r2, cell):
    """Wrap positions to unit cell. 3D"""
    d = 1000.0
    drij = cp.zeros_like(r1)
    if r1.ndim==1:
        r1 = cp.array([r1])
        r2 = cp.array([r2])

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                pbc = cp.array([i, j, k])

                RIJ = r2 + cp.dot(pbc, cell) - r1
                norm_RIJ = cp.linalg.norm(RIJ, axis=1)

                mask = norm_RIJ < d
                d = cp.where(mask, norm_RIJ, d)
                drij = cp.where(mask[:, None], RIJ, drij)

    return np.squeeze(drij)


def gen_ham_ovrlp(atom_positions, atom_types, cell, kpoint, params):
    natoms = atom_positions.shape[0]
    Ham = cp.zeros((natoms, natoms), dtype=cp.complex64)
    atom_types = np.array(atom_types)
    #upper triangle indices
    i_indices, j_indices = cp.triu_indices(natoms, k=0)
    disp = wrap_disp(atom_positions[i_indices], atom_positions[j_indices], cell)
    dist = cp.linalg.norm(disp, axis=1)
    self_energies = cp.ones(natoms)
    #get the different atomic species in system
    atom_types_list = set(atom_types)
    # Use atom_types directly for indexing
    for i,typei in enumerate(atom_types_list):
        for j,typej in enumerate(atom_types_list):
            #if j<i:
            #    continue
            hop_func = params[typei][typej]["hopping"]
            rcut = params[typei][typej]["rcut"]
            valid_indices = (dist < rcut) & (dist > 0.6)
            #valid_indices &= (i_indices != j_indices)
            #print(atom_types[i_indices.get()])
            #print("typej ",atom_types[j_indices.get()])
            #print(typei,typej)
            valid_indices &= cp.array(atom_types[i_indices.get()]==typei)
            valid_indices &= cp.array(atom_types[j_indices.get()]==typej)

            phase = cp.exp(1j * cp.dot(kpoint, disp[valid_indices].T))
            Ham[i_indices[valid_indices], j_indices[valid_indices]] += 1 #hop_func(disp[valid_indices])*phase
            Ham[j_indices[valid_indices], i_indices[valid_indices]] += 1 #hop_func(disp[valid_indices])*phase.conj()
            if i==j:
                self_energy = params[typei][typej]["self_energy"]
                valid_indices = cp.array(atom_types[i_indices.get()]==typei)
                Ham[i_indices[valid_indices],i_indices[valid_indices]] = self_energy
    
    plt.imshow(cp.asnumpy(Ham).real)
    plt.colorbar()
    plt.savefig("gpu_ham.png")
    plt.clf()

    return Ham

def get_helem_fxn(r2, cell, typei, typen, params):
    def helem(r1):
        disp = wrap_disp(r1, r2, cell)
        hop = params[typei][typen]["hopping"](disp)
        return hop

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

"""def get_hellman_feynman(atom_positions, atom_types, cell, eigvec, kpoint, params):
    natoms = atom_positions.shape[0]
    nocc = eigvec.shape[0] //2
    Force = cp.zeros((natoms, 3), dtype=cp.complex64)
    i_indices, j_indices = cp.triu_indices(natoms, k=1)
    r1 = atom_positions[i_indices]
    r2 = atom_positions[j_indices]
    atom_types_list = set(atom_types)
    # Use atom_types directly for indexing
    for i,typei in enumerate(atom_types_list):
        valid_indices = cp.array(atom_types[i_indices.get()]==typei)
        for j,typej in enumerate(atom_types_list):        
            r1_list = r1[valid_indices,:]
            valid_indices &= cp.array(atom_types[j_indices.get()]==typej)
            r2_list = r2[valid_indices,:]
            dr = 1e-3
            for r_current in r1_list:
                disp = wrap_disp(r_current, r2_list, cell)
                dist = cp.linalg.norm(disp, axis=1)
                rcut = params[typei][typej]["rcut"]
                rcut = cp.asarray(rcut)
                
                valid_indices &= (0.5 < dist) & (dist < rcut)
                
                i_indices = i_indices[valid_indices]
                j_indices = j_indices[valid_indices]

                helem_fxn = get_helem_fxn(r2_list, cell, typei, typej, params)

                ave_gradH = np.zeros((natoms,3))
                for i in range(3):
                    dr_v = np.zeros_like(r1[j,:])
                    dr_v[:,i]+=dr
                    H_d = helem_fxn(r1[j,:]-dr_v)
                    H = helem_fxn(r1[j,:])
                    H_up = helem_fxn(r1[j,:]+dr_v)
                    print(np.shape(H))
                    print(H)
                    H_diff = np.stack((H_up,H,H_d),axis=2)
                    #I need to make sure this works
                    helem = helem_fxn(H_diff)
                    #gradH = sp.get_gradients(helem)
                    gradH = cp.gradient(helem_fxn,axis=2)[:,:,1]

                    rho = cp.sum(cp.conj(eigvec[j_indices, :nocc]) * eigvec[i_indices, :nocc], axis=1)
                    disp = wrap_disp(r1, r2[valid_indices], cell)
                    phase = cp.exp(1j * cp.dot(kpoint, disp))

                    ave_gradH[j,i] = -cp.sum(4 * gradH * rho * phase)

                for i in range(natoms):
                    i_mask = (i_indices == i)
                    Force[i, 0] += -cp.sum(ave_gradH[i_mask, 0])
                    Force[i, 1] += -cp.sum(ave_gradH[i_mask, 1])
                    Force[i, 2] += -cp.sum(ave_gradH[i_mask, 2])

    return Force"""
def get_hellman_feynman(atom_positions, atom_types, cell, eigvec, kpoint, params):
    natoms = atom_positions.shape[0]
    nocc = eigvec.shape[0] //2
    Force = cp.zeros((natoms, 3), dtype=cp.complex64)
    atom_types_list = set(atom_types)
    # Use atom_types directly for indexing
    for i,typei in enumerate(atom_types_list):
        type1_indices = cp.array(np.squeeze(np.where(atom_types==typei)))
        for j,typej in enumerate(atom_types_list):
            type2_indices = cp.array(np.squeeze(np.where(atom_types==typej)))
            dr = 1e-3
            for i_ind in type1_indices:
                r1 = atom_positions[i_ind,:]
                for j_ind in type2_indices:
                    if i_ind==j_ind:
                        continue
                    r2 = atom_positions[j_ind,:]
                    disp = wrap_disp(r1, r2, cell)
                    dist = cp.linalg.norm(disp)
                    rcut = params[typei][typej]["rcut"]
                    rcut = cp.asarray(rcut)
                    if dist > rcut:
                        continue

                    ave_gradH = cp.zeros((3))
                    disp = wrap_disp(r1, r2, cell)
                    phase = cp.exp(1j * cp.dot(kpoint, disp))
                    
                    for i in range(3):
                        dr_v = cp.zeros_like(r1)
                        dr_v[i]+=dr
                        disp = wrap_disp(r1-dr_v, r2, cell)
                        H_d = params[typei][typej]["hopping"](cp.array([disp]))

                        disp = wrap_disp(r1+dr_v, r2, cell)
                        H_up = params[typei][typej]["hopping"](cp.array([disp]))
                        gradH = (H_up - H_d)/2/dr

                        rho = cp.dot(eigvec[j_ind,:nocc],cp.conj(eigvec[i_ind,:nocc]))

                        ave_gradH = -cp.sum(4 * gradH * rho * phase)
                        Force[i_ind,:] = ave_gradH

    return Force
