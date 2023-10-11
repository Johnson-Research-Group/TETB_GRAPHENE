using LinearAlgebra

#=function wrap_disp(displacement, cell_vectors)
    for dim in 1:3
        if displacement[dim] > cell_vectors[dim] / 2
            displacement[dim] -= cell_vectors[dim]
        end
        if displacement[dim] <= -cell_vectors[dim] / 2
            displacement[dim] += cell_vectors[dim]
        end
    end
    return displacement
end=#

@everywhere function wrap_disp(r1, r2, cell)
    """Wrap positions to unit cell. 3D"""
    RIJ = zeros(3)
    d = 1000.0
    drij = zeros(3)
    for i in [-1, 0, 1], j in [-1, 0, 1], k in [-1, 0 ,1]
        pbc = [i, j, k]
        RIJ = r2 + cell'*pbc - r1
        norm_RIJ = norm(RIJ)
        
        if norm_RIJ < d
            d = norm_RIJ
            drij = copy(RIJ)
        end
    end
    
    return drij
end

@everywhere function gen_ham_ovrlp(atom_positions, neighbor_list,atom_types,cell_vectors, kpoint,params)
    natoms = size(atom_positions)[1]
    Ham = zeros(ComplexF64,natoms,natoms)
    Overlap = zeros(ComplexF64,natoms,natoms)
    for i in 1:natoms
        Ham[i,i] = params[atom_types[i]][atom_types[i]]["self_energy"]
        Overlap[i,i] = 1
        neighbors = neighbor_list[i]

        for n in neighbors
            #pbc
            disp = wrap_disp(atom_positions[i,begin:end],atom_positions[n,begin:end], cell_vectors)
            dist = norm(disp)
            
            if dist < params[atom_types[i]][atom_types[n]]["rcut"] && dist > 1
                phase = exp(1im*dot(kpoint,disp))
                Ham[i,n] += params[atom_types[i]][atom_types[n]]["hopping"](disp) * phase
                #Overlap[i,j] = params[atom_types[i]][atom_types[j]]["ovrlp"](disp) * phase
            end

        end
    end
    #plt=heatmap(real(Ham), c=:viridis, xlabel="Column Index", ylabel="Row Index", title="Hamiltonian Matrix")
    #display(plt)
    return Hermitian(Ham), Hermitian(Overlap)
end

@everywhere function get_hellman_feynman_fd(atom_positions,neighbor_list,atom_types,cell_vectors,eigvec,kpoint,params)
    #give eigenvector at 1 kpoint, higher up fxn will average over Kpoints
    natoms = size(atom_positions)[1]
    Force = zeros(natoms,3)
    dr=1e-3
    for i in 1:natoms
        neighbors = neighbor_list[i]
        for n in neighbors
            disp = wrap_disp(atom_positions[i,begin:end],atom_positions[n,begin:end], cell_vectors)
            dist = norm(disp)
            if dist < params[atom_types[i]][atom_types[n]]["rcut"] && dist > 1
                phase = exp(-1im*dot(kpoint,disp))
                # i don't think you need to differentiate the phase, since we only want real part of dH/dr
                # derivative of phase will cancel out when we take complex conjugate

                dtdx = (params[atom_types[i]][atom_types[n]]["hopping"](disp+[dr,0,0])\
                    -params[atom_types[i]][atom_types[n]]["hopping"](disp+[-dr,0,0]))*phase/2/dr

                dtdy = (params[atom_types[i]][atom_types[n]]["hopping"](disp+[0,dr,0])\
                    -params[atom_types[i]][atom_types[n]]["hopping"](disp+[0,-dr,0]))*phase/2/dr

                dtdz = (params[atom_types[i]][atom_types[n]]["hopping"](disp+[0,0,dr])\
                    -params[atom_types[i]][atom_types[n]]["hopping"](disp+[0,0,-dr]))*phase/2/dr

                Force[i,1] += conj(eigvec[n,i]) * dtdx * eigvec[i,n]/2
                Force[i,2] += conj(eigvec[n,i]) * dtdy * eigvec[i,n]/2
                Force[i,3] += conj(eigvec[n,i]) * dtdz * eigvec[i,n]/2
            end

        end
    end
    return Force
end