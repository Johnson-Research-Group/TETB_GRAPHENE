using LinearAlgebra
using ForwardDiff
#using CUDA
using SparseArrays


function wrap_disp(r1, r2, cell)
    """Wrap positions to unit cell. 3D"""
    RIJ = zeros(3)
    d = 1000.0
    drij = zeros(Float64,3)
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

function gen_ham_ovrlp(atom_positions, neighbor_list,atom_types,cell, kpoint,params)
    natoms = size(atom_positions)[1]
    Ham = spzeros(ComplexF64,natoms,natoms)
    Overlap = spzeros(ComplexF64,natoms,natoms)
    for i in 1:natoms
        Ham[i,i] = params[atom_types[i]][atom_types[i]]["self_energy"]
        Overlap[i,i] = 1
        neighbors = neighbor_list[i]

        for n in neighbors
            #pbc
            disp = wrap_disp(atom_positions[i,begin:end],atom_positions[n,begin:end], cell)
            dist = norm(disp)
            
            if dist < params[atom_types[i]][atom_types[n]]["rcut"] && dist > 1
                phase = exp(1im*dot(kpoint,disp))
                Ham[i,n] += params[atom_types[i]][atom_types[n]]["hopping"](disp) * phase
                Overlap[i,n] = params[atom_types[i]][atom_types[n]]["ovrlp"](disp) * phase
            end

        end
    end

    return Hermitian(Ham), Hermitian(Overlap)
end

function gen_ham(atom_positions,atom_types,cell,kpoint,params)
    """
    Compute hoppings in a hexagonal environment of the computation cell 
    Adequate for large unit cells (> 100 atoms)
    Input:
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in angstroms
        atom_types    - int   (natoms) atom type index of atom i
        cell - float (nlat x 3) where nlat = lattice vectors for graphene in angstroms
        kpoint  - wave vector
        params   - model for computing hoppings

    Output:
        Ham - (natoms x natoms) sparse Hamiltonian matrix at given kpoint
    """
    angstroms_to_bohr = 1.88973
    atom_positions *= angstroms_to_bohr
    cell *= angstroms_to_bohr
    kpoint /= angstroms_to_bohr
    models_functions = Dict("letb"=>letb,"mk"=>mk,"popov"=>popov)
    hopping_model = models_functions[params]
    natom = size(atomic_basis, 1)
    di = Int[]
    dj = Int[]
    i = Int[]
    j = Int[]
    extended_coords = Float64[]
    disp_table = spzeros(natoms,natoms,3)
    Ham = spzeros(ComplexF64,natoms,natoms)
    for dx in [-1, 0, 1], dy in [-1, 0, 1]
        for atom_index in 1:natom
            append!(di, dx)
            append!(dj, dy)
            append!(i, atom_index)
            append!(j, ((atom_index - 1) * 9 + (dx + 2) * 3 + dy + 2) % natom)
            extended_coords .= atomic_basis[atom_index, :] .+ lattice_vectors[:, dx+2] .+ lattice_vectors[:, dy+2]
        end
    end

    hoppings = hopping_model(cell, atom_positions, i, j, di, dj, atom_types) / 2.0  # Divide by 2 since we are double counting every pair
    kpoint = reshape(kpoint, (1, 1, 3))
    phases = sum(kpoint .* disp_table,dims=3)
    phases = dropdims(phases,dims=3)
    Ham[i,j] = hoppings .* exp(1im.* phases)
    return Ham
end

function get_helem_fxn(r2,cell,typei,typen,params,kpoint)
    #function to get hamiltonian matrix element, write function just in terms of r1 
    #so that we can take derivative wrt to r1
    function helem(r1)
	    disp = wrap_disp(r1,r2, cell)
	    dist = norm(disp)
        #just in case neighbor list cutoff is larger than hopping cutoff
        if dist < params[typei][typen]["rcut"] && dist > 0.5
            #phase = cos(dot(kpoint,disp))
            hop = params[typei][typen]["hopping"](disp)
            return hop #.* phase
        else
            return 0
        end
    end
    return helem
end

function diagH(matrix,device_type,device_num)
    if device_type=="gpu"
        #CuArrays.allowscalar(false)  # Disallow scalar operations on GPU
        #device = CuDevice(device_num)
        #CuArrays.init(device)
        
        #gpu_matrix = CuArray(matrix)
        gpu_matrix = CuSparseMatrixCSR(matrix)
        eigenvalues, eigenvectors = eigen(gpu_matrix)
        cpu_eigenvalues = Array(eigenvalues)
        cpu_eigenvectors = Array(eigenvectors)
        return cpu_eigenvalues,cpu_eigenvectors

    else
        matrix = Hermitian(ComplexF64.(Matrix(matrix)))
        eigdata =  eigen(matrix)
        return eigdata.values,eigdata.vectors
    end
end

function get_hellman_feynman_fd(atom_positions,neighbor_list,atom_types,
                                cell,eigvec,kpoint,params,device_type="cpu",device_num=1)
    dr=1e-4
    natoms = size(atom_positions)[1]
    nocc = trunc(Int,size(eigvec)[1]/2)
    Forces = zeros(natoms,3)
    for i in 1:natoms
        for dir_ind in [1 2 3] 
            atom_positions_pert = copy(atom_positions)
            atom_positions_pert[i,dir_ind] += dr  
            Ham,Overlap = gen_ham_ovrlp(atom_positions_pert, neighbor_list,atom_types,cell, kpoint,params)
            eigvalues,eigvectors = diagH(Ham,device_type,device_num)
            Energy_up = 2*sum(eigvalues[begin:nocc])

            atom_positions_pert = copy(atom_positions)
            atom_positions_pert[i,dir_ind] -= dr  
            Ham,Overlap = gen_ham_ovrlp(atom_positions_pert, neighbor_list,atom_types,cell, kpoint,params)
            eigvalues,eigvectors = diagH(Ham,device_type,device_num)
            Energy_dwn = 2*sum(eigvalues[begin:nocc])

            Forces[i,dir_ind] = -(Energy_up - Energy_dwn)/2/dr
        end
    end
    return Forces
end

function get_hellman_feynman(atom_positions,neighbor_list,atom_types,
                            cell,eigvec,kpoint,params)
    #give eigenvector at 1 kpoint, higher up fxn will average over Kpoints
    natoms = size(atom_positions)[1]
    nocc = trunc(Int,size(eigvec)[1]/2)
    Force = zeros(ComplexF64,natoms,3)
    for i in 1:natoms
        neighbors = neighbor_list[i]
        for n in neighbors
            r1 = atom_positions[i,begin:end]
            r2 = atom_positions[n,begin:end] 
            typei = atom_types[i]
            typen = atom_types[n]
            helem_fxn = get_helem_fxn(r2,cell,typei,typen,params,kpoint)
            gradH = ForwardDiff.gradient(helem_fxn, r1)
            rho = dot(eigvec[n,begin:nocc],conj(eigvec[i,begin:nocc]))
	        disp = wrap_disp(r1,r2, cell)
            dist = norm(disp)
            phase = exp(1im*dot(kpoint,disp))
            # *2 for doubly occupied states
	        ave_gradH = 4 .* gradH  .* rho .* phase #sqrt(rho .* phase .* conj(rho .* phase)) # .* abs.(disp./dist)
            Force[i,1] -= ave_gradH[1] 
            Force[i,2] -= ave_gradH[2] 
            Force[i,3] -= ave_gradH[3] 
        end
    end
    return Force
end