using LinearAlgebra
using ForwardDiff
#using CUDA
using SparseArrays
include("letb_model.jl")

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


function get_disp_table(atom_positions, cell, cutoff_radius)
    natoms= size(atom_positions)[1]
    disp_table = [spzeros(Float64,natoms,natoms),spzeros(Float64,natoms,natoms),spzeros(Float64,natoms,natoms)]
    dist_table = spzeros(Float64,natoms,natoms)
    for i in 1:natoms

        for j in 1:natoms
            disp = wrap_disp(atom_positions[i,begin:end],atom_positions[j,begin:end], cell)
            dist = norm(disp)
            if i != j && dist <= cutoff_radius
                disp_table[1][i,j] = disp[1]
                disp_table[2][i,j] = disp[2]
                disp_table[3][i,j] = disp[3]
                dist_table[i,j] = dist
            end
        end
    end
    
    return disp_table, dist_table
end

function gen_ham(atom_positions,atom_types,cell,kpoint,params,rcut=10)
    disp_table, dist_table = get_disp_table(atom_positions,cell,rcut)
    kpoint = reshape(kpoint, (1, 1, 3))
    phases = exp.(-1im .* (kpoint[1] .* disp_table[1] + kpoint[2] .* disp_table[2] + kpoint[3] .* disp_table[3]))
    ham_fxn = get_Ham_fxn(atom_types,cell,kpoint,params)
    Ham = ham_fxn(atom_positions) .* phases
    return Hermitian(Ham)
end

function get_Ham_fxn(atom_types,cell,kpoint,params)
    function ham_fxn(atom_positions)
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
        rcut = 10
        angstroms_to_bohr = 1.88973
        atom_positions *= angstroms_to_bohr
        cell *= angstroms_to_bohr
        #cell=cell[begin:2,begin:2]
        kpoint /= angstroms_to_bohr
        models_functions = Dict("letb"=>letb,"mk"=>mk,"popov"=>popov)
        hopping_model = models_functions[params]
        natom = size(atom_positions, 1)
        di = Int[]
        dj = Int[]
        
        extended_coords = []
    
        for dx in [-1, 0, 1], dy in [-1, 0, 1]
            translate_coords = (atom_positions .+ (dx .* reshape(cell[1, :],(1,3)))) .+ (dy .* reshape(cell[2, :],(1,3)))
            push!(extended_coords,translate_coords) 
            append!(di, fill(dx, natom))
            append!(dj, fill(dy, natom))
        end
        extended_coords = vcat(extended_coords...)
        distances = [norm(atom_positions[i, :] - extended_coords[j, :]) for i in 1:natom, j in 1:size(extended_coords, 1)]
        indices = findall(0 .< distances .< rcut)  # 10 Bohr cutoff
        indi = [coord[1] for coord in indices]
        indj = [coord[2] for coord in indices]

        di = di[indj]
        dj = dj[indj]
        i = indi
        j = indj .% natom .+ 1
        hoppings = hopping_model(cell, atom_positions, i, j, di, dj, atom_types) / 2.0  # Divide by 2 since we are double counting every pair
        Ham = spzeros(ComplexF64,natom,natom)
        for (index,h) in enumerate(hoppings)
            Ham[i[index],j[index]] = h
            Ham[j[index],i[index]] = h
        end
        return Ham
    end
end

function get_hellman_feynman_fd(atom_positions,atom_types,cell,eigvec,kpoint,params,device_type="cpu",device_num=1)
    dr=1e-4
    natoms = size(atom_positions)[1]
    nocc = trunc(Int,size(eigvec)[1]/2)
    Forces = zeros(natoms,3)
    for i in 1:natoms
        for dir_ind in [1 2 3] 
            atom_positions_pert = copy(atom_positions)
            atom_positions_pert[i,dir_ind] += dr  
            Ham = gen_ham(atom_positions,atom_types,cell,kpoint,params)
            eigvalues,eigvectors = diagH(Ham,device_type,device_num)
            Energy_up = 2*sum(eigvalues[begin:nocc])

            atom_positions_pert = copy(atom_positions)
            atom_positions_pert[i,dir_ind] -= dr  
            Ham = gen_ham(atom_positions,atom_types,cell,kpoint,params)
            eigvalues,eigvectors = diagH(Ham,device_type,device_num)
            Energy_dwn = 2*sum(eigvalues[begin:nocc])

            Forces[i,dir_ind] = -(Energy_up - Energy_dwn)/2/dr
        end
    end
    return Forces
end

function get_hellman_feynman(atom_positions,atom_types,cell,eigvec,kpoint,params,device_num=1)
    nbands=  size(eigvec)[2]
    natoms = atom_positions[1]
    n_i = zeros(nbands)
    n_i[1:nbands ÷ 2] .= 1
    density_matrix = conj(eigvec)' * Diagonal(n_i) * eigvec
    density_matrix = reshape(density_matrix,nbands,nbands,1,1)
    ham_fxn = get_Ham_fxn(atom_types,cell,kpoint,params)
    gradH_all = ForwardDiff.gradient(ham_fxn, atom_positions)

    kpoint = reshape(kpoint, 1, 1, 1, 3)
    disp_table, dist_table = get_disp_table(atom_positions,cell,rcut)
    phases = exp.(-1im .* (kpoint[1] .* disp_table[1] + kpoint[2] .* disp_table[2] + kpoint[3] .* disp_table[3]))
    phases = reshape(phases,natoms,natoms,1,1)
    # (NxNxNX3) .* (1xNxNX3) .* (NxNx1x1)  .* (NxNx1x1) 
    Forces = sum(gradH_all .* phases .* density_matrix,dims=[1,2])
    return Forces
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