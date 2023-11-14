using Statistics
#using Plots
using Distributed
using LinearAlgebra

include("tb_parameters.jl")
include("TB_Utils.jl")

function get_neighbor_list(positions, cell,cutoff_radius)
    num_particles = size(positions, 1)
    neighbor_list = Dict{Int, Array{Int}}()  # Dict to store neighbors
    
    for i in 1:num_particles
        neighbors = Array{Int64}(undef,0)
        for j in 1:num_particles
            disp = wrap_disp(positions[i,begin:end],positions[j,begin:end], cell)
            dist = norm(disp)
            if i != j && dist <= cutoff_radius
                append!(neighbors, j)
            end
        end
        neighbor_list[i] = neighbors
    end
    
    return neighbor_list
end

function get_recip_cell(cell)
    a1 = cell[:, 1]
    a2 = cell[:, 2]
    a3 = cell[:, 3]

    volume = dot(a1, cross(a2, a3))
    
    b1 = 2π * cross(a2, a3) / volume
    b2 = 2π * cross(a3, a1) / volume
    b3 = 2π * cross(a1, a2) / volume

    return [b1 b2 b3]
end

function wrap_disp(r1, r2, cell)
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

function generate_kpoint_path(high_symmetry_points::Matrix{Float64}, num_points_between::Int)
    num_high_symmetry_points, dim = size(high_symmetry_points)
    
    if num_high_symmetry_points < 2
        throw(ArgumentError("At least two high symmetry points are required."))
    end
    
    kpoint_paths = Matrix{Float64}(undef, ((num_high_symmetry_points-1) * (num_points_between + 1), dim))
    distances = Matrix{Float64}(undef, ((num_high_symmetry_points-1) * (num_points_between + 1),1))
    nodes = zeros(num_high_symmetry_points)
    row_index = 1
    total_distance = 0.0
    nodes[1] = 0
    for i in 1:num_high_symmetry_points - 1
        start_point = high_symmetry_points[i, :]
        end_point = high_symmetry_points[i + 1, :]
        segment_distance = norm(end_point - start_point)
        for j in 0:num_points_between
            if num_points_between > 0
                t = j / num_points_between
            else
                t = 0.0
            end
            
            kpoint = start_point + (end_point - start_point) * t
            kpoint_paths[row_index, :] = kpoint
            distances[row_index] = total_distance + t * segment_distance
            row_index += 1
        end
        
        total_distance += segment_distance
        nodes[i+1] = total_distance
    end
    
    # Add the last high symmetry point
    kpoint_paths[end, :] = high_symmetry_points[end, :]
    distances[end] = total_distance
    
    return kpoint_paths, distances,nodes
end



function generate_monkhorst_pack_grid(nkx::Int, nky::Int, nkz::Int)
    kpoints = []

    for i in 1:nkx
        for j in 1:nky
            for k in 1:nkz
                kx = (i - 1) / nkx
                ky = (j - 1) / nky
                kz = (k - 1) / nkz
                push!(kpoints, [kx, ky, kz])
            end
        end
    end

    return kpoints
end

function JULIA_get_tb_forces_energy(atom_positions,atom_types,cell,kpoints,params_str,device_num,device_type,rcut = 10)
    params = get_param_dict(params_str)
    recip_cell = get_recip_cell(cell)
    if size(kpoints) == (3,)
        kpoints = reshape(kpoints, 1, 3)
    end
    kpoints = kpoints * recip_cell
    nkp = size(kpoints)[1]
    neighbor_list = get_neighbor_list(atom_positions,cell,rcut)
    natoms = size(atom_positions)[1]
    Energy = 0
    Forces = zeros(ComplexF64,natoms,3)
    for k in 1:nkp
        Ham,Overlap = gen_ham_ovrlp(atom_positions, neighbor_list,
            atom_types,cell, kpoints[k,begin:end],params)
        eigvalues,eigvectors = diagH(Ham,device_type,device_num)
        nocc = trunc(Int,natoms/2)
        Energy += 2*sum(eigvalues[begin:nocc])
        Forces += get_hellman_feynman(atom_positions,neighbor_list,atom_types,cell, eigvectors,kpoints[k,:],params)
        
    end
 
    return Energy, Forces
end

function JULIA_get_tb_forces_energy_fd(atom_positions,atom_types,cell,kpoints,params_str,rcut = 10)
    params = get_param_dict(params_str)
    recip_cell = get_recip_cell(cell)
    device_num=1
    device_type="cpu"
    if size(kpoints) == (3,)
        kpoints = reshape(kpoints, 1, 3)
    end
    kpoints = kpoints * recip_cell
    nkp = size(kpoints)[1]
    neighbor_list = get_neighbor_list(atom_positions,cell,rcut)
    natoms = size(atom_positions)[1]
    Energy = 0
    Forces = zeros(ComplexF64,natoms,3)
    for k in 1:nkp
        Ham,Overlap = gen_ham_ovrlp(atom_positions, neighbor_list,
            atom_types,cell, kpoints[k,begin:end],params)
        kpoint  = kpoints[k,:]
        eigvalues,eigvectors = diagH(Ham,device_type,device_num)
        nocc = trunc(Int,natoms/2)
        Energy += 2*sum(eigvalues[begin:nocc])
        Forces += get_hellman_feynman_fd(atom_positions,neighbor_list,atom_types,cell, eigvectors,kpoints[k,:],params)
        
    end
 
    return Energy, Forces
end

function JULIA_get_band_structure(atom_positions,atom_types,cell,kpoints,params_str,device_num,device_type,rcut=10)
    params = get_param_dict(params_str)
    recip_cell = get_recip_cell(cell)
    if size(kpoints) == (3,)
        kpoints = reshape(kpoints, 1, 3)
    end
    kpoints = kpoints * recip_cell
    neighbor_list = get_neighbor_list(atom_positions,cell,rcut)
    natoms = size(atom_positions)[1]
    nkp = size(kpoints)[1]
    evals = zeros(natoms,nkp)
    evecs = zeros(Complex{Float64},natoms,natoms,nkp)
    for k in 1:nkp
        Ham,Overlap = gen_ham_ovrlp(atom_positions, neighbor_list,
            atom_types,cell, kpoints[k,:],params)
        eigvalues,eigvectors = diagH(Ham,device_type,device_num)
        evals[:,k] = eigvalues
        evecs[:,:,k] = eigvectors
    end
    return evals,evecs
end

function get_param_dict(params_str)
    if params_str=="popov"
        params = Dict("B"=>Dict("B"=>Dict("hopping"=>popov_hopping,"ovrlp"=>overlapIntra,"self_energy"=>-5.2887,"rcut"=>3.7), 
                         "Ti"=>Dict("hopping"=>porezag_hopping,"ovrlp"=>overlapInter,"rcut"=>5.29)),
              "Ti"=>Dict("B"=>Dict("hopping"=>porezag_hopping,"ovrlp"=>overlapInter,"rcut"=>5.29), 
                         "Ti"=>Dict("hopping"=>popov_hopping,"ovrlp"=>overlapIntra,"self_energy"=>-5.2887,"rcut"=>3.7) ))

    else #params_str=="nn"
        params = Dict("B"=>Dict("B"=>Dict("hopping"=>nnhop,"ovrlp"=>overlapIntra,"self_energy"=>0,"rcut"=>3), 
                         "Ti"=>Dict("hopping"=>nnhop,"ovrlp"=>overlapInter,"rcut"=>3)),
              "Ti"=>Dict("B"=>Dict("hopping"=>nnhop,"ovrlp"=>overlapInter,"rcut"=>3), 
                         "Ti"=>Dict("hopping"=>nnhop,"ovrlp"=>overlapIntra,"self_energy"=>0,"rcut"=>3) ))
    end
    return params
end

if abspath(PROGRAM_FILE) == @__FILE__
    #write test code
    atom_positions = [0 0 0; 1.42 0 0; 0 1.42 0; 1.42 1.42 0]
    cell = [2.84 0 0; 0 2.84 0 ; 0 0 10]
    atom_types = ["B" "B" "Ti" "Ti"]
    params_str="popov"
    kpoints = [0 0 0]
    println("testing total energy and force calculations")
    Energy, Forces = JULIA_get_tb_forces_energy(atom_positions,atom_types,cell,kpoints,params_str,1,"cpu")
    println("Energy, <Forces> = ",Energy, " ",mean(Forces))

    println("testing band structure calculation")
    evals,evecs = JULIA_get_band_structure(atom_positions,atom_types,cell,kpoints,params_str,1,"cpu")
    println("<evals>,<evecs> = ",mean(evals)," ",mean(evecs))
end
