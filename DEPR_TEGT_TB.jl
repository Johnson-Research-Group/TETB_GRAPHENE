using Statistics
using Plots
using Distributed
@everywhere using LinearAlgebra

include("Lammps_CorrPot.jl")
include("lammps_io.jl")
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

function get_recip_cell(cell::Matrix{Float64})
    a1 = cell[:, 1]
    a2 = cell[:, 2]
    a3 = cell[:, 3]

    volume = dot(a1, cross(a2, a3))
    
    b1 = 2π * cross(a2, a3) / volume
    b2 = 2π * cross(a3, a1) / volume
    b3 = 2π * cross(a1, a2) / volume

    return [b1 b2 b3]
end


function get_forces_energy(atoms,kpoints,params,rcut = 10,data_file="tegt.data")
    @everywhere atom_positions = $atoms["positions"]
    @everywhere atom_types = $atoms["symbols"]
    @everywhere cell = $atoms["cell"]
    recip_cell = get_recip_cell(cell)
    kpoints = kpoints * recip_cell
    @everywhere kpoints = $kpoints
    @everywhere rcut = $rcut
    @everywhere neighbor_list = $get_neighbor_list(atom_positions,cell,rcut)
    natoms = size(atom_positions)[1]
    nkp = size(kpoints)[1]
    @everywhere EnergyK = zeros($nkp)
    @everywhere ForcesK = zeros($natoms,3,$nkp)

    ncpu = nprocs()
    ndiv = trunc(Int,nkp/ncpu)
    #@distributed for k in 1:ndiv:nkp
    @distributed vcat for k in 1:nkp
    #for k in 1:nkp #parallelize this loop
        Ham,Overlap = gen_ham_ovrlp(atom_positions, neighbor_list,
            atom_types,cell, kpoints[k,begin:end],params)
        eigdata = eigen(Ham) #,Overlap)
        nocc = trunc(Int,natoms/2)
        evals = eigdata.values
        EnergyK[k] = sum(eigdata.values[begin:end])
        tmp_forces = get_hellman_feynman_fd(atom_positions,neighbor_list,atom_types,cell, eigdata.vectors,kpoints[k,:],params)
        ForcesK[begin:end,begin:end,k] = tmp_forces
    end
    #corrective potential calculated from lammps
    Energy = mean(EnergyK)
    Forces = mean(ForcesK,dims=3)
    write_lammps_data(data_file,atoms)
    corr_forces,corr_energies = CorrectivePotential(data_file)
    Energy += corr_energies
    Forces += corr_forces

    #for i in 1:natoms
    #    nneighbors = size(neighbor_list[i])
    #    for j in nneighbors
    #        Local_energy, local_force = params.potential(i, neighbors[i],atom_types[neighbors[i]])
    #        Energy += Local_energy
    #        Force[neighbors[i],:] += local_force
    #    end
    #end
    return Energy, Forces
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


function fire_minimization(atoms,params, kpoints=[[0,0,0]], dt=0.1, dt_max=1.0, 
    alpha_start=0.1,f_inc=1.1, f_dec=0.5, alpha_scale=0.1,N_min=1, N_max=2,Ftol=1e-9)

    """Perform energy minimization using the FIRE (Fast Inertial Relaxation Engine) algorithm.

    # Arguments
    - 'atoms_object' : mutable object containing attributes:
        * `positions`: An array of initial particle positions.
        * `atom_types`: types of atoms corresponding to indices in `positions`
        * `cell`: ndim x ndim array of cell vectors [vector 1;vector2;vector 3]
    - `params`:dictionary including MD potential and tb parameters
    - `kpoints`: array of momentum vectors to calculate hamiltonians over 
    - `dt`: The time step for integration.
    - `dt_max`: Maximum allowed time step.
    - `N_min`: Number of consecutive steps with energy increase before decreasing `dt`.
    - `N_max`: Maximum number of iterations.
    - `alpha_start`: Initial value of the parameter `alpha` (dt scaling factor).
    - `f_inc`: Factor by which `alpha` is increased.
    - `f_dec`: Factor by which `alpha` is decreased.
    - `alpha_scale`: Factor by which `dt` is scaled during each step.
    - `Ftol`: force tolerance. If average forces are less than Ftol, stop

    This function performs energy minimization using the FIRE algorithm with a Velocity Verlet integrator.
    """
    positions = atoms["positions"]
    atom_types = atoms["symbols"]
    cell = atoms["cell"]
    kpoints = hcat(kpoints...)'
    velocities = zeros(size(positions))
    Energy, forces = get_forces_energy(atoms,kpoints,params)
    aveF = mean(forces)
    alpha = alpha_start
    dt_current = dt
    
    N_iterations = 0
    N_consecutive = 0
    prev_energy = sum(0.5 .* velocities.^2)
    
    while (N_iterations < N_max)
        velocities += forces .* dt_current
        positions += velocities .* dt_current
        
        new_energy, new_forces = get_forces_energy(atoms,kpoints,params)
        aveF = mean(norm.(new_forces))
        #println("Energy, <Force> ",new_energy," ",aveF)
        if new_energy > prev_energy
            N_consecutive += 1
            if N_consecutive >= N_min
                dt_current = min(dt_current * alpha_scale, dt_max)
                velocities = zeros(size(positions))
                N_consecutive = 0
            end
        else
            N_consecutive = 0
            dt_current = min(dt_current * alpha_scale, dt_max)
            alpha *= f_dec
            velocities += 0.5 * (forces + new_forces) .* dt_current
            forces = new_forces
        end
        
        if N_iterations % 1 == 0
            println("Iteration: $N_iterations, Energy: $new_energy, <F>: $aveF")
        end
        
        if aveF < Ftol
            break
        end
        
        N_iterations += 1
        prev_energy = new_energy
    end
end
    
function get_band_structure(atoms,params,kdat,rcut=10)
    @everywhere atom_positions = $atoms["positions"]
    @everywhere atom_types = $atoms["symbols"]
    @everywhere cell = $atoms["cell"]
    kpoints,kdist,nodes = kdat
    recip_cell = get_recip_cell(cell)
    kpoints = kpoints * recip_cell
    @everywhere kpoints = $kpoints
    @everywhere rcut = $rcut
    @everywhere neighbor_list = $get_neighbor_list(atom_positions,cell,rcut)
    natoms = size(atom_positions)[1]
    nkp = size(kpoints)[1]
    @everywhere evals = zeros($natoms,$nkp)
    @distributed vcat for k in 1:nkp
        Ham,Overlap = gen_ham_ovrlp(atom_positions, neighbor_list,
            atom_types,cell, kpoints[k,begin:end],params)
        eigdata = eigen(Ham) #,Overlap)
        evals[begin:end,k] = eigdata.values
    end
    nocc = trunc(Int,natoms/2)
    fermi_energy = mean([minimum(evals[nocc+1,begin:end]),maximum(evals[nocc,begin:end])])    
    evals .-= fermi_energy

    plt = plot(legend = false)
    # Plot each band
    for band in 1:natoms
        plot!(plt, kdist, evals[band, :], lw = 2, linecolor = :black)
    end
    labels = ["K", "Γ", "M", "K"]
    if !isempty(labels)
        vline!(plt, nodes, label = labels)
        xticks!(plt, nodes, labels )
    end
    ylims!(-2, 2)
    xlabel!(plt,"kpath")
    ylabel!(plt,"Energy (eV)")
    title!(plt,"Eigenvalue Spectrum")
    display(plt)
end

@everywhere params = Dict("C1"=>Dict("C1"=>Dict("hopping"=>hoppingIntra,"ovrlp"=>overlapIntra,"self_energy"=>-5.2887,"rcut"=>3.7), 
                         "C2"=>Dict("hopping"=>hoppingInter,"ovrlp"=>overlapInter,"rcut"=>5.29)),
              "C2"=>Dict("C1"=>Dict("hopping"=>hoppingInter,"ovrlp"=>overlapInter,"rcut"=>5.29), 
                         "C2"=>Dict("hopping"=>hoppingIntra,"ovrlp"=>overlapIntra,"self_energy"=>-5.2887,"rcut"=>3.7) ))

twist_angle = 9.43
atoms = get_tBLG_atoms(twist_angle)

relaxation = true
bands = false
if relaxation
    fire_minimization(atoms,params)
end

if bands
    nkp = 15
    high_sym_pts = [2/3 1/3 0;
                    0   0   0;
                    1/2 0   0;
                    2/3 1/3 0]
    kdat = generate_kpoint_path(high_sym_pts, nkp)
    get_band_structure(atoms,params,kdat)
end