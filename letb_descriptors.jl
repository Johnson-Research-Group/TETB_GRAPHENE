using DataFrames

function nnmat(lattice_vectors, atomic_basis)
    """
    Build matrix which tells you relative coordinates
    of nearest neighbors to an atom i in the supercell

    Returns: nnmat [natom x 3 x 3]
    """
    nnmat = zeros(Float64, length(atomic_basis), 3, 3)

    # Extend atom list
    atoms = []
    for i in [0, -1, 1]
        for j in [0, -1, 1]
            displaced_atoms = atomic_basis .+ i .* lattice_vectors[1, :] .+ j .* lattice_vectors[2, :]
            atoms = vcat(atoms, displaced_atoms)
        end
    end
    atoms = convert(Array{Float64, 2}, atoms)
    atomic_basis = convert(Array{Float64, 2}, atomic_basis)

    # Loop
    for i in 1:length(atomic_basis)
        displacements = atoms .- atomic_basis[i, :]
        distances = NNlib.norm(displacements, dims=2)
        ind = sortperm(vec(distances))
        nnmat[i, :, :] = displacements[ind[2:4], :]'
    end

    return nnmat
end

function ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj)
    """ 
    Converts displacement indices to physical distances
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    dxy - Distance in Bohr, projected in the x/y plane
    dz  - Distance in Bohr, projected onto the z axis
    """
    num_neighbors = size(aj)[1]
    displacement_vector = zeros(num_neighbors,3)
    dxy = zeros(num_neighbors)
    dz = zeros(num_neighbors)
    for i in 1:num_neighbors
        displacement_vector[i,:] = di[i] .* reshape(lattice_vectors[:, 1],(3,1)) .+ dj[i] .* reshape(lattice_vectors[:, 2],(3,1)) .+ atomic_basis[aj[i], :] .- atomic_basis[ai[i], :]
        dxy[i] = norm(displacement_vector[i, 1:2])
        dz[i] =  abs.(displacement_vector[i,3])
    end
    return dxy, dz
end

function ix_to_disp(lattice_vectors, atomic_basis, di, dj, ai, aj)
    displacement_vector = di' .* reshape(lattice_vectors[:, 1],(1,3)) .+ dj' .* reshape(lattice_vectors[:, 2],(1,3)) .+ atomic_basis[aj, :] .- atomic_basis[ai, :]
    return displacement_vector
end

function partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj)
    """
    Given displacement indices and geometry,
    get indices for partitioning the data
    """
    # First find the smallest distance in the lattice -> reference for NN 
    distances = ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj)
    distances = sqrt.(distances[1].^2 .+ distances[2].^2)
    min_distance = minimum(distances)

    # NN should be within 5% of min_distance
    t01_ix = (0.95 * min_distance .<= distances .<= 1.05 * min_distance)

    # NNN should be withing 5% of sqrt(3)x of min_distance
    t02_ix = (0.95 * sqrt(3) * min_distance .<= distances .<= 1.05 * sqrt(3) * min_distance)

    # NNNN should be within 5% of 2x of min_distance
    t03_ix = (0.95 * 2 * min_distance .<= distances .<= 1.05 * 2 * min_distance)

    # Anything else, we zero out
    t00 = (distances .< 0.95 * min_distance) .| (distances .> 1.05 * 2 * min_distance)

    return t01_ix, t02_ix, t03_ix, t00
end

function triangle_height(a, base)
    """
    Give area of a triangle given two displacement vectors for 2 sides
    """
    area = abs(det(hcat(a, base, [1.0, 1.0, 1.0])))
    height = 2 * area / NNlib.norm(base)
    return height
end

function t01_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj)
    # Compute NN distances
    r = di' .* lattice_vectors[:, 1] .+ dj' .* lattice_vectors[:, 2] .+
        atomic_basis[aj, :] .- atomic_basis[ai, :] # Relative coordinates
    a = NNlib.norm(r, dims=2)
    return DataFrame(a=a)
end

function t02_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj)
    # Compute NNN distances
    r = di' .* lattice_vectors[:, 1] .+ dj' .* lattice_vectors[:, 2] .+
        atomic_basis[aj, :] .- atomic_basis[ai, :] # Relative coordinates
    b = NNlib.norm(r, dims=2)

    # Compute h
    h1 = Float64[]
    h2 = Float64[]
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in 1:size(r, 1)
        nn = mat[aj[i], :, :] + r[i, :]'
        nn[:, 3] .= 0.0 # Project into xy-plane
        nndist = NNlib.norm(nn, dims=2)
        ind = sortperm(vec(nndist))
        push!(h1, triangle_height(nn[ind[1], :], r[i, :]'))
        push!(h2, triangle_height(nn[ind[2], :], r[i, :]'))
    end
    return DataFrame(h1=h1, h2=h2, b=b)
end

function t03_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj)
    """
    Compute t03 descriptors
    """
    # Compute NNNN distances
    r = di' .* lattice_vectors[:, 1] .+ dj' .* lattice_vectors[:, 2] .+
        atomic_basis[aj, :] .- atomic_basis[ai, :] # Relative coordinates
    c = NNlib.norm(r, dims=2)

    # All other hexagon descriptors
    l = Float64[]
    h = Float64[]
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in 1:size(r, 1)
        nn = mat[aj[i], :, :] + r[i, :]'
        nn[:, 3] .= 0.0 # Project into xy-plane
        nndist = NNlib.norm(nn, dims=2)
        ind = sortperm(vec(nndist))
        b = nndist[ind[1]]
        d = nndist[ind[2]]
        h3 = triangle_height(nn[ind[1], :], r[i, :]')
        h4 = triangle_height(nn[ind[2], :], r[i, :]')

        nn = r[i, :]' - mat[ai[i], :, :]
        nn[:, 3] .= 0.0 # Project into xy-plane
        nndist = NNlib.norm(nn, dims=2)
        ind = sortperm(vec(nndist))
        a = nndist[ind[1]]
        e = nndist[ind[2]]
        h1 = triangle_height(nn[ind[1], :], r[i, :]')
        h2 = triangle_height(nn[ind[2], :], r[i, :]')

        push!(l, (a + b + d + e) / 4)
        push!(h, (h1 + h2 + h3 + h4) / 4)
    end
    return DataFrame(c=c, h=h, l=l)
end

function descriptors_intralayer(lattice_vectors, atomic_basis, di, dj, ai, aj)
    """ 
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    # Partition 
    partition = partition_tb(lattice_vectors, atomic_basis, di, dj, ai, aj)
    
    # Compute descriptors
    t01 = t01_descriptors(lattice_vectors, atomic_basis, di[partition[1]], dj[partition[1]], ai[partition[1]], aj[partition[1]])
    t02 = t02_descriptors(lattice_vectors, atomic_basis, di[partition[2]], dj[partition[2]], ai[partition[2]], aj[partition[2]])
    t03 = t03_descriptors(lattice_vectors, atomic_basis, di[partition[3]], dj[partition[3]], ai[partition[3]], aj[partition[3]])
    return t01, t02, t03
end


function ix_to_orientation(lattice_vectors, atomic_basis, di, dj, ai, aj)
    """
    Converts displacement indices to orientations of the 
    nearest neighbor environments using definitions in 
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    theta_12 - Orientation of upper-layer relative to bond length
    theta_21 - Orientation of lower-layer relative to bond length
    """
    displacement_vector = di' .* lattice_vectors[:, 1] .+ dj' .* lattice_vectors[:, 2] .+
                          atomic_basis[aj, :] .- atomic_basis[ai, :]
    mat = descriptors_intralayer.nnmat(lattice_vectors, atomic_basis)

    # Compute distances and angles
    theta_12 = Float64[]
    theta_21 = Float64[]
    for (disp, i, j, inn, jnn) in zip(eachcol(displacement_vector), ai, aj, eachrow(mat[ai, :, :]), eachrow(mat[aj, :, :]))
        sin_jnn = cross(jnn[1:2], disp[1:2])
        sin_inn = cross(inn[1:2], disp[1:2])
        cos_jnn = dot(jnn[1:2], disp[1:2])
        cos_inn = dot(inn[1:2], disp[1:2])
        theta_jnn = atan(sin_jnn, cos_jnn)
        theta_inn = atan(sin_inn, cos_inn)

        push!(theta_12, π - theta_jnn)
        push!(theta_21, theta_inn)
    end
    return theta_12, theta_21
end

function descriptors_interlayer(lattice_vectors, atomic_basis, di, dj, ai, aj)
    """
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    
    output = Dict(
        "dxy" => Float64[],  # Distance in Bohr, xy plane
        "dz" => Float64[],   # Distance in Bohr, z
        "d" => Float64[],    # Distance in Bohr 
        "theta_12" => Float64[],  # Orientation of upper layer NN environment
        "theta_21" => Float64[]   # Orientation of lower layer NN environment
    )

    # 1-body terms
    dist_xy, dist_z = descriptors_intralayer.ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj)
    dist = sqrt.(dist_z .^ 2 .+ dist_xy .^ 2)
    append!(output["dxy"], dist_xy)
    append!(output["dz"], dist_z)
    append!(output["d"], dist)

    # Many-body terms
    theta_12, theta_21 = ix_to_orientation(lattice_vectors, atomic_basis, di, dj, ai, aj)
    append!(output["theta_12"], theta_12)
    append!(output["theta_21"], theta_21)
   
    # Return DataFrame
    df = DataFrame(output)
    return df
end
