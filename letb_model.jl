
using HDF5
using LinearAlgebra
include("letb_descriptors.jl")

function moon(r, a::Float64, b::Float64, c::Float64)
    """
    Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012)
    """
    d, dz = r
    return a .* exp.(-1 .*b .* (d .- 2.68)) .* (1 .- (dz ./ d).^2) + c .* exp.(-1 .* b .* (d .- 6.33)) .* (dz ./ d).^2
end

function fang(rvec::Tuple{Float64, Float64, Float64}, a0::Float64, b0::Float64, c0::Float64, 
              a3::Float64, b3::Float64, c3::Float64, 
              a6::Float64, b6::Float64, c6::Float64, d6::Float64)
    """
    Parameterization from Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)
    """
    r, theta12, theta21 = rvec
    r = r ./ 4.649

    function v0(x, a, b, c)
        return a .* exp.(-1 .*b .*  x .^2) .* cos.(c .* x)
    end

    function v3(x, a, b, c)
        return a .* x.^2 .* exp.(-1 .* b .* (x .- c).^2)
    end

    function v6(x, a, b, c, d)
        return a .* exp.(-1 .* b .* (x .- c).^2) .* sin.(d .* x)
    end

    f = v0(r, a0, b0, c0)
    f += v3(r, a3, b3, c3) * (cos(3 * theta12) + cos(3 * theta21))
    f += v6(r, a6, b6, c6, d6) * (cos(6 * theta12) + cos(6 * theta21))
    return f
end

function load_intralayer_fit()
    # Load in fits, average over k-folds
    fit = Dict{String, Array{Float64, 1}}()
    f = join(splitpath(dirname(dirname(pathof(bilayer_letb))), "/")[1:end-1], "/") * "/parameters/fit_intralayer.hdf5"
    hdf = h5open(f, "r")
    fit["t01"] = mean(hdf["t01"]["parameters_test"][:, :], dims=1)[:]
    fit["t02"] = mean(hdf["t02"]["parameters_test"][:, :], dims=1)[:]
    fit["t03"] = mean(hdf["t03"]["parameters_test"][:, :], dims=1)[:]
    close(hdf)
    return fit
end

function load_interlayer_fit()
    # Load in fits, average over k-folds
    fit = Dict{String, Array{Float64, 1}}()
    f = join(splitpath(dirname(dirname(pathof(bilayer_letb))), "/")[1:end-1], "/") * "/parameters/fit_interlayer.hdf5"
    hdf = h5open(f, "r")
    fit["fang"] = mean(hdf["fang"]["parameters_test"][:, :], dims=1)[:]
    close(hdf)
    return fit
end

function intralayer(lattice_vectors, atomic_basis, i, j, di, dj)
    """
    Our model for single layer intralayer
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for intralayer in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj in eV
    """
    # Extend lattice_vectors to (3 x 3) for our descriptors, the third lattice vector is arbitrary
    latt_vecs = vcat(lattice_vectors, [0.0, 0.0, 0.0])
    atomic_basis = convert(Array{Float64, 2}, atomic_basis)
    i = convert(Array{Int, 1}, i)
    j = convert(Array{Int, 1}, j)
    di = convert(Array{Int, 1}, di)
    dj = convert(Array{Int, 1}, dj)

    # Get the descriptors for the fit models
    partition = partition_tb(lattice_vectors, atomic_basis, di, dj, i, j)
    descriptors = descriptors_intralayer(lattice_vectors, atomic_basis, di, dj, i, j)

    # Get the fit model parameters
    fit = load_intralayer_fit()

    # Predict hoppings
    t01 = dot(descriptors[1], fit["t01"][2:end]) + fit["t01"][1]
    t02 = dot(descriptors[2], fit["t02"][2:end]) + fit["t02"][1]
    t03 = dot(descriptors[3], fit["t03"][2:end]) + fit["t03"][1]

    # Reorganize
    hoppings = zeros(Float64, length(i))
    hoppings[partition[1]] .= t01
    hoppings[partition[2]] .= t02
    hoppings[partition[3]] .= t03
    hoppings[partition[4]] .= 0.0

    return hoppings
end

function letb(lattice_vectors, atomic_basis, i, j, di, dj, layer_types)
    """
    Our model for bilayer intralayer
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for intralayer in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    # Extend lattice_vectors to (3 x 3) for our descriptors, the third lattice vector is arbitrary
    latt_vecs = vcat(lattice_vectors, [0.0, 0.0, 0.0])
    atomic_basis = convert(Array{Float64, 2}, atomic_basis)
    i = convert(Array{Int, 1}, i)
    j = convert(Array{Int, 1}, j)
    di = convert(Array{Int, 1}, di)
    dj = convert(Array{Int, 1}, dj)
    natoms = length(atomic_basis)
    for index in i
        if index==0
            println(index)
        end
    end
    # Get the bi-layer descriptors 
    descriptors = descriptors_interlayer(lattice_vectors, atomic_basis, di, dj, i, j)
    
    # Partition the intra- and inter-layer hoppings indices 
    if isa(layer_types, Vector{Int}) || isa(layer_types, Vector{Int64})
        npairs = size(di)[1]
        interlayer = fill(false, npairs)
        for n = 1:npairs
            if layer_types[i[n]] != layer_types[j[n]]
                interlayer[n] = true
            end
        end
    else
        interlayer = (descriptors["dz"] .> 1.0)  # Allows for buckling, doesn't work for large corrugation
    end
    
    # Compute the inter-layer hoppings
    fit = load_interlayer_fit()
    X = convert(Matrix{Float64}, descriptors[!, ["dxy", "theta_12", "theta_21"]])
    interlayer_hoppings = fang(X', fit["fang"]...)

    # Compute the intra-layer hoppings
    intralayer_hoppings = intralayer(lattice_vectors, atomic_basis, 
                                     i[.!interlayer], j[.!interlayer], 
                                     di[.!interlayer], dj[.!interlayer])

    # Reorganize
    hoppings = zeros(Float64, length(i))
    hoppings[interlayer] .= interlayer_hoppings
    hoppings[.!interlayer] .= intralayer_hoppings

    return hoppings
end

function mk(lattice_vectors, atomic_basis, i, j, di, dj, layer_types)
    """
    Moon model for bilayer graphene - Moon and Koshino, PRB 85 (2012)
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for intralayer in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    lattice_vectors = convert(Array{Float64, 2}, lattice_vectors)
    atomic_basis = convert(Array{Float64, 2}, atomic_basis)
    i = convert(Array{Int, 1}, i)
    j = convert(Array{Int, 1}, j)
    di = convert(Array{Int, 1}, di)
    dj = convert(Array{Int, 1}, dj)
    dxy, dz = ix_to_dist(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = moon((sqrt.(dz.^2 + dxy.^2), dz), -2.7, 1.17, 0.48)
    return hoppings
end

function popov(lattice_vectors, atomic_basis, i, j, di, dj, layer_types)
    """
    popov/porezag model for bilayer graphene - Moon and Koshino, PRB 85 (2012)
    Input: 
        lattice_vectors - float (nlat x 3) where nlat = 2 lattice vectors for intralayer in BOHR
        atomic_basis    - float (natoms x 3) where natoms are the number of atoms in the computational cell in BOHR
        i, j            - int   (n) list of atomic bases you are hopping between
        di, dj          - int   (n) list of displacement indices for the hopping
    Output:
        hoppings        - float (n) list of hoppings for the given i, j, di, dj
    """
    lattice_vectors = convert(Array{Float64, 2}, lattice_vectors)
    atomic_basis = convert(Array{Float64, 2}, atomic_basis)
    i = convert(Array{Int, 1}, i)
    j = convert(Array{Int, 1}, j)
    di = convert(Array{Int, 1}, di)
    dj = convert(Array{Int, 1}, dj)
    if isa(layer_types, Vector{Int}) || isa(layer_types, Vector{Int64})
        npairs = size(di)[1]
        interlayer = fill(false, npairs)
        for n = 1:npairs
            if layer_types[i[n]] != layer_types[j[n]]
                interlayer[n] = true
            end
        end
    end
    disp_vector = ix_to_disp(lattice_vectors, atomic_basis, di, dj, i, j)
    hoppings = zeros(Float64, length(i))
    hoppings[interlayer] .= popov_hopping(disp_vector)
    hoppings[.!interlayer] .= porezag_hopping(disp_vector)

    return hoppings
end