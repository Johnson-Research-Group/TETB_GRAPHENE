using Base.Iterators: takewhile
using Printf
using LinearAlgebra
#=
function read_lammps_data(filename, Z_of_type::Dict{Int, Int}=Dict(), sort_by_id::Bool=true, units::String="metal", style::String="")
    # Begin read_lammps_data
    # Initialize variables with default values
    fileobj = open(filename,"r")
    file_comment = strip(readline(fileobj))
    natoms = 0
    xlo, xhi = -0.5, 0.5
    ylo, yhi = -0.5, 0.5
    zlo, zhi = -0.5, 0.5
    xy, xz, yz = 0.0, 0.0, 0.0
    
    mass_in = Dict{Int, Float64}()
    vel_in = Dict{Int, Vector{Float64}}()
    bonds_in = Vector{Vector{Int}}()
    angles_in = Vector{Vector{Int}}()
    dihedrals_in = Vector{Vector{Int}}()
    
    sections = [
        "Atoms", "Velocities", "Masses", "Charges", "Ellipsoids", "Lines",
        "Triangles", "Bodies", "Bonds", "Angles", "Dihedrals", "Impropers",
        "Impropers Pair Coeffs", "PairIJ Coeffs", "Pair Coeffs", "Bond Coeffs",
        "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs", "BondBond Coeffs",
        "BondAngle Coeffs", "MiddleBondTorsion Coeffs", "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs", "AngleAngleTorsion Coeffs", "BondBond13 Coeffs",
        "AngleAngle Coeffs"
    ]
    
    header_fields = [
        "atoms", "bonds", "angles", "dihedrals", "impropers", "atom types",
        "bond types", "angle types", "dihedral types", "improper types",
        "extra bond per atom", "extra angle per atom", "extra dihedral per atom",
        "extra improper per atom", "extra special per atom", "ellipsoids",
        "lines", "triangles", "bodies", "xlo xhi", "ylo yhi", "zlo zhi",
        "xy xz yz"
    ]
    
    #sections_re = r"(" * join(sections, "\\s*|") * r")"
    sections_re = join(sections, "\\s*|")
    #header_fields_re = r"(" * join(header_fields, "\\s*|") * r")"
    header_fields_re = join(header_fields, "\\s*|")
    
    section = nothing
    header = true
    
    for line in fileobj
        line_comment = replace(rstrip(takewhile(c -> c != '#', line)), r"\s+$" => "")
        line = replace(rstrip(replace(line, r"#.*" => "")), r"\s+$" => "")
        
        if isempty(line)
            continue  # Skip blank lines
        end
        
        match = match(sections_re, line)
        if match !== nothing
            section = rstrip(match.match)
            header = false
            
            if section == "Atoms"
                # Guess atom_style from the comment after "Atoms" if it exists
                if isnothing(style) && !isempty(line_comment)
                    style = line_comment
                end
                
                # Call _read_atoms_section function
                mol_id_in, type_in, charge_in, pos_in, travel_in = _read_atoms_section(fileobj, natoms, style)
            end
            
            continue
        end
        
        if header
            field, val = match(r"(.*)\s+" * header_fields_re, line).captures
            field, val = strip(field), strip(val)
            
            if !isnothing(field) && !isnothing(val)
                if field == "atoms"
                    natoms = parse(Int, val)
                elseif field == "xlo xhi"
                    xlo, xhi = map(Float64, split(val))
                elseif field == "ylo yhi"
                    ylo, yhi = map(Float64, split(val))
                elseif field == "zlo zhi"
                    zlo, zhi = map(Float64, split(val))
                elseif field == "xy xz yz"
                    xy, xz, yz = map(Float64, split(val))
                end
            end
        end
        
        if !isnothing(section)
            fields = split(line)
            
            if section == "Velocities"
                atom_id = parse(Int, fields[1])
                vel_in[atom_id] = [parse(Float64, fields[_]) for _ in 2:4]
            elseif section == "Masses"
                mass_in[parse(Int, fields[1])] = parse(Float64, fields[2])
            elseif section == "Bonds"
                push!(bonds_in, [parse(Int, fields[_]) for _ in 2:4])
            elseif section == "Angles"
                push!(angles_in, [parse(Int, fields[_]) for _ in 2:5])
            elseif section == "Dihedrals"
                push!(dihedrals_in, [parse(Int, fields[_]) for _ in 2:6])
            end
        end
    end
    
    # Calculate cell
    cell = zeros(Float64, 3, 3)
    cell[1, 1] = xhi - xlo
    cell[2, 2] = yhi - ylo
    cell[3, 3] = zhi - zlo
    cell[2, 1] = xy
    cell[3, 1] = xz
    cell[3, 2] = yz
    
    # Initialize arrays for per-atom quantities
    positions = zeros(Float64, natoms, 3)
    numbers = zeros(Int, natoms)
    ids = zeros(Int, natoms)
    types = zeros(Int, natoms)
    velocities = len(vel_in) > 0 ? zeros(Float64, natoms, 3) : nothing
    masses = len(mass_in) > 0 ? zeros(Float64, natoms) : nothing
    mol_id = len(mol_id_in) > 0 ? zeros(Int, natoms) : nothing
    charge = len(charge_in) > 0 ? zeros(Float64, natoms) : nothing
    travel = len(travel_in) > 0 ? zeros(Int, natoms, 3) : nothing
    bonds = len(bonds_in) > 0 ? fill("_", natoms) : nothing
    angles = len(angles_in) > 0 ? fill("_", natoms) : nothing
    dihedrals = len(dihedrals_in) > 0 ? fill("_", natoms) : nothing
    
    ind_of_id = Dict{Int, Int}()
    
    for (i, atom_id) in enumerate(keys(pos_in))
        ind = sort_by_id ? atom_id : i
        ind_of_id[atom_id] = ind
        
        atom_type = type_in[atom_id]
        positions[ind, :] = pos_in[atom_id]
        
        if !isempty(vel_in)
            velocities[ind, :] = vel_in[atom_id]
        end
        
        if !isempty(travel_in)
            travel[ind, :] = travel_in[atom_id]
        end
        
        if !isempty(mol_id_in)
            mol_id[ind] = mol_id_in[atom_id]
        end
        
        if !isempty(charge_in)
            charge[ind] = charge_in[atom_id]
        end
        
        ids[ind] = atom_id
        types[ind] = atom_type
        
        if isempty(Z_of_type)
            numbers[ind] = atom_type
        else
            numbers[ind] = get(Z_of_type, atom_type, atom_type)
        end
        
        if !isempty(mass_in)
            masses[ind] = mass_in[atom_type]
        end
    end
    
    # Convert units
    #positions = convert_units(positions, "distance", units, "ASE")
    #cell = convert_units(cell, "distance", units, "ASE")
    
    #if !isempty(masses)
    #    masses = convert_units(masses, "mass", units, "ASE")
    #end
    
    #if !isempty(velocities)
    #    velocities = convert_units(velocities, "velocity", units, "ASE")
    #end
    
    # Guess atomic numbers from atomic masses
    #########I don't think this is right########

    atoms = Dict(
        "positions"=>positions,
        "numbers"=>numbers,
        "masses"=>masses,
        "cell"=>cell,
        "pbc"=>[true, true, true],
        "velocities"=>velocities
    )
    
    # Set velocities
    #########I don't think this is right########
    if !isempty(velocities)
        atoms["velocities"] = velocities
    end
    
    atoms["id"] = ids
    atoms["type"] = types
    
    if !isempty(travel)
        atoms["travel"] = travel
    end
    
    if !isempty(mol_id)
        atoms["mol-id"] = mol_id
    end
    
    if !isempty(charge)
        atoms["initial_charges"] = charge
        atoms["mmcharges"] = copy(charge)
    end
    
    if !isempty(bonds)
        for (atom_type, at1, at2) in bonds_in
            i_a1 = ind_of_id[at1]
            i_a2 = ind_of_id[at2]
            
            if !isempty(bonds[i_a1])
                bonds[i_a1] *= ","
            end
            
            bonds[i_a1] *= "$i_a2($atom_type)"
        end
        
        for i in eachindex(bonds)
            if isempty(bonds[i])
                bonds[i] = "_"
            end
        end
        
        atoms["bonds"] = collect(bonds)
    end
    
    if !isempty(angles)
        for (atom_type, at1, at2, at3) in angles_in
            i_a1 = ind_of_id[at1]
            i_a2 = ind_of_id[at2]
            i_a3 = ind_of_id[at3]
            
            if !isempty(angles[i_a2])
                angles[i_a2] *= ","
            end
            
            angles[i_a2] *= "$i_a1-$i_a3($atom_type)"
        end
        
        for i in eachindex(angles)
            if isempty(angles[i])
                angles[i] = "_"
            end
        end
        
        atoms["angles"] = collect(angles)
    end
    
    if !isempty(dihedrals)
        for (atom_type, at1, at2, at3, at4) in dihedrals_in
            i_a1 = ind_of_id[at1]
            i_a2 = ind_of_id[at2]
            i_a3 = ind_of_id[at3]
            i_a4 = ind_of_id[at4]
            
            if !isempty(dihedrals[i_a1])
                dihedrals[i_a1] *= ","
            end
            
            dihedrals[i_a1] *= "$i_a2-$i_a3-$i_a4($atom_type)"
        end
        
        for i in eachindex(dihedrals)
            if isempty(dihedrals[i])
                dihedrals[i] = "_"
            end
        end
        
        atoms["dihedrals"] = collect(dihedrals)
    end
    
    return atoms
end
=#

function _read_atoms_section(fileobj, natoms::Int, style::String = "full")
    type_in = Dict{Int, Int}()
    mol_id_in = Dict{Int, Int}()
    charge_in = Dict{Int, Float64}()
    pos_in = Dict{Int, Tuple{Float64, Float64, Float64}}()
    travel_in = Dict{Int, Tuple{Int, Int, Int}}()
    readline(fileobj)  # skip blank line just after `Atoms`
    
    for _ in 1:natoms
        line = readline(fileobj)
        fields = split(line)
        
        atom_id = parse(Int, fields[1])
        
        if style == "full" && length(fields) in (7, 10)
            type_in[atom_id] = parse(Int, fields[3])
            pos_in[atom_id] = (parse(Float64, fields[5]), parse(Float64, fields[6]), parse(Float64, fields[7]))
            mol_id_in[atom_id] = parse(Int, fields[2])
            charge_in[atom_id] = parse(Float64, fields[4])
            
            if length(fields) == 10
                travel_in[atom_id] = (parse(Int, fields[8]), parse(Int, fields[9]), parse(Int, fields[10]))
            end
        elseif style == "atomic" && length(fields) in (5, 8)
            type_in[atom_id] = parse(Int, fields[2])
            pos_in[atom_id] = (parse(Float64, fields[3]), parse(Float64, fields[4]), parse(Float64, fields[5]))
            
            if length(fields) == 8
                travel_in[atom_id] = (parse(Int, fields[6]), parse(Int, fields[7]), parse(Int, fields[8]))
            end
        elseif style in ["angle", "bond", "molecular"] && length(fields) in (6, 9)
            type_in[atom_id] = parse(Int, fields[3])
            pos_in[atom_id] = (parse(Float64, fields[4]), parse(Float64, fields[5]), parse(Float64, fields[6]))
            mol_id_in[atom_id] = parse(Int, fields[2])
            
            if length(fields) == 9
                travel_in[atom_id] = (parse(Int, fields[7]), parse(Int, fields[8]), parse(Int, fields[9]))
            end
        elseif style == "charge" && length(fields) in (6, 9)
            type_in[atom_id] = parse(Int, fields[2])
            pos_in[atom_id] = (parse(Float64, fields[3]), parse(Float64, fields[4]), parse(Float64, fields[5]))
            charge_in[atom_id] = parse(Float64, fields[6])
            
            if length(fields) == 9
                travel_in[atom_id] = (parse(Int, fields[7]), parse(Int, fields[8]), parse(Int, fields[9]))
            end
        else
            throw(RuntimeError("Style '$style' not supported or invalid number of fields $(length(fields))"))
        end
    end
    
    return mol_id_in, type_in, charge_in, pos_in, travel_in
end

function convert_cell_vectors(cell)
    println(cell)
    xhi = cell[1, 1]
    yhi = cell[2, 2]
    zhi = cell[3, 3]
    xy = cell[2, 1]
    xz = cell[3, 1]
    yz = cell[3, 2]

    return xhi, yhi, zhi, xy, xz, yz
end

function write_lammps_data(
    filename,
    atoms,
    specorder = nothing,
    velocities::Bool = false,
    units::String = "metal",
    atom_style::String = "full"
)
    # Assumes units in Metallic units, i.e. angstroms, eV, etc.
    fd = open(filename, "w")
    write(fd, "(written by ASE)\n\n")
    cell_vectors = atoms["cell"]
    symbols = atoms["symbols"]
    n_atoms = length(symbols)
    write(fd, "$n_atoms atoms\n")
    species = Dict()
    n = 1
    for i in 1:n_atoms
        if !(symbols[i] in keys(species))
            species[symbols[i]] = n
            n+=1
        end
    end
    
    n_atom_types = length(species)
    write(fd, "$n_atom_types atom types\n\n")

    xhi, yhi, zhi, xy, xz, yz = convert_cell_vectors(cell_vectors)

    @printf(fd, "0.0 %23.17g  xlo xhi\n", xhi)
    @printf(fd, "0.0 %23.17g  ylo yhi\n", yhi)
    @printf(fd, "0.0 %23.17g  zlo zhi\n", zhi)

    @printf(fd, "%23.17g %23.17g %23.17g  xy xz yz\n", xy, xz, yz)
    write(fd, "\n")

    fd, pos = _write_atoms_positions(fd, atoms,  species, symbols, atom_style, units)

    if (velocities)
        write(fd, "\n\nVelocities\n\n")
        vel = atoms["velocities"]
        for i in n_atoms
            @printf(fd, "%6d %23.17g %23.17g %23.17g\n", i + 1, vel[i,1], vel[i,2], vel[i,3])
        end
    end

    flush(fd)
end

function _write_atoms_positions(
    fd, atoms, species,
    symbols, atom_style::String, units::String
)
    write(fd, "Atoms # $atom_style\n\n")
    pos = atoms["positions"]
    if atom_style == "atomic"
        for (i, r) in enumerate(pos)
            s = species[symbols[i]]
            @printf(fd, "%6d %3d %23.17g %23.17g %23.17g\n", i + 1, s, pos[i,1], pos[i,2], pos[i,3])
        end
    elseif atom_style == "charge"
        if !haskey(atoms, "charges")
            charges = atoms["charges"]
        else
            charges = zeros(length(symbols))
        end
        for (i, (q, r)) in enumerate(zip(charges, pos))
            s = species[symbols[i]]
            @printf(fd, "%6d %3d %5d %23.17g %23.17g %23.17g\n", i + 1, s, q, pos[i,1], pos[i,2], pos[i,3])
        end
    elseif atom_style == "full"
        if !haskey(atoms, "charges")
            charges = atoms["charges"]
        else
            charges = zeros(length(symbols))
        end
        if haskey(atoms, "mol-id")
            molecules = atoms["mol-id"]
        else
            molecules = zeros(Int, length(atoms))
        end

        for (i, (m, q, r)) in enumerate(zip(molecules, charges, pos))
            s = species[symbols[i]]
            @printf(fd, "%6d %3d %3d %5d %23.17g %23.17g %23.17g\n", i + 1, m, s, q, pos[i,1], pos[i,2], pos[i,3])
        end
    else
        throw(NotImplementedError("Atom style '$atom_style' is not implemented."))
    end

    return fd, pos
end
######################################################################################################################

# Geometry Generator functions

######################################################################################################################

# Constants for graphene and moire cell

#####################################################################################################################

function reciprocal_lattice_vectors(a1, a2)
    a1 = [a1[1], a1[2], 0]
    a2 = [a2[1], a2[2], 0]
    a3 = [0, 0, 1]
    volume = dot(a1,cross(a2,a3)) 
    b1 = 2*pi * cross(a2,a3) / volume
    b2 = 2*pi * cross(a2,a1) / volume

    return b1[begin:2], b2[begin:2]
end

# lattice constant (angstrom)
A_C = 2.4683456
A_EDGE =  A_C/sqrt(3)

# moire information (angstrom)
D1_LAYER = 3.433333333
D2_LAYER = 0.027777778
D_AB = 3.35

# unit vector for atom system
A_UNITVEC_1 = [sqrt(3)*A_C/2 -A_C/2]
A_UNITVEC_2 = [sqrt(3)*A_C/2 A_C/2]
#A_UNITVEC_1 = [A_C 0]
#A_UNITVEC_2 = [sqrt(3)*A_C/2 A_C/2]
# reciprocal unit vector for atom system
A_G_UNITVEC_1 = [2*pi/(3*A_EDGE) -2*pi/(sqrt(3)*A_EDGE)]
A_G_UNITVEC_2 = [2*pi/(3*A_EDGE) 2*pi/(sqrt(3)*A_EDGE)]

# atom postion in graphene
ATOM_PSTN_1 = [0 0]
ATOM_PSTN_2 = [2*A_C/sqrt(3) 0]

#####################################################################################################################

function find_moire_int(theta, tol=1e-2)
    theta *= π / 180

    imax = 100
    for i in 0:imax-1
        theta_guess = asin((sqrt(3) * (2 * i + 1)) / (6 * i^2 + 6 * i + 2))
        if isapprox(theta, theta_guess, atol=tol)
            return i
        end
    end
end

function _set_moire_angle(n_moire::Int)
    angle_r = asin(sqrt(3) * (2 * n_moire + 1) / (6 * n_moire^2 + 6 * n_moire + 2))
    angle_d = angle_r / π * 180
    return (angle_r, angle_d)
end

function _set_rt_mtrx(theta::Float64)
    rt_mtrx = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    return rt_mtrx
end

function _set_moire(n_moire::Int)
    rt_angle_r, rt_angle_d = _set_moire_angle(n_moire)
    rt_mtrx = _set_rt_mtrx(rt_angle_r)
    rt_mtrx_half = _set_rt_mtrx(rt_angle_r / 2)

    m_unitvec_1 = (-n_moire * A_UNITVEC_1 + (2 * n_moire + 1) * A_UNITVEC_2) * rt_mtrx_half'
    m_unitvec_2 = (-(2 * n_moire + 1) * A_UNITVEC_1 + (n_moire + 1) * A_UNITVEC_2) * rt_mtrx_half'
    
    m_g_unitvec_1 = A_G_UNITVEC_1 * rt_mtrx_half' - A_G_UNITVEC_1 * rt_mtrx_half
    m_g_unitvec_2 = A_G_UNITVEC_2 * rt_mtrx_half' - A_G_UNITVEC_2 * rt_mtrx_half

    m_basis_vecs = Dict(
        "mu1" => m_unitvec_1,
        "mu2" => m_unitvec_2,
        "mg1" => m_g_unitvec_1,
        "mg2" => m_g_unitvec_2
    )

    high_symm_pnts = Dict(
        "gamma" => [0.0, 0.0],
        "k1" => (m_g_unitvec_1 + m_g_unitvec_2) / 3 + m_g_unitvec_2 / 3,
        "k2" => (m_g_unitvec_1 + m_g_unitvec_2) / 3 + m_g_unitvec_1 / 3,
        "m" => ((m_g_unitvec_1 + m_g_unitvec_2) / 3 + m_g_unitvec_2 / 3 + m_g_unitvec_1 / 3) / 2
    )

    return ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts)
end

function set_atom_pstn_list(n_moire::Int, corru::Bool=false)
    (rt_angle_r, _), m_basis_vecs, _ = _set_moire(n_moire)
    m_g_unitvec_1 = m_basis_vecs["mg1"]
    m_g_unitvec_2 = m_basis_vecs["mg2"]
    m_unitvec_1 = m_basis_vecs["mu1"]
    rt_mtrx_half = _set_rt_mtrx(rt_angle_r / 2)

    atom_b_pstn = ATOM_PSTN_2 - A_UNITVEC_1
    small_g_vec = hcat(m_g_unitvec_1, m_g_unitvec_2, -m_g_unitvec_1, -m_g_unitvec_2)
    ly = m_unitvec_1[2]
    n = trunc(Int,(2 * ly / A_C)) + 2
    delta = 0.0001
    z=30

    atom_pstn_list = []
    layer_index = []
    mol_id = []
    num_a1 = num_b1 = num_a2 = num_b2 = 0

    for ix in 0:n-1, iy in 0:n-1
        atom_pstn = -ix * A_UNITVEC_1 + iy * A_UNITVEC_2
        atom_pstn = atom_pstn * rt_mtrx_half'

        x = dot(atom_pstn, m_g_unitvec_1) / (2 * π)
        y = dot(atom_pstn, m_g_unitvec_2) / (2 * π)

        if (x > -delta) && (x < (1 - delta)) && (y > -delta) && (y < (1 - delta))
            #out_plane = D2_LAYER * sum(cos.(small_g_vec .* atom_pstn))
            #d = corru ? 0.5 * D1_LAYER + out_plane : 0.5 * D1_LAYER
            d = 0.5 * D1_LAYER +z/2
            atom = [atom_pstn[1], atom_pstn[2], d]

            push!(atom_pstn_list, atom)
            num_a1 += 1
            push!(layer_index,"B")
            push!(mol_id,1)
        end
    end

    for ix in 0:n-1, iy in 0:n-1
        atom_pstn = -ix * A_UNITVEC_1 + iy * A_UNITVEC_2 + atom_b_pstn
        atom_pstn = atom_pstn * rt_mtrx_half'

        x = dot(atom_pstn, m_g_unitvec_1) / (2 * π)
        y = dot(atom_pstn, m_g_unitvec_2) / (2 * π)

        if (x > -delta) && (x < (1 - delta)) && (y > -delta) && (y < (1 - delta))
            #out_plane = D2_LAYER * sum(cos.(small_g_vec .* atom_pstn))
            #d = corru ? 0.5 * D1_LAYER + out_plane : 0.5 * D1_LAYER
            d = 0.5 * D1_LAYER +z/2

            atom = [atom_pstn[1], atom_pstn[2], d]

            push!(atom_pstn_list, atom)
            num_b1 += 1
            push!(layer_index,"B")
            push!(mol_id,1)
        end
    end

    for ix in 0:n-1, iy in 0:n-1
        atom_pstn = -ix * A_UNITVEC_1 + iy * A_UNITVEC_2
        atom_pstn = atom_pstn * rt_mtrx_half

        x = dot(atom_pstn, m_g_unitvec_1) / (2 * π)
        y = dot(atom_pstn, m_g_unitvec_2) / (2 * π)

        if (x > -delta) && (x < (1 - delta)) && (y > -delta) && (y < (1 - delta))
            #out_plane = D2_LAYER * sum(cos.(small_g_vec .* atom_pstn))
            #d = corru ? -0.5 * D1_LAYER - out_plane : -0.5 * D1_LAYER
            d = -0.5 * D1_LAYER+z/2

            atom = [atom_pstn[1], atom_pstn[2], d]

            push!(atom_pstn_list, atom)
            num_a2 += 1
            push!(layer_index,"Ti")
            push!(mol_id,2)
        end
    end

    for ix in 0:n-1, iy in 0:n-1
        atom_pstn = -ix * A_UNITVEC_1 + iy * A_UNITVEC_2 + atom_b_pstn
        atom_pstn = atom_pstn * rt_mtrx_half

        x = dot(atom_pstn, m_g_unitvec_1) / (2 * π)
        y = dot(atom_pstn, m_g_unitvec_2) / (2 * π)

        if (x > -delta) && (x < (1 - delta)) && (y > -delta) && (y < (1 - delta))
            #out_plane = D2_LAYER * sum(cos.(small_g_vec .* atom_pstn))
            #d = corru ? -0.5 * D1_LAYER - out_plane : -0.5 * D1_LAYER
            d = -0.5 * D1_LAYER+z/2

            atom = [atom_pstn[1], atom_pstn[2], d]

            push!(atom_pstn_list, atom)
            num_b2 += 1
            push!(layer_index,"Ti")
            push!(mol_id,2)
        end
    end
    
    cell = [m_basis_vecs["mu1"][1] m_basis_vecs["mu1"][2] 0;
                    m_basis_vecs["mu2"][1] m_basis_vecs["mu2"][2] 0;
                    0 0 z]
    natoms = length(layer_index)
    atoms = Dict("positions"=>hcat(atom_pstn_list...)',"mol-id"=>mol_id,
             "symbols"=>hcat(layer_index),"cell"=>cell,"charges"=>zeros(natoms))
    return atoms
end

#=mutable struct Atoms
    positions::Vector{Vector{Float64}}  # Array of position vectors (x, y, z)
    masses::masses
    cell::Matrix{Float64}  # 3x3 matrix representing the unit cell vectors
    pbc::pbc,
    arrays::arrays
    atom_types::atom_types
end
=#

function get_tBLG_atoms(twist_angle,z=30)
    n_moire = find_moire_int(twist_angle)
    ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts) = _set_moire(n_moire::Int)
    cell = [m_basis_vecs["mu1"][1] m_basis_vecs["mu1"][2] 0;
                    m_basis_vecs["mu2"][1] m_basis_vecs["mu2"][2] 0;
                    0 0 z]
    atoms = set_atom_pstn_list(n_moire)
    # Create Atoms
    #atom = Atoms(atom_pstn_list,masses,cell,pbc,arrays,layer_index)
    return atoms
end
