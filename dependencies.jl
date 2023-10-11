import Pkg
Pkg.add("CEnum")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("Distributed")
Pkg.add("Plots")
Pkg.add("Preferences")
Pkg.add("ForwardDiff")
Pkg.add("CUDA")
Pkg.add("SparseArrays")
Pkg.add("HDF5")
Pkg.add("DataFrames")
Pkg.add("PyCall")
Pkg.add("Polynomials")

run_all_from_julia = false
if run_all_from_julia
    Pkg.add("MPI")
    Pkg.add("LAMMPS_jll")
    Pkg.add("LAMMPS")
    using LAMMPS
    shared_lib_dir ="/global/homes/d/dpalmer3/lammps/src/liblammps.so" 
    LAMMPS.set_library!(shared_lib_dir)
end
