using LinearAlgebra
using CUDA
using SparseArrays
using CUDA.CUSPARSE

function gen_ham_ovrlp(natoms)
    Ham = spzeros(Complex{Float64},natoms,natoms)
    Overlap = spzeros(Complex{Float64},natoms,natoms)
    for i in 1:natoms-1
        Overlap[i,i] = 1

        phase = exp(1im*0.5)
        Ham[i,i+1] +=  phase
        Ham[i+1,i] += conj(phase)

    end

    return Ham, Hermitian(Overlap)
end

#write test code
natoms = 10
ham,ovrlap = gen_ham_ovrlp(natoms)
#CuArrays.allowscalar(false)  # Disallow scalar operations on GPU
#device = CuDevice(device_num)
#CuArrays.init(device)
#gpu_matrix = CuSparseMatrixCSR(ham) #CuSparseVector
gpu_matrix = CuArray{Int}(undef, (natoms,natoms))
eigenvalues, eigenvectors = eigen(gpu_matrix)
cpu_eigenvalues = Array(eigenvalues)
cpu_eigenvectors = Array(eigenvectors)
println("eigenvalues = ",cpu_eigenvalues)